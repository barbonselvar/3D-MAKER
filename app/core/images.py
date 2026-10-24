"""Upload validation and sanitisation (blueprint pipeline stage 1, security section).

Type is sniffed from magic bytes, never from the filename or client MIME.
Images are decoded with a pixel cap, EXIF orientation is applied, colour is
converted to sRGB, and the result is re-encoded as PNG so no EXIF/GPS or
other metadata is ever stored.
"""
from __future__ import annotations

import hashlib
import io
import warnings
from dataclasses import dataclass

from PIL import Image, ImageCms, ImageOps

MAX_BYTES = 12 * 1024 * 1024
MAX_PIXELS = 40_000_000
Image.MAX_IMAGE_PIXELS = MAX_PIXELS  # Pillow's decompression-bomb guard

_FORMATS = {'image/png': 'PNG', 'image/jpeg': 'JPEG', 'image/webp': 'WEBP'}


class InvalidImage(ValueError):
    pass


@dataclass
class CleanImage:
    png: bytes
    sha256: str
    width: int
    height: int
    has_alpha: bool
    original_mime: str


def sniff(data: bytes) -> str | None:
    if data.startswith(b'\x89PNG\r\n\x1a\n'):
        return 'image/png'
    if data.startswith(b'\xff\xd8\xff'):
        return 'image/jpeg'
    if len(data) >= 12 and data[:4] == b'RIFF' and data[8:12] == b'WEBP':
        return 'image/webp'
    return None


def _to_srgb(im: Image.Image) -> Image.Image:
    icc = im.info.get('icc_profile')
    if not icc:
        return im
    try:
        src = ImageCms.ImageCmsProfile(io.BytesIO(icc))
        mode = 'RGBA' if 'A' in im.getbands() else 'RGB'
        return ImageCms.profileToProfile(im.convert(mode), src, ImageCms.createProfile('sRGB'), outputMode=mode)
    except (ImageCms.PyCMSError, OSError, ValueError):
        return im  # unreadable profile: keep pixels as-is rather than reject


def sanitize(data: bytes) -> CleanImage:
    if len(data) > MAX_BYTES:
        raise InvalidImage('Image is larger than 12 MB.')
    mime = sniff(data)
    if mime is None:
        raise InvalidImage('Only PNG, JPEG and WEBP images are accepted.')
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            im = Image.open(io.BytesIO(data))
            if im.format != _FORMATS[mime]:
                raise InvalidImage('File content does not match its image type.')
            if im.width * im.height > MAX_PIXELS:
                raise InvalidImage('Image is larger than 40 megapixels.')
            im.seek(0)  # first frame of animated PNG/WEBP
            im.load()
    except InvalidImage:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise InvalidImage('Image is larger than 40 megapixels.') from None
    except Exception:  # noqa: BLE001 - any decoder failure means a corrupt/hostile file
        raise InvalidImage('Image could not be decoded.') from None
    im = ImageOps.exif_transpose(im)
    has_alpha = im.mode in ('RGBA', 'LA', 'PA') or (im.mode == 'P' and 'transparency' in im.info)
    im = _to_srgb(im.convert('RGBA' if has_alpha else 'RGB'))
    if has_alpha and im.getchannel('A').getextrema()[0] == 255:
        im, has_alpha = im.convert('RGB'), False  # fully opaque alpha carries no mask
    im.info = {}  # Pillow's PNG writer re-emits info['icc_profile'] etc. unless cleared
    out = io.BytesIO()
    im.save(out, 'PNG')
    png = out.getvalue()
    return CleanImage(png, hashlib.sha256(png).hexdigest(), im.width, im.height, has_alpha, mime)
