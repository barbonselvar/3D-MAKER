"""Foreground segmentation (blueprint pipeline stage 2, ticket P1-03).

Configurable so the default stays commercially usable. RMBG-2.0 (CC BY-NC 4.0)
is only allowed with SEGMENTER_ALLOW_NONCOMMERCIAL=1. Inputs that already have
a real alpha mask skip segmentation entirely.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageOps

from .engines.base import EngineError

MIN_COVERAGE, MAX_COVERAGE = 0.02, 0.95


@dataclass(frozen=True)
class SegmenterSpec:
    name: str
    model: str  # Hugging Face model id
    licence: str
    commercial: bool | None  # None = not verified


SEGMENTERS = {
    'birefnet': SegmenterSpec('birefnet', 'ZhengPeng7/BiRefNet',
                              'BiRefNet general weights: licence NOT VERIFIED by this project (code is MIT); '
                              'confirm on the model card before commercial use', None),
    'rmbg-2.0': SegmenterSpec('rmbg-2.0', 'briaai/RMBG-2.0', 'CC BY-NC 4.0 (non-commercial only)', False),
}


def choose(name: str | None = None, allow_noncommercial: bool | None = None) -> SegmenterSpec:
    name = name or os.environ.get('SEGMENTER', 'birefnet')
    if allow_noncommercial is None:
        allow_noncommercial = os.environ.get('SEGMENTER_ALLOW_NONCOMMERCIAL') == '1'
    if name not in SEGMENTERS:
        raise ValueError(f'unknown segmenter {name!r}; choose one of {", ".join(SEGMENTERS)}')
    spec = SEGMENTERS[name]
    if spec.commercial is False and not allow_noncommercial:
        raise ValueError(f'{spec.model} is licensed {spec.licence}; set SEGMENTER_ALLOW_NONCOMMERCIAL=1 to use it')
    return spec


class BiRefNetSegmenter:
    """BiRefNet-architecture matting (same pre/post-processing as TRELLIS.2's rembg wrapper). Needs a GPU stack."""

    def __init__(self, spec: SegmenterSpec, device: str = 'cuda'):
        self.spec, self.device, self.model = spec, device, None

    def load(self) -> BiRefNetSegmenter:
        from torchvision import transforms  # noqa: PLC0415 - GPU image only
        from transformers import AutoModelForImageSegmentation  # noqa: PLC0415

        revision = os.environ.get('SEGMENTER_REVISION')  # pin to a commit hash in production
        self.model = AutoModelForImageSegmentation.from_pretrained(self.spec.model, trust_remote_code=True,
                                                                   revision=revision).eval().to(self.device)
        self.revision = revision
        self.transform = transforms.Compose([
            transforms.Resize((1024, 1024)), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        return self

    def mask(self, rgb: Image.Image) -> Image.Image:
        import torch  # noqa: PLC0415
        from torchvision import transforms  # noqa: PLC0415

        batch = self.transform(rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            pred = self.model(batch)[-1].sigmoid().cpu()[0].squeeze()
        return transforms.ToPILImage()(pred).resize(rgb.size)


def has_real_alpha(image: Image.Image) -> bool:
    return image.mode == 'RGBA' and image.getchannel('A').getextrema()[0] < 255


def segment(image: Image.Image, segmenter, contrast_boost: bool = False) -> tuple[Image.Image, dict]:
    """Return an RGBA image whose alpha isolates the subject, plus a record for the manifest.

    contrast_boost feeds an autocontrast copy to the segmenter only (helps dark subjects such as the
    ninja reference); the colours used for texturing are never modified.
    """
    if has_real_alpha(image):
        rgba, info = image.convert('RGBA'), {'method': 'input alpha (segmentation skipped)'}
    else:
        rgb = image.convert('RGB')
        mask = segmenter.mask(ImageOps.autocontrast(rgb, cutoff=1) if contrast_boost else rgb)
        rgba = rgb.copy()
        rgba.putalpha(mask.convert('L'))
        spec = segmenter.spec
        info = {'method': spec.name, 'model': spec.model, 'revision': getattr(segmenter, 'revision', None),
                'licence': spec.licence, 'contrast_boost': contrast_boost}
    coverage = float((np.asarray(rgba.getchannel('A')) > 0.8 * 255).mean())
    info['coverage'] = round(coverage, 4)
    if coverage < MIN_COVERAGE:
        raise EngineError('SEGMENTATION_EMPTY', 'No clear subject was found in the image. Use a photo where the object '
                          'fills more of the frame, or a PNG with a transparent background.')
    if coverage > MAX_COVERAGE:
        raise EngineError('SEGMENTATION_EMPTY', 'The background could not be separated from the subject. Use a plainer '
                          'background or a PNG with a transparent background.')
    return rgba, info
