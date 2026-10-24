"""Minimal GLB reader shared by validation and tests (glTF 2.0 binary container)."""
from __future__ import annotations

import json
import struct

import numpy as np

COMPONENT = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
WIDTH = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4, 'MAT4': 16}


class GlbError(ValueError):
    pass


def read_glb(data: bytes) -> tuple[dict, bytes]:
    if len(data) < 20 or data[:4] != b'glTF':
        raise GlbError('not a GLB file')
    version, length = struct.unpack_from('<II', data, 4)
    if version != 2 or length > len(data):
        raise GlbError('unsupported or truncated GLB')
    json_len, json_type = struct.unpack_from('<II', data, 12)
    if json_type != 0x4E4F534A:
        raise GlbError('first GLB chunk is not JSON')
    doc = json.loads(data[20:20 + json_len])
    binary = b''
    offset = 20 + json_len
    if offset + 8 <= length:
        bin_len, bin_type = struct.unpack_from('<II', data, offset)
        if bin_type == 0x004E4942:
            binary = data[offset + 8: offset + 8 + bin_len]
    return doc, binary


def buffer_view_bytes(doc: dict, binary: bytes, index: int) -> bytes:
    view = doc['bufferViews'][index]
    start = view.get('byteOffset', 0)
    return binary[start:start + view['byteLength']]


def read_accessor(doc: dict, binary: bytes, index: int) -> np.ndarray:
    acc = doc['accessors'][index]
    dtype, width = COMPONENT[acc['componentType']], WIDTH[acc['type']]
    if 'bufferView' not in acc:
        return np.zeros((acc['count'], width), dtype=dtype)
    view = doc['bufferViews'][acc['bufferView']]
    item = np.dtype(dtype).itemsize * width
    stride = view.get('byteStride') or item
    start = view.get('byteOffset', 0) + acc.get('byteOffset', 0)
    raw = np.frombuffer(binary, dtype=np.uint8, count=stride * (acc['count'] - 1) + item if acc['count'] else 0,
                        offset=start)
    rows = np.lib.stride_tricks.as_strided(raw, shape=(acc['count'], item), strides=(stride, 1))
    out = np.ascontiguousarray(rows).view(dtype).reshape(acc['count'], width)
    if acc.get('normalized'):
        out = out.astype(np.float64) / np.iinfo(dtype).max
    return out
