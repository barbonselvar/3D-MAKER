"""P1-03 segmenter rules and the stage-6 cleanup (CPU only; the real BiRefNet model needs the GPU image)."""
from pathlib import Path

import numpy as np
import pytest
import trimesh
from PIL import Image

from worker import postprocess, segment
from worker.engines.base import EngineError

FIX = Path(__file__).parent / 'fixtures'


class FakeSegmenter:
    """MOCK segmenter: returns a fixed mask; stands in for the BiRefNet model."""

    def __init__(self, coverage):
        self.coverage, self.calls = coverage, []
        self.spec = segment.SEGMENTERS['birefnet']

    def mask(self, rgb):
        self.calls.append(rgb)
        w, h = rgb.size
        m = np.zeros((h, w), np.uint8)
        m[: int(round(h * self.coverage)), :] = 255
        return Image.fromarray(m)


def test_default_segmenter_is_not_rmbg(monkeypatch):
    monkeypatch.delenv('SEGMENTER', raising=False)
    monkeypatch.delenv('SEGMENTER_ALLOW_NONCOMMERCIAL', raising=False)
    assert segment.choose().model == 'ZhengPeng7/BiRefNet'


def test_rmbg_requires_noncommercial_flag(monkeypatch):
    monkeypatch.delenv('SEGMENTER_ALLOW_NONCOMMERCIAL', raising=False)
    with pytest.raises(ValueError, match='SEGMENTER_ALLOW_NONCOMMERCIAL'):
        segment.choose('rmbg-2.0')
    monkeypatch.setenv('SEGMENTER_ALLOW_NONCOMMERCIAL', '1')
    assert segment.choose('rmbg-2.0').licence.startswith('CC BY-NC')


def test_alpha_input_bypasses_segmenter():
    rgba = Image.new('RGBA', (100, 100), (255, 0, 0, 0))
    rgba.paste((255, 0, 0, 255), (20, 20, 80, 80))
    seg = FakeSegmenter(0.5)
    out, info = segment.segment(rgba, seg)
    assert seg.calls == [] and info['method'].startswith('input alpha')
    assert info['coverage'] == pytest.approx(0.36)


def test_segmenter_mask_applied_and_recorded():
    out, info = segment.segment(Image.new('RGB', (100, 100), (10, 10, 10)), FakeSegmenter(0.5))
    assert out.mode == 'RGBA' and info['model'] == 'ZhengPeng7/BiRefNet' and info['coverage'] == 0.5
    assert 'NOT VERIFIED' in info['licence']


@pytest.mark.parametrize('coverage', [0.01, 0.97])
def test_empty_or_full_masks_rejected(coverage):
    with pytest.raises(EngineError) as e:
        segment.segment(Image.new('RGB', (100, 100)), FakeSegmenter(coverage))
    assert e.value.code == 'SEGMENTATION_EMPTY'


def test_contrast_boost_only_affects_segmenter_input():
    dark = Image.new('RGB', (100, 100), (20, 20, 20))
    dark.paste((40, 40, 40), (0, 0, 50, 100))
    seg = FakeSegmenter(0.5)
    out, _ = segment.segment(dark, seg, contrast_boost=True)
    assert seg.calls[0].getextrema() != dark.getextrema()  # boosted copy went to the model
    assert out.convert('RGB').getextrema() == dark.getextrema()  # texture colours untouched


def load(name):
    from worker.validate import load_mesh
    return load_mesh((FIX / name).read_bytes())


def test_cleanup_drops_tiny_debris_keeps_uvs():
    big = trimesh.creation.box(extents=(1, 1, 1))
    speck = trimesh.creation.box(extents=(0.01, 0.01, 0.01))
    speck.apply_translation([3, 0, 0])
    mesh = trimesh.util.concatenate([big, speck])
    mesh.visual = trimesh.visual.TextureVisuals(uv=np.random.default_rng(0).random((len(mesh.vertices), 2)))
    cleaned, report = postprocess.cleanup(mesh)
    assert report == {'removed_components': 1, 'removed_faces': 12, 'components_kept': 1}
    assert len(cleaned.faces) == 12 and len(cleaned.visual.uv) == len(cleaned.vertices)


def test_cleanup_keeps_large_parts_and_unwelded_islands():
    two_cubes = load('bad_fragmented.glb')  # two equal parts: both kept
    assert postprocess.cleanup(two_cubes)[1]['removed_components'] == 0
    cube = load('pbr_cube.glb')  # 12 unwelded faces = 1 part, not 12 specks
    assert postprocess.cleanup(cube)[1] == {'removed_components': 0, 'removed_faces': 0, 'components_kept': 1}
