"""P1-08: validation gates V1-V4 on good and known-bad GLB fixtures."""
import os
import stat
from pathlib import Path

import pytest

from worker import validate

FIX = Path(__file__).parent / 'fixtures'


def run(name, **kw):
    kw.setdefault('face_budget', 200_000)
    kw.setdefault('texture_size', 1024)
    kw.setdefault('blender', '')
    return validate.validate_glb((FIX / name).read_bytes(), **kw)


def gates(report):
    return {g.id: g for g in report.gates}


def test_v1_validator_is_installed():
    # CI installs it with `npm ci --prefix worker`; a skipped V1 would make every report fail.
    assert gates(run('pbr_cube.glb'))['V1'].status == 'passed'


def test_good_pbr_fixture_passes():
    report = run('pbr_cube.glb')
    assert report.passed, report.to_dict()
    g = gates(report)
    assert g['V3'].details['components'] == 1  # unwelded UV-seam vertices are not separate parts
    assert g['V2'].status == 'skipped'


def test_textured_fixture_needs_pbr_flag():
    assert not run('textured_cube.glb', texture_size=64).passed  # no metallic-roughness texture
    assert run('textured_cube.glb', texture_size=64, require_pbr=False).passed


@pytest.mark.parametrize('name,gate,phrase', [
    ('bad_nan.glb', 'V3', 'NaN'),
    ('bad_nan.glb', 'V1', 'ACCESSOR_INVALID_FLOAT'),
    ('bad_uv_range.glb', 'V4', 'outside 0-1'),
    ('bad_webp.glb', 'V4', 'WebP'),
    ('bad_truncated.glb', 'V1', 'GLB_UNEXPECTED_END_OF_CHUNK_DATA'),
    ('bad_truncated.glb', 'V4', 'could not be parsed'),
    ('bad_fragmented.glb', 'V3', 'largest connected part covers 50%'),
    ('bad_degenerate.glb', 'V3', 'degenerate'),
    ('bad_untextured.glb', 'V4', 'no base colour texture'),
])
def test_bad_fixtures_fail_with_the_right_gate(name, gate, phrase):
    report = run(name)
    assert not report.passed
    assert gates(report)[gate].status == 'failed'
    assert phrase in gates(report)[gate].message


def test_multi_object_allows_fragmented():
    assert run('bad_fragmented.glb', multi_object=True).passed


def test_face_budget():
    report = run('pbr_cube.glb', face_budget=10)
    assert 'exceeds the budget of 10' in gates(report)['V3'].message


def test_texture_size_mismatch():
    report = run('pbr_cube.glb', texture_size=2048)
    assert 'expected 2048x2048' in gates(report)['V4'].message


needs_posix = pytest.mark.skipif(os.name == 'nt', reason='fake Blender is a /bin/sh script')


def fake_blender(tmp_path, line, code=0):
    """MOCK Blender: prints a canned import_check report. Tests the wrapper only, not Blender."""
    script = tmp_path / 'fake-blender'
    script.write_text(f"#!/bin/sh\necho 'Blender 4.2.0 (fake)'\necho '{line}'\nexit {code}\n", encoding='utf-8')
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return str(script)


@needs_posix
def test_v2_wrapper_pass_and_count_mismatch(tmp_path):
    ok = 'IMPORT_CHECK_JSON:{"ok": true, "mesh_objects": 1, "triangles": 12, "materials": 1, "images": 2, "missing_images": []}'
    assert gates(run('pbr_cube.glb', blender=fake_blender(tmp_path, ok)))['V2'].status == 'passed'
    bad = ok.replace('"triangles": 12', '"triangles": 10')
    g = gates(run('pbr_cube.glb', blender=fake_blender(tmp_path, bad)))['V2']
    assert g.status == 'failed' and 'triangle count 10 != 12' in g.message


@needs_posix
def test_v2_wrapper_import_failure(tmp_path):
    line = 'IMPORT_CHECK_JSON:{"ok": false, "error": "bad glTF"}'
    report = run('pbr_cube.glb', blender=fake_blender(tmp_path, line, code=1))
    assert not report.passed and gates(report)['V2'].message == 'bad glTF'


def test_v2_required_when_configured(monkeypatch):
    monkeypatch.setenv('VALIDATION_REQUIRE_BLENDER', '1')
    report = run('pbr_cube.glb')
    assert not report.passed
    assert [g.id for g in report.failures()] == ['V2']


@pytest.mark.skipif(not os.environ.get('BLENDER_BIN'), reason='real Blender not installed')
def test_v2_real_blender():
    report = run('pbr_cube.glb', blender=os.environ['BLENDER_BIN'], require_blender=True)
    assert gates(report)['V2'].status == 'passed', report.to_dict()


def test_sf3d_style_output_passes_basecolor_normal_profile():
    report = run('sf3d_style_512.glb', texture_size=512, profile='basecolor_normal')
    assert report.passed, report.to_dict()
    textures = {t['role']: t for t in gates(report)['V4'].details['textures']}
    assert textures['material 0 base colour']['mime'] == 'image/jpeg'
    assert textures['material 0 normal']['size'] == [512, 512]


def test_sf3d_style_output_fails_trellis_profile():
    g = gates(run('sf3d_style_512.glb', texture_size=512, profile='pbr_textures'))['V4']
    assert g.status == 'failed' and 'no metallic-roughness texture' in g.message


def test_basecolor_normal_profile_requires_normal_map():
    g = gates(run('pbr_cube.glb', profile='basecolor_normal'))['V4']
    assert g.status == 'failed' and 'no normal texture' in g.message
