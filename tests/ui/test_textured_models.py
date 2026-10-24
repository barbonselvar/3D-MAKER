"""P0-01 / P0-02: textured and imported models must reach the user unmodified."""
import hashlib
import json
import struct
from pathlib import Path

import pytest

pytestmark = pytest.mark.ui
FIXTURE = Path(__file__).resolve().parents[1] / 'fixtures' / 'textured_cube.glb'


def glb_json(data: bytes) -> dict:
    length = struct.unpack_from('<I', data, 12)[0]
    return json.loads(data[20:20 + length])


def import_fixture(page):
    page.set_input_files('#importfile', str(FIXTURE))
    page.wait_for_function("document.querySelector('#stats').textContent.includes('Imported GLB')")


def download(page, fmt) -> bytes:
    with page.expect_download() as info:
        page.click(f'[data-export={fmt}]')
    return Path(info.value.path()).read_bytes()


def test_colour_tool_disabled_for_textured_model(page):
    import_fixture(page)
    assert page.is_disabled('#color')
    # A programmatic click on a disabled button must not run the handler either.
    page.evaluate("document.querySelector('#color').click()")
    data = download(page, 'glb')
    material = glb_json(data)['materials'][0]['pbrMetallicRoughness']
    assert material.get('baseColorFactor', [1, 1, 1, 1]) == [1, 1, 1, 1]
    assert page.errors == []


def test_colour_tool_still_works_for_local_modes(page):
    page.select_option('#mode', 'table')
    page.click('#generate')
    page.wait_for_function("document.querySelector('#modelname').textContent === 'Editable table'")
    assert page.is_enabled('#color')
    page.click('#color')
    data = download(page, 'glb')
    factors = [m['pbrMetallicRoughness'].get('baseColorFactor') for m in glb_json(data)['materials']]
    assert any(f and f[:3] != [1, 1, 1] for f in factors)


def test_imported_glb_downloads_original_bytes(page):
    import_fixture(page)
    data = download(page, 'glb')
    assert hashlib.sha256(data).hexdigest() == hashlib.sha256(FIXTURE.read_bytes()).hexdigest()


def obj_vertices(text: str):
    return [tuple(map(float, line.split()[1:4])) for line in text.splitlines() if line.startswith('v ')]


def test_derived_export_keeps_original_units_and_origin(page):
    # The fixture cube spans x 2.5..3.5 (1 m, offset from origin); the viewer rescales it to 2.6 units.
    import_fixture(page)
    xs = [v[0] for v in obj_vertices(download(page, 'obj').decode())]
    assert min(xs) == pytest.approx(2.5, abs=1e-4)
    assert max(xs) == pytest.approx(3.5, abs=1e-4)


def test_viewer_fit_is_restored_after_export(page):
    import_fixture(page)
    before = page.evaluate("document.querySelector('#stats').textContent")
    download(page, 'stl')
    assert page.evaluate("document.querySelector('#stats').textContent") == before
    assert page.errors == []
