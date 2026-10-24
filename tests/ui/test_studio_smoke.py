"""Smoke tests for the existing studio modes (regression baseline for P1-09)."""
import pytest

pytestmark = pytest.mark.ui


def test_page_boots_without_errors(page):
    page.wait_for_timeout(300)
    assert page.errors == []
    assert page.locator('#generate').is_visible()


def test_silhouette_sample_builds_mesh(page):
    page.select_option('#mode', 'silhouette')
    page.click('[data-sample=vase]')
    page.wait_for_function("document.querySelector('#status').textContent.startsWith('Ready')")
    page.click('#generate')
    page.wait_for_function("document.querySelector('#stats').textContent.includes('Silhouette extrusion')")
    assert 'triangles' in page.text_content('#stats')
    assert page.is_enabled('[data-export=glb]')
    assert page.errors == []


def test_table_builder(page):
    page.select_option('#mode', 'table')
    page.click('#generate')
    page.wait_for_function("document.querySelector('#modelname').textContent === 'Editable table'")
    assert 'not AI' in page.text_content('#stats')


def test_ai_mode_requires_consent(page):
    page.click('[data-sample=bolt]')
    page.wait_for_function("document.querySelector('#status').textContent.startsWith('Ready')")
    page.click('#generate')
    assert 'image-sharing box' in page.text_content('#status')


@pytest.mark.parametrize('fmt', ['glb', 'obj', 'stl', 'ply'])
def test_local_exports_download(page, fmt):
    page.select_option('#mode', 'table')
    page.click('#generate')
    page.wait_for_function("document.querySelector('#modelname').textContent === 'Editable table'")
    with page.expect_download() as info:
        page.click(f'[data-export={fmt}]')
    path = info.value.path()
    data = open(path, 'rb').read()
    assert len(data) > 100
    if fmt == 'glb':
        assert data[:4] == b'glTF'
