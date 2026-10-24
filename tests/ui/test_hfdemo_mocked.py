"""HF TRELLIS.2 demo flow against a MOCK gradio client.

MOCK: huggingface.co is not reachable from CI or this dev environment, so the
vendored @gradio/client module is replaced in the browser by a test double.
These tests prove the studio's handling of the demo protocol (stages, consent,
errors, original bytes); they do NOT prove the real Space works.
"""
import hashlib
from pathlib import Path

import pytest

pytestmark = pytest.mark.ui
FIXTURE = Path(__file__).resolve().parents[1] / 'fixtures' / 'textured_cube.glb'

MOCK_CLIENT = """
// MOCK @gradio/client for tests only.
window.__calls = [];
function job(endpoint, args) {
  window.__calls.push([endpoint, args.length]);
  const fail = window.__mockFail === endpoint;
  const data = endpoint === '/preprocess_image' ? ['prepped']
    : endpoint === '/extract_glb' ? [{url: 'https://mock.invalid/model.glb'}] : ['ok'];
  const msgs = fail ? [{type: 'status', stage: 'error', message: 'mock failure'}]
    : [{type: 'status', stage: 'pending', position: 2}, {type: 'data', data}];
  return {cancel() {}, async *[Symbol.asyncIterator]() { for (const m of msgs) yield m; }};
}
export const Client = {async connect(space) { window.__space = space; return {submit: job, close() {}}; }};
export const handle_file = (blob) => blob;
"""


@pytest.fixture()
def mocked(page):
    page.route('**/vendor/gradio-client@2.7.1/index.min.js',
               lambda r: r.fulfill(body=MOCK_CLIENT, content_type='text/javascript'))
    page.route('https://mock.invalid/**', lambda r: r.fulfill(body=FIXTURE.read_bytes(), content_type='model/gltf-binary'))
    page.click('[data-sample=bolt]')
    page.wait_for_function("document.querySelector('#status').textContent.startsWith('Ready')")
    return page


def start_demo(page):
    page.click('#generate')
    page.wait_for_selector('#genpanel[open]')
    page.check('input[value="external:hf-demo"]')
    page.check('#cloudconsent')
    page.click('#start-external')


def test_demo_flow_loads_model_and_keeps_original_bytes(mocked):
    start_demo(mocked)
    mocked.wait_for_function("document.querySelector('#stats').textContent.includes('TRELLIS.2 AI')")
    assert mocked.evaluate('window.__space') == 'microsoft/TRELLIS.2'
    calls = mocked.evaluate('window.__calls')
    assert [c[0] for c in calls] == ['/start_session', '/preprocess_image', '/image_to_3d', '/extract_glb']
    assert dict(calls)['/image_to_3d'] == 15
    assert mocked.is_disabled('#color')
    with mocked.expect_download() as info:
        mocked.click('[data-export=glb]')
    data = Path(info.value.path()).read_bytes()
    assert hashlib.sha256(data).digest() == hashlib.sha256(FIXTURE.read_bytes()).digest()
    assert mocked.errors == []


def test_demo_error_never_substitutes_a_local_mesh(mocked):
    mocked.evaluate("window.__mockFail = '/image_to_3d'")
    before = mocked.text_content('#stats')
    start_demo(mocked)
    mocked.wait_for_function("document.querySelector('#status').textContent.startsWith('AI unavailable')")
    assert 'No local substitute was generated' in mocked.text_content('#status')
    assert mocked.text_content('#stats') == before
    assert mocked.is_enabled('#generate')
