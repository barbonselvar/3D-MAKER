"""The public website as Vercel serves it: app/static only, no backend, no GPU (see vercel.json, .vercelignore).

A plain static file server stands in for Vercel's CDN; every off-site request is blocked.
"""
import functools
import http.server
import threading
from pathlib import Path

import pytest

pytestmark = pytest.mark.ui
STATIC = Path(__file__).resolve().parents[2] / 'app' / 'static'


@pytest.fixture(scope='module')
def static_site():
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(STATIC))
    handler.log_message = lambda *a: None
    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f'http://127.0.0.1:{server.server_address[1]}'
    server.shutdown()


def open_site(browser, url, width, height):
    context = browser.new_context(viewport={'width': width, 'height': height}, accept_downloads=True)
    page = context.new_page()
    page.set_default_timeout(10_000)
    page.errors = []
    page.on('pageerror', lambda e: page.errors.append(str(e)))
    page.route('**/*', lambda r: r.continue_() if r.request.url.startswith(url) else r.abort())
    page.goto(url + '/')
    page.wait_for_function("document.querySelector('#status').textContent.length > 0")
    return context, page


@pytest.mark.parametrize('width,height', [(1280, 900), (390, 844)], ids=['desktop', 'mobile'])
def test_public_site_works_without_backend(browser, static_site, width, height):
    context, page = open_site(browser, static_site, width, height)
    try:
        # Local tools work entirely in the browser.
        page.select_option('#mode', 'table')
        page.click('#generate')
        page.wait_for_function("document.querySelector('#modelname').textContent === 'Editable table'")
        with page.expect_download() as info:
            page.click('[data-export=glb]')
        assert Path(info.value.path()).read_bytes()[:4] == b'glTF'
        # AI: Local GPU explains it runs in the installed app; the public demo stays available with consent.
        page.select_option('#mode', 'ai')
        page.click('[data-sample=vase]')
        page.wait_for_function("document.querySelector('#status').textContent.startsWith('Ready')")
        page.click('#generate')
        page.wait_for_selector('#genpanel[open]')
        page.wait_for_function("!document.querySelector('#int-status').textContent.startsWith('Checking')")
        assert 'runs in 3D-MAKER installed on your own computer' in page.text_content('#int-status')
        assert page.get_attribute('#int-status a', 'href').endswith('docs/WINDOWS.md')
        assert page.is_disabled('#start-internal')
        assert page.is_enabled('input[value="external:hf-demo"]') and page.is_disabled('#start-external')
        page.check('#cloudconsent')
        assert page.is_enabled('#start-external')
        assert page.evaluate('document.documentElement.scrollWidth') <= width
        assert page.errors == []
    finally:
        context.close()


def test_no_backend_or_private_routes_on_static_site(static_site):
    import urllib.error
    import urllib.request
    for path in ('/api/health', '/api/v1/gpu-targets', '/internal/v1/claim', '/main.py', '/../main.py'):
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(static_site + path)
        assert e.value.code == 404, path
