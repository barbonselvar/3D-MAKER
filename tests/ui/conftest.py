"""Live-server + Chromium fixtures for UI tests.

Every request that does not go to the local test server is aborted, so the
tests also prove the studio runs without any CDN.
"""
import os
import socket
import threading
import time

import pytest
import uvicorn

if os.path.isdir('/opt/pw-browsers'):
    os.environ.setdefault('PLAYWRIGHT_BROWSERS_PATH', '/opt/pw-browsers')

from playwright.sync_api import sync_playwright  # noqa: E402


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


@pytest.fixture(scope='session')
def live_server():
    from app.main import app

    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(app, host='127.0.0.1', port=port, log_level='warning'))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    yield f'http://127.0.0.1:{port}'
    server.should_exit = True
    thread.join(5)


@pytest.fixture(scope='session')
def browser():
    with sync_playwright() as p:
        b = p.chromium.launch(args=['--use-angle=swiftshader', '--enable-unsafe-swiftshader'])
        yield b
        b.close()


@pytest.fixture()
def page(browser, live_server):
    context = browser.new_context(accept_downloads=True, viewport={'width': 1280, 'height': 900})
    pg = context.new_page()
    pg.set_default_timeout(10_000)
    pg.errors = []
    pg.on('pageerror', lambda e: pg.errors.append(str(e)))
    pg.route('**/*', lambda route: route.continue_() if route.request.url.startswith(live_server) else route.abort())
    pg.goto(live_server + '/')
    pg.wait_for_function("document.querySelector('#status').textContent.length > 0")
    yield pg
    context.close()
