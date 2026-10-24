"""P1-10: Generate panel (Internal / External GPU), SSE progress and loading the validated result.

The internal GPU worker here is the MOCK StubEngine driven in-process (tests/stub_engine.py):
it proves the browser <-> API <-> worker flow, not real inference.
"""
import hashlib
import threading
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core import clock
from app.internal import create_internal_app
from worker.main import ApiClient, Worker

from ..helpers import L40S_PROBE
from ..stub_engine import FIX, StubEngine
from ..test_selftest import RTX4050, FakeGpuEngine

pytestmark = pytest.mark.ui
TOKEN = 'ui-worker-token-0123456789'


@pytest.fixture()
def ctx(live_server):
    from app.main import app

    context = app.state.ctx
    context.settings.worker_tokens[TOKEN] = 'ui-worker'
    with context.db.connect() as conn:  # isolate tests sharing the live server
        conn.execute('DELETE FROM workers')
        conn.execute("UPDATE jobs SET status='cancelled' WHERE status NOT IN ('completed','failed','cancelled','expired')")
    return context


def make_worker(ctx, fixture='pbr_cube_2048.glb'):
    client = TestClient(create_internal_app(context=ctx), headers={'Authorization': f'Bearer {TOKEN}'})
    worker = Worker(ApiClient(client), {'trellis2-4b': StubEngine(fixture=fixture)}, L40S_PROBE, heartbeat_interval=3600)
    worker.api.heartbeat(worker.probe, worker.engine_ids)
    return worker


def open_panel(page):
    page.click('[data-sample=vase]')
    page.wait_for_function("document.querySelector('#status').textContent.startsWith('Ready')")
    page.select_option('#mode', 'ai')
    page.click('#generate')
    page.wait_for_selector('#genpanel[open]')
    page.wait_for_function("!document.querySelector('#int-status').textContent.startsWith('Checking')")


def wait_for_job(page, ctx, timeout=10):
    deadline = clock.now() + timeout
    while clock.now() < deadline:
        with ctx.db.connect() as conn:
            row = conn.execute("SELECT id FROM jobs WHERE status='queued' ORDER BY created_at DESC LIMIT 1").fetchone()
        if row:
            return row['id']
        # Pumps Playwright's loop: the conftest request router only runs inside Playwright calls.
        page.wait_for_timeout(100)
    raise AssertionError('no job was created')


def run_in_thread(worker):
    """Run one worker iteration in a thread, keeping any exception for the failure report."""
    worker.thread_error = None

    def target():
        try:
            worker.run_once()
        except BaseException as e:  # noqa: BLE001 - surfaced by wait_for_result
            worker.thread_error = repr(e)
    worker.thread = threading.Thread(target=target, daemon=True)
    worker.thread.start()
    page_worker['current'] = worker


page_worker = {}


def wait_for_result(page, ctx, timeout=30_000):
    try:
        page.wait_for_function("document.querySelector('#stats').textContent.includes('Local GPU')", timeout=timeout)
    except Exception:
        worker = page_worker.get('current')
        with ctx.db.connect() as conn:
            rows = [dict(r) for r in conn.execute(
                'SELECT id, status, stage, error_code, error_message FROM jobs ORDER BY created_at DESC LIMIT 2')]
        raise AssertionError(f"model never appeared. status={page.text_content('#status')!r} jobs={rows} "
                             f"worker_error={getattr(worker, 'thread_error', None)} "
                             f"worker_alive={worker.thread.is_alive() if worker else None}") from None
    page_worker['current'].thread.join(10)


def test_internal_card_disabled_with_reason_when_no_worker(page, ctx):
    open_panel(page)
    assert page.is_disabled('#start-internal')
    assert 'No GPU worker online' in page.text_content('#int-status')
    assert page.get_attribute('#card-internal', 'aria-disabled') == 'true'
    # Dedicated external provider is shown but not selectable; the public demo is.
    assert page.is_disabled('input[value="external:modal"]')
    assert 'not configured' in page.text_content('#ext-providers')
    assert page.is_enabled('input[value="external:hf-demo"]')
    page.click('#genclose')
    assert not page.is_visible('#genpanel')


def test_internal_gpu_job_end_to_end(page, ctx):
    worker = make_worker(ctx)
    open_panel(page)
    assert 'NVIDIA L40S' in page.text_content('#int-status')
    assert 'TRELLIS.2-4B · Full quality' in page.text_content('#int-engines li[data-engine="trellis2-4b"]')
    assert page.text_content('#int-engines li[data-engine="trellis2-4b"] .chip') == 'Ready'
    assert page.eval_on_selector('#int-texture', 'el => el.value') == '2048'
    page.click('#start-internal')
    wait_for_job(page, ctx)
    page.wait_for_function("document.querySelector('#status').textContent.startsWith('Queued')")
    assert page.is_visible('#cancelai') and page.is_disabled('#generate')
    run_in_thread(worker)
    wait_for_result(page, ctx)
    assert 'estimated' in page.text_content('#status')
    assert page.eval_on_selector_all('#jobstages li.done', 'els => els.length') == 7
    assert page.is_disabled('#color')  # textured result
    with page.expect_download() as info:
        page.click('[data-export=glb]')
    data = Path(info.value.path()).read_bytes()
    assert hashlib.sha256(data).digest() == hashlib.sha256((FIX / 'pbr_cube_2048.glb').read_bytes()).digest()
    assert page.errors == []


def test_failed_job_shows_reason_and_keeps_viewer(page, ctx):
    worker = make_worker(ctx, fixture='bad_nan.glb')
    open_panel(page)
    before = page.text_content('#stats')
    page.click('#start-internal')
    wait_for_job(page, ctx)
    worker.run_once()
    page.wait_for_function("document.querySelector('#status').textContent.includes('No substitute model')", timeout=20_000)
    assert 'failed validation' in page.text_content('#status')
    assert page.text_content('#stats') == before
    assert page.is_enabled('#generate')


def test_cancel_queued_internal_job(page, ctx):
    make_worker(ctx)
    open_panel(page)
    page.click('#start-internal')
    job_id = wait_for_job(page, ctx)
    page.wait_for_selector('#cancelai:not([hidden])')
    page.click('#cancelai')
    page.wait_for_function("document.querySelector('#status').textContent.startsWith('Job cancelled')", timeout=20_000)
    with ctx.db.connect() as conn:
        assert conn.execute('SELECT status FROM jobs WHERE id=?', (job_id,)).fetchone()[0] == 'cancelled'


def test_panel_is_a_bottom_sheet_on_mobile(browser, live_server, ctx):
    context = browser.new_context(viewport={'width': 390, 'height': 844})
    page = context.new_page()
    page.goto(live_server + '/')
    open_panel(page)
    box = page.locator('#genpanel').bounding_box()
    assert box['width'] >= 389 and abs(box['y'] + box['height'] - 844) < 2
    for sel in ('#start-internal', '#start-external', '#genclose'):
        assert page.locator(sel).bounding_box()['height'] >= 44
    assert page.evaluate('document.documentElement.scrollWidth') <= 390
    context.close()


def make_reporting_worker(ctx, engine):
    """Worker that self-tests (MOCK FakeGpuEngine) and reports statuses, like the Windows local worker."""
    client = TestClient(create_internal_app(context=ctx), headers={'Authorization': f'Bearer {TOKEN}'})
    reports = {'sf3d': {'installed': True, 'problem': None, 'selftest': None},
               'trellis2-4b': {'installed': False, 'problem': 'TRELLIS.2 is not installed (needs a 24 GB GPU).',
                               'selftest': None}}
    worker = Worker(ApiClient(client), {'sf3d': engine}, RTX4050, heartbeat_interval=3600, reports=reports)
    return worker


def test_rtx4050_statuses_and_texture_cap(page, ctx, tmp_path, monkeypatch):
    monkeypatch.setenv('MAKER_WORKER_STATE', str(tmp_path))
    worker = make_reporting_worker(ctx, FakeGpuEngine(max_ok=512, low_needed=True))
    worker.api.heartbeat(worker.probe, worker.engine_ids)  # before the self-test
    open_panel(page)
    assert page.text_content('#int-engines li[data-engine="sf3d"] .chip') == 'Not tested yet'
    assert page.is_disabled('#start-internal')
    page.click('#genclose')
    worker.run_selftests(log_fn=lambda m: None)
    worker.api.heartbeat(worker.probe, worker.engine_ids)
    page.click('#generate')
    page.wait_for_function("document.querySelector('#int-engines li[data-engine=\"sf3d\"] .chip')?.textContent === 'Ready'")
    chips = {li: page.text_content(f'#int-engines li[data-engine="{li}"] .chip') for li in ('trellis2-4b', 'sf3d')}
    assert chips == {'trellis2-4b': 'Unsupported', 'sf3d': 'Ready'}
    assert 'needs at least 24 GB' in page.text_content('#int-engines li[data-engine="trellis2-4b"]')
    assert 'textures up to 512 px' in page.text_content('#int-engines li[data-engine="sf3d"]')
    assert page.eval_on_selector_all('#int-texture option', 'os => os.map(o => o.value)') == ['512']
    assert 'RTX 4050' in page.text_content('#int-status')
    page.click('#start-internal')
    wait_for_job(page, ctx)
    run_in_thread(worker)
    wait_for_result(page, ctx)
    stats = page.text_content('#stats')
    assert 'sf3d' in stats and '512 px' in stats and 'Powered by Stability AI' in stats


def test_out_of_memory_status_shown(page, ctx, tmp_path, monkeypatch):
    monkeypatch.setenv('MAKER_WORKER_STATE', str(tmp_path))
    worker = make_reporting_worker(ctx, FakeGpuEngine(max_ok=128))
    worker.run_selftests(log_fn=lambda m: None)
    worker.api.heartbeat(worker.probe, worker.engine_ids)
    open_panel(page)
    assert page.text_content('#int-engines li[data-engine="sf3d"] .chip') == 'Out of memory'
    assert 'Ran out of GPU memory' in page.text_content('#int-engines li[data-engine="sf3d"]')
    assert page.is_disabled('#start-internal')


def test_not_installed_status_shown(page, ctx):
    client = TestClient(create_internal_app(context=ctx), headers={'Authorization': f'Bearer {TOKEN}'})
    reports = {'sf3d': {'installed': False, 'problem': 'SF3D weights are not downloaded.', 'selftest': None}}
    worker = Worker(ApiClient(client), {}, RTX4050, heartbeat_interval=3600, reports=reports)
    worker.api.heartbeat(worker.probe, worker.engine_ids)
    open_panel(page)
    assert page.text_content('#int-engines li[data-engine="sf3d"] .chip') == 'Not installed'
    assert 'weights are not downloaded' in page.text_content('#int-engines li[data-engine="sf3d"]')
