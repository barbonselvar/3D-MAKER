"""Internal GPU worker loop (blueprint P1-07).

Pulls jobs from the internal API with a per-worker token, heartbeats every
15 s, streams stage changes, validates the GLB (gates V1-V4) and uploads
artifacts. One job per GPU at a time. Never substitutes a non-AI mesh: any
failure is reported with an error code.

    MAKER_INTERNAL_URL=http://api-host:8001 MAKER_WORKER_TOKEN=... python -m worker.main
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import threading
import time
from collections.abc import Callable
from contextlib import contextmanager
from io import BytesIO

import httpx
from PIL import Image

from app.core.engines import ENGINES

from .engines.base import Cancelled, Engine, EngineError
from .selftest import SelfTestStore, ensure_selftest, fingerprint, step_down
from .validate import validate_glb

log = logging.getLogger('maker.worker')
WORKER_VERSION = '0.2.0'


def material_profile(engine: Engine) -> str:
    """Gate V4 profile from the catalogue; engines outside it fall back to their textured flag."""
    catalogue = ENGINES.get(engine.id)
    return catalogue.material if catalogue else ('pbr_textures' if engine.textured else 'basecolor')


class JobGone(Exception):
    """The job was cancelled or reassigned server-side."""


class ApiClient:
    def __init__(self, http: httpx.Client):
        self.http = http

    def _post(self, path: str, body: dict) -> httpx.Response:
        r = self.http.post(path, json=body)
        if r.status_code in (404, 409) and path.startswith('/internal/v1/jobs/'):
            raise JobGone(r.json().get('error', {}).get('code', str(r.status_code)))
        r.raise_for_status()
        return r

    def heartbeat(self, probe: dict, engines: list[str], job_id: str | None = None) -> bool:
        r = self.http.post('/internal/v1/heartbeat', json={'probe': probe, 'engines': engines, 'job_id': job_id})
        r.raise_for_status()
        return bool(r.json()['cancel'])

    def claim(self, engines: list[str]) -> dict | None:
        r = self.http.post('/internal/v1/claim', json={'engines': engines})
        r.raise_for_status()
        return None if r.status_code == 204 else r.json()['job']

    def input(self, job: dict) -> bytes:
        r = self.http.get(job['input']['url'])
        if r.status_code in (404, 409):
            raise JobGone(str(r.status_code))
        r.raise_for_status()
        return r.content

    def progress(self, job_id: str, **fields) -> None:
        self._post(f'/internal/v1/jobs/{job_id}/progress', {k: v for k, v in fields.items() if v is not None})

    def upload(self, job_id: str, fmt: str, data: bytes, content_type: str) -> None:
        r = self.http.put(f'/internal/v1/jobs/{job_id}/artifacts/{fmt}', content=data, headers={'content-type': content_type})
        if r.status_code in (404, 409):
            raise JobGone(str(r.status_code))
        r.raise_for_status()

    def complete(self, job_id: str, manifest: dict, validation: dict, gpu_seconds: float | None) -> str:
        return self._post(f'/internal/v1/jobs/{job_id}/complete',
                          {'manifest': manifest, 'validation': validation, 'gpu_seconds': gpu_seconds}).json()['model_id']

    def fail(self, job_id: str, code: str, message: str, retryable: bool) -> None:
        self._post(f'/internal/v1/jobs/{job_id}/fail', {'code': code, 'message': message[:500], 'retryable': retryable})


class JobReporter:
    """Passed to engines: forwards stage changes and raises Cancelled once the user cancels."""

    def __init__(self, api: ApiClient, job_id: str, cancel: threading.Event):
        self.api, self.job_id, self.cancel = api, job_id, cancel
        self.timings: dict[str, float] = {}
        self._stage, self._since = None, time.monotonic()

    def check_cancel(self) -> None:
        if self.cancel.is_set():
            raise Cancelled()

    def stage(self, status: str, stage: str, progress: float | None = None) -> None:
        self.check_cancel()
        now = time.monotonic()
        if self._stage:
            self.timings[self._stage] = round(self.timings.get(self._stage, 0) + now - self._since, 3)
        self._stage, self._since = stage, now
        try:
            self.api.progress(self.job_id, status=status, stage=stage, progress=progress)
        except JobGone:
            self.cancel.set()
            raise Cancelled() from None

    def finish(self) -> None:
        if self._stage:
            self.timings[self._stage] = round(self.timings.get(self._stage, 0) + time.monotonic() - self._since, 3)
            self._stage = None


class Worker:
    def __init__(self, api: ApiClient, engines: dict[str, Engine], probe: dict,
                 validator: Callable = validate_glb, heartbeat_interval: float = 15, poll_interval: float = 3,
                 reports: dict | None = None, store: SelfTestStore | None = None):
        self.api, self.engines, self._probe, self.validator = api, engines, probe, validator
        self.heartbeat_interval, self.poll_interval = heartbeat_interval, poll_interval
        # Per-engine install/self-test reports (None: legacy worker without self-tests, e.g. the TRELLIS.2 image).
        self.reports, self.store = reports, store

    @property
    def probe(self) -> dict:
        """What the API sees: the hardware probe plus, when self-tests are used, each engine's report."""
        return self._probe if self.reports is None else self._probe | {'engine_reports': self.reports}

    @property
    def engine_ids(self) -> list[str]:
        """Engines loaded on this worker (installed)."""
        return list(self.engines)

    @property
    def ready_engine_ids(self) -> list[str]:
        """Engines this worker may claim jobs for: all loaded ones, or only those that passed the self-test."""
        if self.reports is None:
            return self.engine_ids
        return [e for e in self.engines if ((self.reports.get(e) or {}).get('selftest') or {}).get('status') == 'passed']

    @contextmanager
    def background_heartbeat(self):
        """Keep the worker 'online' (with its current reports) during long startup work such as self-tests."""
        done = threading.Event()

        def loop():
            while True:
                try:
                    self.api.heartbeat(self.probe, self.engine_ids)
                except httpx.HTTPError as e:
                    log.warning('heartbeat failed: %s', e)
                if done.wait(self.heartbeat_interval):
                    return

        thread = threading.Thread(target=loop, daemon=True)
        thread.start()
        try:
            yield
        finally:
            done.set()
            thread.join(timeout=2)

    def run_selftests(self, force: bool = False, log_fn: Callable[[str], None] = print) -> None:
        for engine_id, engine in self.engines.items():
            if self.reports is None or engine_id not in self.reports:
                continue
            self.reports[engine_id]['selftest'] = ensure_selftest(
                engine, self._probe, self.store or SelfTestStore(), self.validator, material_profile(engine),
                force=force, log=log_fn)

    def _out_of_memory(self, engine_id: str, texture: int) -> None:
        """A real job ran out of GPU memory: step the engine down its self-test ladder and remember it."""
        report = (self.reports or {}).get(engine_id)
        if not report or not report.get('selftest'):
            return
        engine = self.engines[engine_id]
        result = step_down(engine, report['selftest'], texture)
        report['selftest'] = result
        if result.get('low_vram') is not None:
            engine.set_memory_mode(bool(result['low_vram']))
        (self.store or SelfTestStore()).put(engine_id, fingerprint(engine, self._probe), result)
        log.warning('%s: %s', engine_id, result['message'])

    def run_forever(self, stop: threading.Event | None = None) -> None:
        stop = stop or threading.Event()
        while not stop.is_set():
            try:
                if not self.run_once():
                    stop.wait(self.poll_interval)
            except httpx.HTTPError as e:
                log.warning('API unreachable: %s', e)
                stop.wait(self.poll_interval * 5)

    def run_once(self) -> bool:
        """Heartbeat, claim at most one job and process it. Returns True if a job was processed."""
        self.api.heartbeat(self.probe, self.engine_ids)
        job = self.api.claim(self.ready_engine_ids)
        if job is None:
            return False
        self.process(job)
        return True

    def _heartbeat_loop(self, job_id: str, cancel: threading.Event, done: threading.Event) -> None:
        while not done.wait(self.heartbeat_interval):
            try:
                if self.api.heartbeat(self.probe, self.engine_ids, job_id):
                    cancel.set()
            except httpx.HTTPError as e:
                log.warning('heartbeat failed: %s', e)

    def process(self, job: dict) -> None:
        job_id, params = job['id'], job['params']
        cancel, done = threading.Event(), threading.Event()
        beat = threading.Thread(target=self._heartbeat_loop, args=(job_id, cancel, done), daemon=True)
        beat.start()
        reporter = JobReporter(self.api, job_id, cancel)
        started = time.monotonic()
        try:
            reporter.stage('preparing', 'load', 0.02)
            data = self.api.input(job)
            if hashlib.sha256(data).hexdigest() != job['input']['sha256']:
                raise EngineError('INPUT_INVALID', 'Input image checksum mismatch.', retryable=True)
            image = Image.open(BytesIO(data))
            image.load()
            engine = self.engines[job['engine']]
            result = engine.run(image, params, reporter)
            reporter.stage('validating', 'validate', 0.9)
            report = self.validator(result.glb, face_budget=params['face_budget'], texture_size=params['texture_size'],
                                    profile=material_profile(engine))
            if not report.passed:
                summary = '; '.join(f'{g.id} {g.message or g.status}' for g in report.failures())
                raise EngineError('VALIDATION_FAILED', f'The generated model failed validation: {summary}', retryable=True)
            reporter.stage('exporting', 'export', 0.95)
            reporter.finish()
            manifest = self.manifest(job, engine, result, report, reporter.timings, time.monotonic() - started)
            self.api.upload(job_id, 'glb', result.glb, 'model/gltf-binary')
            self.api.upload(job_id, 'manifest', json.dumps(manifest, indent=2).encode(), 'application/json')
            model_id = self.api.complete(job_id, manifest, {'glb': report.to_dict()}, result.gpu_seconds)
            log.info('job %s completed -> %s', job_id, model_id)
        except (Cancelled, JobGone):
            log.info('job %s cancelled', job_id)
        except EngineError as e:
            log.warning('job %s failed: %s %s', job_id, e.code, e.message)
            if e.code == 'INFERENCE_OOM':
                self._out_of_memory(job['engine'], params['texture_size'])
            self._fail(job_id, e.code, e.message, e.retryable)
        except Exception as e:  # noqa: BLE001 - never leave a job hanging; report and continue
            log.exception('job %s crashed', job_id)
            self._fail(job_id, 'ENGINE_ERROR', f'{type(e).__name__}: {e}', True)
        finally:
            done.set()
            beat.join(timeout=1)

    def _fail(self, job_id: str, code: str, message: str, retryable: bool) -> None:
        try:
            self.api.fail(job_id, code, message, retryable)
        except JobGone:
            pass

    def manifest(self, job, engine, result, report, timings, wall) -> dict:
        """Reproducibility record stored next to every model (blueprint section D)."""
        catalogue = ENGINES.get(engine.id)
        devices = self.probe.get('devices') or []
        return {
            'manifest_version': 1,
            'job_id': job['id'],
            'input_sha256': job['input']['sha256'],
            'engine': {'id': engine.id, **result.metadata.get('engine', {})},
            'params': job['params'],
            'device': {'backend': self.probe.get('backend'), **(devices[0] if devices else {})},
            'timings_s': timings | {'total': round(wall, 3)},
            'gpu_seconds': result.gpu_seconds,
            'peak_vram_mb': result.metadata.get('peak_vram_mb'),
            'pipeline': {k: v for k, v in result.metadata.items() if k not in ('engine', 'peak_vram_mb')},
            'validation': report.to_dict(),
            'licences': [catalogue.licence] if catalogue else [],
            'notices': result.metadata.get('notices', []),
            'worker': {'version': WORKER_VERSION, 'python': platform.python_version()},
            'note': 'Generated from one image: back and hidden parts are estimated, not reconstructed.',
        }


class _PrintReporter:
    def stage(self, status, stage, progress=None):
        print(f'[smoke] {status:<15} {stage}', flush=True)

    def check_cancel(self):
        pass


def smoke(engines: dict[str, Engine], probe: dict, image_path: str, out_dir: str, params: dict | None = None) -> bool:
    """Run each engine once on a fixture image, validate, and write GLB + manifest + report to out_dir."""
    from pathlib import Path  # noqa: PLC0415

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = Path(image_path).read_bytes()
    params = {'seed': 1, 'resolution': 1024, 'texture_size': 2048, 'face_budget': 200_000} | (params or {})
    ok = True
    for engine in engines.values():
        started = time.monotonic()
        result = engine.run(Image.open(BytesIO(data)), params, _PrintReporter())
        report = validate_glb(result.glb, face_budget=params['face_budget'], texture_size=params['texture_size'],
                              profile=material_profile(engine))
        job = {'id': 'smoke', 'params': params, 'input': {'sha256': hashlib.sha256(data).hexdigest()}}
        manifest = Worker(None, engines, probe).manifest(job, engine, result, report, {}, time.monotonic() - started)
        (out / f'{engine.id}.glb').write_bytes(result.glb)
        (out / f'{engine.id}.manifest.json').write_text(json.dumps(manifest, indent=2))
        print(f'[smoke] {engine.id}: {len(result.glb)} bytes, validation {"PASSED" if report.passed else "FAILED"}')
        for gate in report.gates:
            print(f'[smoke]   {gate.id} {gate.status:<7} {gate.message}')
        ok = ok and report.passed
    return ok


def prepare_engines(requested: list[str], classes: dict | None = None) -> tuple[dict[str, Engine], dict]:
    """Check, then load, each requested engine. Missing pieces become 'Not installed' reports, never crashes."""
    from .engines import PRODUCTION_ENGINES, engine_class  # noqa: PLC0415

    loaded, reports = {}, {}
    for engine_id in requested:
        if classes is None and engine_id not in PRODUCTION_ENGINES:
            reports[engine_id] = {'installed': False, 'problem': f'Unknown engine {engine_id!r}.', 'selftest': None}
            continue
        cls = (classes or {}).get(engine_id) or engine_class(engine_id)
        problem = cls.install_problem()
        if problem:
            log.warning('%s not installed: %s', engine_id, problem)
            reports[engine_id] = {'installed': False, 'problem': problem, 'selftest': None}
            continue
        engine = cls()
        try:
            engine.load()
        except Exception as e:  # noqa: BLE001 - surface the reason in the UI instead of crashing
            log.exception('%s failed to load', engine_id)
            reports[engine_id] = {'installed': False, 'problem': f'Loading failed: {type(e).__name__}: {str(e)[:300]}',
                                  'selftest': None}
            continue
        loaded[engine_id] = engine
        reports[engine_id] = {'installed': True, 'problem': None, 'selftest': None}
    return loaded, reports


def main() -> None:
    import argparse

    from .gpu_probe import probe

    ap = argparse.ArgumentParser(description='3D Maker internal GPU worker')
    ap.add_argument('--engines', default=os.environ.get('WORKER_ENGINES', 'trellis2-4b'))
    ap.add_argument('--probe-only', action='store_true', help='print the capability probe and exit')
    ap.add_argument('--smoke', action='store_true', help='generate one GLB from a fixture image, validate it, exit')
    ap.add_argument('--smoke-input', default=os.path.join(os.path.dirname(__file__), 'smoke', 'sphere.png'))
    ap.add_argument('--smoke-out', default='/out' if os.path.isdir('/out') else 'out')  # /out: Docker volume
    ap.add_argument('--selftest', action='store_true', help='re-run the GPU self-test, print the results and exit')
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format='{"ts":"%(asctime)s","level":"%(levelname)s","msg":"%(message)s"}')
    from .windows.setup_helpers import load_env  # noqa: PLC0415 - settings written by scripts/windows/setup.ps1

    load_env()
    caps = probe()
    if args.probe_only:
        print(json.dumps(caps, indent=2))
        return
    engines, reports = prepare_engines([e.strip() for e in args.engines.split(',') if e.strip()])
    for engine_id, report in reports.items():
        print(f'[worker] {engine_id}: {"installed" if report["installed"] else "NOT INSTALLED - " + report["problem"]}')
    if args.smoke:
        raise SystemExit(0 if engines and smoke(engines, caps, args.smoke_input, args.smoke_out) else 1)
    if args.selftest:
        worker = Worker(None, engines, caps, reports=reports)
        worker.run_selftests(force=True)
        print(json.dumps({k: v['selftest'] for k, v in reports.items()}, indent=2))
        raise SystemExit(0 if worker.ready_engine_ids else 1)
    serve(os.environ['MAKER_INTERNAL_URL'], os.environ['MAKER_WORKER_TOKEN'], caps=caps, prepared=(engines, reports))


def serve(internal_url: str, token: str, engine_ids: list[str] | None = None, caps: dict | None = None,
          classes: dict | None = None, prepared: tuple | None = None, stop: threading.Event | None = None,
          heartbeat_interval: float = 15, poll_interval: float = 3) -> Worker:
    """Connect to the worker API, self-test the engines (staying visible meanwhile), then process jobs until stop."""
    if caps is None:
        from .gpu_probe import probe  # noqa: PLC0415

        caps = probe()
    engines, reports = prepared or prepare_engines(engine_ids or [], classes)
    http = httpx.Client(base_url=internal_url, timeout=120, headers={'Authorization': f'Bearer {token}'})
    worker = Worker(ApiClient(http), engines, caps, reports=reports, heartbeat_interval=heartbeat_interval,
                    poll_interval=poll_interval)
    with worker.background_heartbeat():  # the UI shows "Not tested yet" while self-tests run
        worker.run_selftests()
    worker.run_forever(stop)
    return worker


if __name__ == '__main__':
    main()
