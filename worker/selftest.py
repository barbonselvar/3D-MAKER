"""Per-machine engine self-test: measures what actually fits on this GPU (no assumptions).

At worker start each loaded engine runs on a fixture photo, trying its selftest_ladder of
(texture size, low-VRAM mode) from best to most frugal. The first setting whose output passes
validation makes the engine "Ready" and caps the texture size jobs may request. If every step
runs out of GPU memory the engine is "Out of memory". Results are cached per fingerprint
(GPU, driver, PyTorch, engine code and weights) so the test only reruns when something changes.
A real job that later runs out of memory moves the engine one step down the ladder.
"""
from __future__ import annotations

import json
import os
import platform
import time
from collections.abc import Callable
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

from PIL import Image

from .engines.base import Engine, EngineError

HERE = Path(__file__).resolve().parent
SELFTEST_IMAGE = HERE / 'smoke' / 'sphere_on_white.png'  # no alpha: exercises background removal too
SELFTEST_PARAMS = {'seed': 1, 'resolution': 1024, 'face_budget': 200_000}


def state_dir() -> Path:
    if os.environ.get('MAKER_WORKER_STATE'):
        return Path(os.environ['MAKER_WORKER_STATE'])
    if platform.system() == 'Windows' and os.environ.get('LOCALAPPDATA'):
        return Path(os.environ['LOCALAPPDATA']) / '3D-MAKER'
    return Path.home() / '.3d-maker'


def fingerprint(engine: Engine, probe: dict) -> dict:
    device = (probe.get('devices') or [{}])[0]
    module = __import__(type(engine).__module__, fromlist=['_'])
    return {'engine': engine.id, 'engine_code': next((getattr(module, n) for n in dir(module) if n.endswith('_COMMIT')), None),
            'weights': getattr(engine, 'weights_revision', None), 'backend': probe.get('backend'),
            'gpu': device.get('name'), 'vram_total_mb': device.get('vram_total_mb'), 'driver': device.get('driver'),
            'torch': (probe.get('torch') or {}).get('version'), 'ladder': [list(step) for step in engine.selftest_ladder]}


class _QuietReporter:
    def stage(self, status, stage, progress=None):
        pass

    def check_cancel(self):
        pass


def run_selftest(engine: Engine, validator: Callable, profile: str, image_path: Path = SELFTEST_IMAGE,
                 log: Callable[[str], None] = print) -> dict:
    data = image_path.read_bytes()
    attempts = []
    started = time.monotonic()
    for texture, low_vram in engine.selftest_ladder:
        engine.set_memory_mode(low_vram)
        label = f'{texture} px{" + low-VRAM" if low_vram else ""}'
        log(f'[selftest] {engine.id}: trying {label}')
        t0 = time.monotonic()
        try:
            result = engine.run(Image.open(BytesIO(data)), SELFTEST_PARAMS | {'texture_size': texture}, _QuietReporter())
        except EngineError as e:
            attempts.append({'texture_size': texture, 'low_vram': low_vram, 'result': e.code})
            if e.code == 'INFERENCE_OOM':
                log(f'[selftest] {engine.id}: out of GPU memory at {label}')
                continue
            return _result('failed', attempts, started, message=f'{e.code}: {e.message}')
        except Exception as e:  # noqa: BLE001 - report the failure instead of crashing the worker
            attempts.append({'texture_size': texture, 'low_vram': low_vram, 'result': type(e).__name__})
            return _result('failed', attempts, started, message=f'{type(e).__name__}: {str(e)[:300]}')
        report = validator(result.glb, face_budget=SELFTEST_PARAMS['face_budget'], texture_size=texture, profile=profile)
        attempts.append({'texture_size': texture, 'low_vram': low_vram, 'result': 'passed' if report.passed else
                         'validation_failed', 'peak_vram_mb': result.metadata.get('peak_vram_mb'),
                         'seconds': round(time.monotonic() - t0, 2)})
        if not report.passed:
            failures = '; '.join(f'{g.id} {g.message or g.status}' for g in report.failures())
            return _result('failed', attempts, started, message=f'Output failed validation: {failures}')
        log(f'[selftest] {engine.id}: passed at {label}, peak VRAM {result.metadata.get("peak_vram_mb")} MB')
        return _result('passed', attempts, started, max_texture_size=texture, low_vram=low_vram,
                       peak_vram_mb=result.metadata.get('peak_vram_mb'), seconds=attempts[-1]['seconds'])
    smallest = engine.selftest_ladder[-1][0]
    return _result('out_of_memory', attempts, started,
                   message=f'Ran out of GPU memory at every setting down to {smallest} px. Close other programs '
                           'using the GPU and restart the worker to test again.')


def _result(status: str, attempts: list, started: float, **fields) -> dict:
    return {'status': status, 'attempts': attempts, 'total_seconds': round(time.monotonic() - started, 2),
            'tested_at': datetime.now(timezone.utc).isoformat(timespec='seconds'), 'max_texture_size': None,
            'low_vram': None, 'peak_vram_mb': None, 'seconds': None, 'message': None} | fields


def step_down(engine: Engine, result: dict, failed_texture: int) -> dict:
    """After a real job ran out of memory at failed_texture, move to the next ladder step below it."""
    ladder = list(engine.selftest_ladder)
    current = (failed_texture, bool(result.get('low_vram')))
    index = ladder.index(current) if current in ladder else max(
        (i for i, (t, _) in enumerate(ladder) if t == failed_texture), default=-1)
    remaining = ladder[index + 1:]
    note = f'A job ran out of GPU memory at {failed_texture} px'
    if not remaining:
        return result | {'status': 'out_of_memory', 'max_texture_size': None,
                         'message': f'{note}, the smallest setting. Close other programs using the GPU and restart the worker.'}
    texture, low_vram = remaining[0]
    return result | {'status': 'passed', 'max_texture_size': texture, 'low_vram': low_vram,
                     'message': f'{note}; now limited to {texture} px{" with low-VRAM mode" if low_vram else ""}.'}


class SelfTestStore:
    """JSON cache: {engine_id: {"fingerprint": {...}, "result": {...}}}."""

    def __init__(self, path: Path | None = None):
        self.path = path or state_dir() / 'selftest.json'

    def _read(self) -> dict:
        try:
            return json.loads(self.path.read_text(encoding='utf-8'))
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def get(self, engine_id: str, fp: dict) -> dict | None:
        entry = self._read().get(engine_id)
        return entry['result'] if entry and entry.get('fingerprint') == fp else None

    def put(self, engine_id: str, fp: dict, result: dict) -> None:
        data = self._read()
        data[engine_id] = {'fingerprint': fp, 'result': result}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix('.tmp')
        tmp.write_text(json.dumps(data, indent=2), encoding='utf-8')
        os.replace(tmp, self.path)


def ensure_selftest(engine: Engine, probe: dict, store: SelfTestStore, validator: Callable, profile: str,
                    force: bool = False, log: Callable[[str], None] = print) -> dict:
    fp = fingerprint(engine, probe)
    cached = None if force else store.get(engine.id, fp)
    if cached is not None:
        log(f'[selftest] {engine.id}: using cached result ({cached["status"]}, tested {cached["tested_at"]})')
        result = cached
    else:
        result = run_selftest(engine, validator, profile, log=log)
        store.put(engine.id, fp, result)
    if result.get('low_vram') is not None:
        engine.set_memory_mode(bool(result['low_vram']))
    return result
