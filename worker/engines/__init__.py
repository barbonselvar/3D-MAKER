"""Production engine registry.

Only real engines are listed. Test doubles live under tests/ and cannot be
loaded through this registry (asserted by tests/test_worker_integration.py).
"""
from __future__ import annotations

import importlib

from .base import Engine

PRODUCTION_ENGINES: dict[str, str] = {
    'trellis2-4b': 'worker.engines.trellis2:Trellis2Engine',
    'sf3d': 'worker.engines.sf3d:Sf3dEngine',
}


def engine_class(engine_id: str):
    module, cls = PRODUCTION_ENGINES[engine_id].split(':')
    return getattr(importlib.import_module(module), cls)


def load_engines(ids: list[str]) -> dict[str, Engine]:
    engines = {}
    for engine_id in ids:
        if engine_id not in PRODUCTION_ENGINES:
            raise ValueError(f'unknown engine {engine_id!r}; available: {", ".join(PRODUCTION_ENGINES)}')
        engines[engine_id] = engine_class(engine_id)()
    return engines
