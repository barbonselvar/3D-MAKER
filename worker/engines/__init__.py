"""Production engine registry.

Only real engines are listed. Test doubles live under tests/ and cannot be
loaded through this registry (asserted by tests/test_worker_integration.py).
"""
from __future__ import annotations

import importlib

from .base import Engine

PRODUCTION_ENGINES: dict[str, str] = {
    'trellis2-4b': 'worker.engines.trellis2:Trellis2Engine',
}


def load_engines(ids: list[str]) -> dict[str, Engine]:
    engines = {}
    for engine_id in ids:
        if engine_id not in PRODUCTION_ENGINES:
            raise ValueError(f'unknown engine {engine_id!r}; available: {", ".join(PRODUCTION_ENGINES)}')
        module, cls = PRODUCTION_ENGINES[engine_id].split(':')
        engines[engine_id] = getattr(importlib.import_module(module), cls)()
    return engines
