"""MOCK engine for tests only.

Returns a hand-built fixture GLB instead of running a model. It exists to
exercise the job plumbing; it is not, and must never be presented as, AI
output. It lives under tests/ and is absent from worker.engines.PRODUCTION_ENGINES.
"""
from pathlib import Path

from worker.engines.base import Engine, EngineError, EngineResult

FIX = Path(__file__).parent / 'fixtures'


class StubEngine(Engine):
    id = 'trellis2-4b'  # impersonates the catalogue id so routing can be tested; output is a fixture

    def __init__(self, fixture='pbr_cube.glb', error: EngineError | None = None, on_stage=None):
        self.fixture, self.error, self.on_stage, self.seen = fixture, error, on_stage, {}

    def load(self):
        pass

    def run(self, image, params, report):
        self.seen = {'size': image.size, 'mode': image.mode, 'seed': params['seed']}
        for status, stage, p in [('preparing', 'segment', 0.1), ('running', 'infer', 0.3), ('postprocessing', 'cleanup', 0.7)]:
            report.stage(status, stage, p)
            if self.on_stage:
                self.on_stage(stage)
            if self.error and stage == 'infer':
                raise self.error
        return EngineResult(glb=(FIX / self.fixture).read_bytes(),
                            metadata={'engine': {'mock': True, 'fixture': self.fixture}, 'peak_vram_mb': None},
                            gpu_seconds=1.5)
