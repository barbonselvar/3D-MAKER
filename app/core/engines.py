"""Engine catalogue and eligibility rules shared by the API and the worker.

Pure metadata: no GPU or model code lives here. Numbers are the upstream
published minimums (docs/PLAN.md, blueprint section B), not measurements.
"""
from __future__ import annotations

from dataclasses import dataclass

GB = 1024  # MiB per GiB, matching nvidia-smi's MiB units


@dataclass(frozen=True)
class Engine:
    id: str
    label: str
    quality: str  # "full" | "preview" | "untextured"
    backends: frozenset[str]  # subset of {"cuda", "rocm", "mps", "cpu"}
    min_vram_mb: int  # for GPU backends; CPU ignores it
    textured: bool
    licence: str


ENGINES: dict[str, Engine] = {e.id: e for e in [
    Engine('trellis2-4b', 'TRELLIS.2-4B', 'full', frozenset({'cuda'}), 24 * GB, True,
           'MIT; default pipeline also pulls RMBG-2.0 (CC BY-NC 4.0), nvdiffrast (NVIDIA non-commercial) '
           'and DINOv3 (custom licence, "Built with DINOv3" attribution)'),
    Engine('hunyuan3d-2.1-shape', 'Hunyuan3D-2.1 (shape only)', 'untextured', frozenset({'cuda'}), 10 * GB, False,
           'Tencent Hunyuan 3D 2.1 Community License (excludes EU, UK, South Korea)'),
    Engine('sf3d', 'Stable Fast 3D', 'preview', frozenset({'cuda', 'mps', 'cpu'}), 6 * GB, True,
           'Stability AI Community License (free below US$1M annual revenue)'),
]}

# Preference order when several engines fit (best quality first).
PREFERENCE = ['trellis2-4b', 'hunyuan3d-2.1-shape', 'sf3d']


def _gb(mb: int | None) -> str:
    return f'{round((mb or 0) / GB)} GB'


def check(engine: Engine, probe: dict, installed: list[str]) -> str | None:
    """None if `engine` can run on the probed device, else a one-sentence reason."""
    if engine.id not in installed:
        return f'{engine.label} is not installed on this worker.'
    backend = probe.get('backend', 'cpu')
    if backend not in engine.backends:
        names = {'cuda': 'an NVIDIA CUDA GPU', 'rocm': 'an AMD ROCm GPU', 'mps': 'Apple MPS', 'cpu': 'CPU'}
        need = ' or '.join(names[b] for b in sorted(engine.backends))
        return f'{engine.label} needs {need}; this worker has {names.get(backend, backend)}.'
    if backend in ('cuda', 'rocm'):
        devices = probe.get('devices') or []
        best = max((d.get('vram_free_mb') or d.get('vram_total_mb') or 0 for d in devices), default=0)
        if best < engine.min_vram_mb:
            return f'GPU has {_gb(best)} free; {engine.label} needs {_gb(engine.min_vram_mb)}.'
    return None


def eligible(probe: dict, installed: list[str]) -> tuple[list[Engine], list[str]]:
    """Engines that fit, best first, plus the reasons the others do not."""
    fits, reasons = [], []
    for engine_id in PREFERENCE:
        engine = ENGINES[engine_id]
        if engine_id not in installed:
            continue
        reason = check(engine, probe, installed)
        (reasons.append(reason) if reason else fits.append(engine))
    return fits, reasons
