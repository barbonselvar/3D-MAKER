"""Engine catalogue, per-machine engine status, and eligibility rules shared by the API and the worker.

Pure metadata: no GPU or model code lives here. VRAM numbers are upstream published
figures (docs/PLAN.md), not measurements. Where upstream states a hard requirement
(TRELLIS.2: "at least 24GB ... is necessary") it is enforced here; where it only gives a
typical figure (SF3D: "about 6GB VRAM" at default settings) nothing is assumed and the
worker's self-test on the real machine decides (worker/selftest.py).
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
    min_vram_mb: int | None  # hard requirement stated upstream; None = decided by the self-test
    textured: bool
    licence: str
    material: str = 'pbr_textures'  # gate V4 profile: see worker/validate.py MATERIAL_PROFILES
    typical_vram_mb: int | None = None  # published typical use (informational)


ENGINES: dict[str, Engine] = {e.id: e for e in [
    Engine('trellis2-4b', 'TRELLIS.2-4B', 'full', frozenset({'cuda'}), 24 * GB, True,
           'MIT; default pipeline also pulls RMBG-2.0 (CC BY-NC 4.0), nvdiffrast (NVIDIA non-commercial) '
           'and DINOv3 (custom licence, "Built with DINOv3" attribution)'),
    Engine('hunyuan3d-2.1-shape', 'Hunyuan3D-2.1 (shape only)', 'untextured', frozenset({'cuda'}), 10 * GB, False,
           'Tencent Hunyuan 3D 2.1 Community License (excludes EU, UK, South Korea)'),
    Engine('sf3d', 'Stable Fast 3D', 'preview', frozenset({'cuda', 'mps', 'cpu'}), None, True,
           'Stability AI Community License: free for research, non-commercial use and organisations under '
           'US$1M annual revenue (commercial use requires registration with Stability AI); '
           'attribution "Powered by Stability AI" required',
           material='basecolor_normal', typical_vram_mb=6 * GB),
]}

# Preference order when several engines fit (best quality first).
PREFERENCE = ['trellis2-4b', 'hunyuan3d-2.1-shape', 'sf3d']

# Honest per-machine statuses shown in the Generate panel.
STATUS_LABELS = {
    'ready': 'Ready',
    'unsupported': 'Unsupported',
    'out_of_memory': 'Out of memory',
    'not_installed': 'Not installed',
    'untested': 'Not tested yet',
    'failed': 'Self-test failed',
}


def _gb(mb: int | None) -> str:
    return f'{round((mb or 0) / GB)} GB'


def hardware_problem(engine: Engine, probe: dict) -> str | None:
    """Reason the probed hardware can never run `engine`, or None. Only upstream-stated hard limits count."""
    backend = probe.get('backend', 'cpu')
    if backend not in engine.backends:
        names = {'cuda': 'an NVIDIA CUDA GPU', 'rocm': 'an AMD ROCm GPU', 'mps': 'Apple MPS', 'cpu': 'CPU'}
        need = ' or '.join(names[b] for b in sorted(engine.backends))
        return f'{engine.label} needs {need}; this worker has {names.get(backend, backend)}.'
    if backend in ('cuda', 'rocm') and engine.min_vram_mb:
        devices = probe.get('devices') or []
        total = max((d.get('vram_total_mb') or 0 for d in devices), default=0)
        if total < engine.min_vram_mb:
            return f'GPU has {_gb(total)}; {engine.label} needs at least {_gb(engine.min_vram_mb)}.'
    return None


def check(engine: Engine, probe: dict, installed: list[str]) -> str | None:
    """Legacy rule for workers that send no self-test reports: None if `engine` can run, else a reason."""
    if engine.id not in installed:
        return f'{engine.label} is not installed on this worker.'
    problem = hardware_problem(engine, probe)
    if problem:
        return problem
    if probe.get('backend') in ('cuda', 'rocm') and engine.min_vram_mb:
        devices = probe.get('devices') or []
        best = max((d.get('vram_free_mb') or d.get('vram_total_mb') or 0 for d in devices), default=0)
        if best < engine.min_vram_mb:
            return f'GPU has {_gb(best)} free; {engine.label} needs {_gb(engine.min_vram_mb)}.'
    return None


def engine_status(engine: Engine, probe: dict, installed: list[str]) -> dict:
    """Status of one engine on one worker: ready | unsupported | out_of_memory | not_installed | untested | failed."""
    out = {'id': engine.id, 'label': engine.label, 'quality': engine.quality, 'textured': engine.textured,
           'status': 'ready', 'reason': None, 'max_texture_size': None, 'peak_vram_mb': None, 'seconds': None}
    reports = probe.get('engine_reports')
    problem = hardware_problem(engine, probe)
    if problem:
        return out | {'status': 'unsupported', 'reason': problem}
    if reports is None:  # worker without self-test support (e.g. the Linux TRELLIS.2 image): legacy rule
        reason = check(engine, probe, installed)
        if reason is None:
            return out
        status = 'not_installed' if engine.id not in installed else 'out_of_memory'
        return out | {'status': status, 'reason': reason}
    report = reports.get(engine.id) or {}
    if not report.get('installed') or engine.id not in installed:
        return out | {'status': 'not_installed',
                      'reason': report.get('problem') or f'{engine.label} is not installed on this worker.'}
    test = report.get('selftest') or {}
    measured = {'max_texture_size': test.get('max_texture_size'), 'peak_vram_mb': test.get('peak_vram_mb'),
                'seconds': test.get('seconds')}
    result = test.get('status')
    if result == 'passed':
        return out | measured
    if result == 'out_of_memory':
        return out | measured | {'status': 'out_of_memory', 'reason': test.get('message') or
                                 f'{engine.label} ran out of GPU memory even at the smallest texture size.'}
    if result == 'failed':
        return out | measured | {'status': 'failed', 'reason': test.get('message') or 'The self-test failed.'}
    return out | {'status': 'untested', 'reason': 'The worker has not finished its self-test on this GPU yet.'}


def statuses(probe: dict, installed: list[str]) -> list[dict]:
    """Every catalogue engine, best first, with its status on this worker."""
    return [engine_status(ENGINES[i], probe, installed) for i in PREFERENCE]


def eligible(probe: dict, installed: list[str]) -> tuple[list[Engine], list[str]]:
    """Installed engines that are ready, best first, plus the reasons the other installed ones are not."""
    fits, reasons = [], []
    for st in statuses(probe, installed):
        if st['id'] not in installed:
            continue
        (fits.append(ENGINES[st['id']]) if st['status'] == 'ready' else reasons.append(st['reason']))
    return fits, reasons
