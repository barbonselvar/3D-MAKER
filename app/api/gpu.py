"""GET /api/v1/gpu-targets: what the Generate panel can offer right now."""
from fastapi import APIRouter, Depends

from ..core import clock
from ..core.context import Context, ctx, current_owner
from ..core.db import loads
from ..core.engines import eligible

router = APIRouter(prefix='/api/v1')


def internal_target(c: Context) -> dict:
    """Internal GPU = the best online worker (fresh heartbeat) by usable VRAM."""
    target = {'id': 'internal', 'kind': 'internal', 'configured': True, 'available': False, 'reason': None,
              'device': None, 'backend': None, 'engines': [], 'requires_consent': False, 'price_estimate_usd': None}
    with c.db.connect() as conn:
        rows = conn.execute("SELECT * FROM workers WHERE kind='internal' AND heartbeat_at >= ?",
                            (clock.now() - c.settings.heartbeat_fresh,)).fetchall()
    if not rows:
        target['reason'] = 'No GPU worker online. Start a worker on a machine with an NVIDIA GPU (see worker/README.md).'
        return target
    best = None
    for row in rows:
        probe, installed = loads(row['probe']) or {}, loads(row['engines']) or []
        fits, reasons = eligible(probe, installed)
        vram = max((d.get('vram_free_mb') or 0 for d in probe.get('devices') or []), default=0)
        score = (bool(fits), vram)
        if best is None or score > best[0]:
            best = (score, row, probe, fits, reasons)
    _, row, probe, fits, reasons = best
    devices = probe.get('devices') or []
    target['backend'] = probe.get('backend')
    target['device'] = max(devices, key=lambda d: d.get('vram_free_mb') or 0) if devices else None
    target['engines'] = [{'id': e.id, 'label': e.label, 'quality': e.quality, 'textured': e.textured,
                          'est_seconds': None} for e in fits]
    target['available'] = bool(fits)
    if not fits:
        target['reason'] = ' '.join(reasons) or 'No image-to-3D engine is installed on the online worker.'
    return target


def external_targets() -> list[dict]:
    # The implementation prompt asks to show configured/unconfigured state, so the dedicated provider
    # is listed with configured=false until Phase 2 (paid; needs owner approval) instead of hidden.
    return [
        {'id': 'external:modal', 'kind': 'external', 'configured': False, 'available': False,
         'reason': 'Dedicated External GPU is not configured yet (Phase 2; needs owner approval and a budget cap).',
         'engines': [], 'requires_consent': True, 'price_estimate_usd': None},
        {'id': 'external:hf-demo', 'kind': 'external', 'configured': True, 'available': True, 'reason': None,
         'runs_in': 'browser', 'engines': [{'id': 'trellis2-4b', 'label': 'TRELLIS.2 (public demo)', 'quality': 'full',
                                            'textured': True, 'est_seconds': None}],
         'requires_consent': True, 'price_estimate_usd': 0,
         'notes': 'Shared Hugging Face Space: free quota, queues, no SLA; the image leaves your device.'},
    ]


@router.get('/gpu-targets')
def gpu_targets(c: Context = Depends(ctx), _owner: str = Depends(current_owner)):
    return {'targets': [internal_target(c), *external_targets()]}
