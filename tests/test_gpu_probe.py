"""P1-01: GPU probe on recorded (synthetic) outputs, engine eligibility and /api/v1/gpu-targets."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.core import clock
from app.core.db import dumps
from app.core.engines import ENGINES, check, eligible
from worker import gpu_probe

FIX = Path(__file__).parent / 'fixtures' / 'nvidia_smi'


def fake_torch(cuda=False, hip=None, mps=False, cap=(8, 9)):
    return SimpleNamespace(
        __version__='2.6.0', version=SimpleNamespace(cuda='12.4' if cuda and not hip else None, hip=hip),
        cuda=SimpleNamespace(is_available=lambda: cuda, get_device_capability=lambda i: cap),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps)))


def runner(nvidia=None, rocm=None, nvidia_error=None):
    def run(cmd):
        if cmd[0] == 'nvidia-smi':
            if nvidia is None:
                raise gpu_probe.ProbeError(nvidia_error or 'nvidia-smi not found')
            return (FIX / nvidia).read_text()
        if rocm is None:
            raise gpu_probe.ProbeError('rocm-smi not found')
        return (FIX / rocm).read_text()
    return run


def test_parse_l40s():
    [d] = gpu_probe.parse_nvidia_smi((FIX / 'l40s_48gb.txt').read_text())
    assert d == {'name': 'NVIDIA L40S', 'vram_total_mb': 46068, 'vram_free_mb': 45512, 'driver': '550.54.15'}


def test_parse_rejects_garbage():
    with pytest.raises(gpu_probe.ProbeError):
        gpu_probe.parse_nvidia_smi((FIX / 'driver_missing.txt').read_text())


def test_no_gpu_at_all():
    p = gpu_probe.probe(runner(), torch=fake_torch())
    assert p['backend'] == 'cpu' and p['devices'] == []
    assert any('nvidia-smi' in e for e in p['errors'])


def test_driver_missing():
    msg = (FIX / 'driver_missing.txt').read_text().strip()
    p = gpu_probe.probe(runner(nvidia_error=msg), torch=fake_torch())
    assert p['backend'] == 'cpu'
    assert any("couldn't communicate with the NVIDIA driver" in e for e in p['errors'])


def test_cuda_48gb():
    p = gpu_probe.probe(runner('l40s_48gb.txt'), torch=fake_torch(cuda=True))
    assert p['backend'] == 'cuda'
    assert p['devices'][0]['compute_capability'] == '8.9' and p['devices'][0]['cuda'] == '12.4'


def test_gpu_present_but_cpu_only_torch():
    p = gpu_probe.probe(runner('l40s_48gb.txt'), torch=fake_torch(cuda=False))
    assert p['backend'] == 'cpu'
    assert any('PyTorch cannot use CUDA' in e for e in p['errors'])


def test_torch_missing_is_reported(monkeypatch):
    monkeypatch.setitem(__import__('sys').modules, 'torch', None)  # makes `import torch` raise ImportError
    p = gpu_probe.probe(runner())
    assert p['torch'] == {'installed': False} and p['backend'] == 'cpu'


def test_rocm():
    p = gpu_probe.probe(runner(rocm='rocm_mi210.json'), torch=fake_torch(cuda=True, hip='6.2'))
    assert p['backend'] == 'rocm'
    assert p['devices'][0]['name'] == 'AMD Instinct MI210'
    assert p['devices'][0]['vram_total_mb'] == 65520


def test_mps():
    p = gpu_probe.probe(runner(), torch=fake_torch(mps=True))
    assert p['backend'] == 'mps'


def probe_for(fixture, **torch_kw):
    return gpu_probe.probe(runner(fixture), torch=fake_torch(cuda=True, **torch_kw))


ALL = ['trellis2-4b', 'hunyuan3d-2.1-shape', 'sf3d']


def test_eligibility_48gb_prefers_trellis2():
    fits, _ = eligible(probe_for('l40s_48gb.txt'), ALL)
    assert [e.id for e in fits] == ['trellis2-4b', 'hunyuan3d-2.1-shape', 'sf3d']


def test_eligibility_12gb_explains_trellis2_refusal():
    probe = probe_for('rtx3060_12gb.txt')
    assert check(ENGINES['trellis2-4b'], probe, ALL) == 'GPU has 11 GB free; TRELLIS.2-4B needs 24 GB.'
    fits, _ = eligible(probe, ALL)
    assert [e.id for e in fits] == ['hunyuan3d-2.1-shape', 'sf3d']


def test_eligibility_uses_best_device_free_memory():
    fits, _ = eligible(probe_for('two_a100.txt'), ['trellis2-4b'])
    assert [e.id for e in fits] == ['trellis2-4b']


def test_cpu_only_gets_sf3d_preview_or_nothing():
    cpu = gpu_probe.probe(runner(), torch=fake_torch())
    assert [e.id for e in eligible(cpu, ALL)[0]] == ['sf3d']
    fits, reasons = eligible(cpu, ['trellis2-4b'])
    assert fits == [] and 'needs an NVIDIA CUDA GPU' in reasons[0]


def test_uninstalled_engine_is_never_offered():
    assert eligible(probe_for('l40s_48gb.txt'), []) == ([], [])


# --- /api/v1/gpu-targets -------------------------------------------------------------------------

def add_worker(api, worker_id, probe, engines, age=0.0):
    with api.app.state.ctx.db.connect() as conn:
        conn.execute('INSERT INTO workers (id, kind, probe, engines, heartbeat_at, created_at) VALUES (?,?,?,?,?,?)',
                     (worker_id, 'internal', dumps(probe), dumps(engines), clock.now() - age, clock.now()))


def targets(api):
    return {t['id']: t for t in api.get('/api/v1/gpu-targets').json()['targets']}


def test_targets_without_worker(api):
    t = targets(api)
    assert t['internal']['available'] is False
    assert 'No GPU worker online' in t['internal']['reason']
    assert t['external:modal']['configured'] is False and t['external:modal']['available'] is False
    assert t['external:hf-demo']['available'] is True and t['external:hf-demo']['requires_consent'] is True


def test_targets_with_48gb_worker(api):
    add_worker(api, 'w1', probe_for('l40s_48gb.txt'), ['trellis2-4b'])
    internal = targets(api)['internal']
    assert internal['available'] is True
    assert internal['device']['name'] == 'NVIDIA L40S'
    assert internal['engines'][0]['id'] == 'trellis2-4b' and internal['engines'][0]['quality'] == 'full'


def test_targets_with_12gb_worker_explains(api):
    add_worker(api, 'w1', probe_for('rtx3060_12gb.txt'), ['trellis2-4b'])
    internal = targets(api)['internal']
    assert internal['available'] is False
    assert internal['reason'] == 'GPU has 11 GB free; TRELLIS.2-4B needs 24 GB.'


def test_stale_worker_ignored(api):
    add_worker(api, 'w1', probe_for('l40s_48gb.txt'), ['trellis2-4b'], age=46)
    assert targets(api)['internal']['available'] is False
