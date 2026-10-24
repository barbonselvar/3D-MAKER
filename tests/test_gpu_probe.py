"""P1-01: GPU probe on recorded (synthetic) outputs, engine eligibility and /api/v1/gpu-targets."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.core import clock
from app.core.db import dumps
from app.core.engines import ENGINES, check, eligible
from worker import gpu_probe

FIX = Path(__file__).parent / 'fixtures' / 'nvidia_smi'


def fake_torch(cuda=False, hip=None, mps=False, cap=(8, 9), runtime='12.4', mem=None, name='NVIDIA L40S'):
    def mem_get_info(i):
        if mem is None:
            raise RuntimeError('no device')
        return mem
    return SimpleNamespace(
        __version__='2.6.0', version=SimpleNamespace(cuda=runtime if cuda and not hip else None, hip=hip),
        cuda=SimpleNamespace(is_available=lambda: cuda, get_device_capability=lambda i: cap,
                             get_device_name=lambda i: name, mem_get_info=mem_get_info),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps)))


def runner(nvidia=None, rocm=None, nvidia_error=None, banner=None):
    def run(cmd):
        if 'nvidia-smi' in cmd[0]:
            if nvidia is None:
                raise gpu_probe.ProbeError(nvidia_error or 'nvidia-smi not found')
            if len(cmd) == 1:
                return (FIX / banner).read_text() if banner else ''
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
    assert check(ENGINES['trellis2-4b'], probe, ALL) == 'GPU has 12 GB; TRELLIS.2-4B needs at least 24 GB.'
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
    assert internal['reason'] == 'TRELLIS.2-4B: Unsupported. GPU has 12 GB; TRELLIS.2-4B needs at least 24 GB.'


def test_stale_worker_ignored(api):
    add_worker(api, 'w1', probe_for('l40s_48gb.txt'), ['trellis2-4b'], age=46)
    assert targets(api)['internal']['available'] is False


# --- native Windows (synthetic RTX 4050 Laptop outputs; see fixtures README) ----------------------

def test_windows_rtx4050_laptop():
    p = gpu_probe.probe(runner('rtx4050_laptop_win.txt', banner='banner_win_cuda127.txt'),
                        torch=fake_torch(cuda=True, name='NVIDIA GeForce RTX 4050 Laptop GPU'))
    assert p['backend'] == 'cuda' and p['driver_cuda'] == '12.7'
    d = p['devices'][0]
    assert (d['name'], d['vram_total_mb'], d['vram_free_mb'], d['driver']) == (
        'NVIDIA GeForce RTX 4050 Laptop GPU', 6141, 5329, '566.36')


def test_windows_na_free_memory_filled_from_torch():
    torch = fake_torch(cuda=True, mem=(5100 * 2**20, 6141 * 2**20))
    p = gpu_probe.probe(runner('rtx4050_laptop_win_na.txt', banner='banner_win_cuda127.txt'), torch=torch)
    assert p['devices'][0]['vram_free_mb'] == 5100


def test_driver_too_old_for_torch_cuda_runtime():
    p = gpu_probe.probe(runner('rtx4050_laptop_win.txt', banner='banner_win_cuda122.txt'), torch=fake_torch(cuda=True))
    assert p['backend'] == 'cpu'
    assert 'driver supports CUDA 12.2 but PyTorch was built for CUDA 12.4' in p['errors'][-1]


def test_gpu_present_torch_not_installed(monkeypatch):
    monkeypatch.setitem(__import__('sys').modules, 'torch', None)
    p = gpu_probe.probe(runner('rtx4050_laptop_win.txt'))
    assert p['backend'] == 'cpu' and 'PyTorch is not installed' in p['errors'][-1]


def test_nvidia_smi_found_in_windows_locations(monkeypatch):
    monkeypatch.setattr(gpu_probe.shutil, 'which', lambda name: None)
    monkeypatch.setattr(gpu_probe.platform, 'system', lambda: 'Windows')
    monkeypatch.setattr(gpu_probe.os.path, 'isfile', lambda p: p == gpu_probe.WINDOWS_SMI[1])
    assert gpu_probe.nvidia_smi() == gpu_probe.WINDOWS_SMI[1]


def test_parse_driver_cuda():
    assert gpu_probe.parse_driver_cuda((FIX / 'banner_win_cuda127.txt').read_text()) == '12.7'
    assert gpu_probe.parse_driver_cuda('no banner') is None


# --- per-machine engine statuses (self-test reports from the worker) -------------------------------

from app.core.engines import STATUS_LABELS, statuses  # noqa: E402

RTX4050 = {'backend': 'cuda', 'devices': [{'name': 'NVIDIA GeForce RTX 4050 Laptop GPU', 'vram_total_mb': 6141,
                                           'vram_free_mb': 5329, 'driver': '566.36'}]}


def with_reports(**reports):
    return RTX4050 | {'engine_reports': reports}


def by_id(probe, installed):
    return {s['id']: s for s in statuses(probe, installed)}


def test_status_labels_cover_required_states():
    assert {'ready', 'unsupported', 'out_of_memory', 'not_installed'} <= set(STATUS_LABELS)


def test_rtx4050_trellis_unsupported_even_if_installed():
    s = by_id(with_reports(**{'trellis2-4b': {'installed': True}}), ['trellis2-4b'])['trellis2-4b']
    assert s['status'] == 'unsupported' and 'needs at least 24 GB' in s['reason']


def test_sf3d_not_installed():
    s = by_id(with_reports(sf3d={'installed': False, 'problem': 'texture_baker was built without CUDA.'}), [])['sf3d']
    assert (s['status'], s['reason']) == ('not_installed', 'texture_baker was built without CUDA.')


def test_sf3d_untested_until_selftest_reports():
    s = by_id(with_reports(sf3d={'installed': True, 'selftest': None}), ['sf3d'])['sf3d']
    assert s['status'] == 'untested'


def test_sf3d_ready_after_passing_selftest():
    test = {'status': 'passed', 'max_texture_size': 512, 'peak_vram_mb': 4870, 'seconds': 9.5}
    s = by_id(with_reports(sf3d={'installed': True, 'selftest': test}), ['sf3d'])['sf3d']
    assert s['status'] == 'ready' and s['max_texture_size'] == 512 and s['peak_vram_mb'] == 4870


def test_sf3d_out_of_memory():
    test = {'status': 'out_of_memory', 'message': 'Out of GPU memory at 512 px.'}
    s = by_id(with_reports(sf3d={'installed': True, 'selftest': test}), ['sf3d'])['sf3d']
    assert (s['status'], s['reason']) == ('out_of_memory', 'Out of GPU memory at 512 px.')


def test_sf3d_has_no_assumed_vram_threshold():
    tiny = {'backend': 'cuda', 'devices': [{'name': 'x', 'vram_total_mb': 2048, 'vram_free_mb': 2000}],
            'engine_reports': {'sf3d': {'installed': True, 'selftest': None}}}
    assert by_id(tiny, ['sf3d'])['sf3d']['status'] == 'untested'  # the real self-test decides


def test_targets_report_engine_statuses_for_rtx4050(api):
    test = {'status': 'passed', 'max_texture_size': 1024, 'peak_vram_mb': 5200, 'seconds': 12.0}
    probe = with_reports(sf3d={'installed': True, 'selftest': test}, **{'trellis2-4b': {'installed': False}})
    add_worker(api, 'w1', probe, ['sf3d'])
    internal = targets(api)['internal']
    assert internal['available'] is True
    assert internal['engines'] == [{'id': 'sf3d', 'label': 'Stable Fast 3D', 'quality': 'preview', 'textured': True,
                                    'max_texture_size': 1024, 'est_seconds': 12.0}]
    st = {s['id']: (s['status'], s['status_label']) for s in internal['engine_status']}
    assert st['trellis2-4b'] == ('unsupported', 'Unsupported') and st['sf3d'] == ('ready', 'Ready')


def test_targets_explain_out_of_memory(api):
    probe = with_reports(sf3d={'installed': True, 'selftest': {'status': 'out_of_memory', 'message': 'OOM at 512 px.'}})
    add_worker(api, 'w1', probe, ['sf3d'])
    internal = targets(api)['internal']
    assert internal['available'] is False
    assert 'Stable Fast 3D: Out of memory. OOM at 512 px.' in internal['reason']
