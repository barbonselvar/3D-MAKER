"""Capability probe for the Internal GPU worker (blueprint P1-01, plan C4).

Detects NVIDIA CUDA (nvidia-smi + torch), AMD ROCm (rocm-smi + torch HIP) and
Apple MPS (torch). WebGL in the browser is never treated as inference compute.
Every external call is injectable so tests run on recorded outputs.
"""
from __future__ import annotations

import json
import subprocess
from collections.abc import Callable

NVIDIA_QUERY = ['nvidia-smi', '--query-gpu=name,memory.total,memory.free,driver_version', '--format=csv,noheader,nounits']
ROCM_QUERY = ['rocm-smi', '--showproductname', '--showmeminfo', 'vram', '--json']

Runner = Callable[[list[str]], str]


class ProbeError(RuntimeError):
    pass


def run_command(cmd: list[str]) -> str:
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
    except FileNotFoundError:
        raise ProbeError(f'{cmd[0]} not found') from None
    except subprocess.TimeoutExpired:
        raise ProbeError(f'{cmd[0]} timed out') from None
    if result.returncode != 0:
        output = (result.stdout + result.stderr).strip()
        raise ProbeError(output.splitlines()[0] if output else f'{cmd[0]} exited with {result.returncode}')
    return result.stdout


def parse_nvidia_smi(text: str) -> list[dict]:
    """Parse `nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv,noheader,nounits`."""
    devices = []
    for line in text.strip().splitlines():
        parts = [p.strip() for p in line.split(',')]
        if len(parts) != 4:
            raise ProbeError(f'unexpected nvidia-smi line: {line!r}')
        name, total, free, driver = parts
        devices.append({'name': name, 'vram_total_mb': int(float(total)), 'vram_free_mb': int(float(free)),
                        'driver': driver})
    return devices


def parse_rocm_smi(text: str) -> list[dict]:
    """Parse `rocm-smi --showproductname --showmeminfo vram --json` (keys vary across ROCm releases)."""
    data = json.loads(text)
    devices = []
    for card, info in sorted(data.items()):
        if not card.lower().startswith('card'):
            continue
        total = next((v for k, v in info.items() if 'vram total memory' in k.lower()), None)
        used = next((v for k, v in info.items() if 'vram total used memory' in k.lower()), None)
        name = next((v for k, v in info.items() if k.lower() in ('card series', 'card model', 'card sku')), card)
        total_mb = int(int(total) / 2**20) if total is not None else None
        free_mb = total_mb - int(int(used) / 2**20) if total is not None and used is not None else None
        devices.append({'name': name, 'vram_total_mb': total_mb, 'vram_free_mb': free_mb, 'driver': None})
    return devices


def torch_info(torch=None) -> dict:
    if torch is None:
        try:
            import torch  # noqa: PLC0415 - optional, heavy
        except ImportError:
            return {'installed': False}
    info = {'installed': True, 'version': torch.__version__,
            'cuda_available': bool(torch.cuda.is_available()),
            'cuda_runtime': getattr(torch.version, 'cuda', None),
            'hip_runtime': getattr(torch.version, 'hip', None),
            'mps_available': bool(getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available())}
    if info['cuda_available']:
        major, minor = torch.cuda.get_device_capability(0)
        info['compute_capability'] = f'{major}.{minor}'
    return info


def probe(run: Runner = run_command, torch=None) -> dict:
    """Returns {backend, devices, torch, errors}. backend is cuda | rocm | mps | cpu."""
    errors: list[str] = []
    t = torch_info(torch)
    try:
        devices = parse_nvidia_smi(run(NVIDIA_QUERY))
    except ProbeError as e:
        devices = []
        errors.append(f'nvidia-smi: {e}')
    if devices:
        if not t.get('cuda_available'):
            errors.append('NVIDIA GPU present but PyTorch cannot use CUDA (driver/runtime mismatch or CPU-only torch).')
            return {'backend': 'cpu', 'devices': devices, 'torch': t, 'errors': errors}
        for d in devices:
            d.update(cuda=t.get('cuda_runtime'), compute_capability=t.get('compute_capability'))
        return {'backend': 'cuda', 'devices': devices, 'torch': t, 'errors': errors}
    try:
        devices = parse_rocm_smi(run(ROCM_QUERY))
    except (ProbeError, ValueError) as e:
        devices = []
        errors.append(f'rocm-smi: {e}')
    if devices and t.get('hip_runtime') and t.get('cuda_available'):  # torch exposes ROCm through torch.cuda
        return {'backend': 'rocm', 'devices': devices, 'torch': t, 'errors': errors}
    if devices:
        errors.append('AMD GPU present but PyTorch was not built with ROCm.')
    if t.get('mps_available'):
        return {'backend': 'mps', 'devices': [{'name': 'Apple GPU (MPS)', 'vram_total_mb': None,
                                                'vram_free_mb': None, 'driver': None}], 'torch': t, 'errors': errors}
    return {'backend': 'cpu', 'devices': devices, 'torch': t, 'errors': errors}


if __name__ == '__main__':
    print(json.dumps(probe(), indent=2))
