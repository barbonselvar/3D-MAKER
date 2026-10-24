"""Capability probe for the Internal GPU worker (blueprint P1-01, plan C4).

Detects NVIDIA CUDA (nvidia-smi + torch), AMD ROCm (rocm-smi + torch HIP) and
Apple MPS (torch) on Linux and native Windows. WebGL in the browser is never
treated as inference compute. Every external call is injectable so tests run
on recorded outputs.
"""
from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
from collections.abc import Callable

NVIDIA_FIELDS = ['--query-gpu=name,memory.total,memory.free,driver_version', '--format=csv,noheader,nounits']
# Where NVIDIA drivers put nvidia-smi on Windows when it is not on PATH (DCH drivers / older drivers).
WINDOWS_SMI = [r'C:\Windows\System32\nvidia-smi.exe', r'C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe']
ROCM_QUERY = ['rocm-smi', '--showproductname', '--showmeminfo', 'vram', '--json']

Runner = Callable[[list[str]], str]


class ProbeError(RuntimeError):
    pass


def run_command(cmd: list[str]) -> str:
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=20)
    except FileNotFoundError:
        raise ProbeError(f'{cmd[0]} not found') from None
    except subprocess.TimeoutExpired:
        raise ProbeError(f'{cmd[0]} timed out') from None
    if result.returncode != 0:
        output = (result.stdout + result.stderr).strip()
        raise ProbeError(output.splitlines()[0] if output else f'{cmd[0]} exited with {result.returncode}')
    return result.stdout


def nvidia_smi() -> str:
    found = shutil.which('nvidia-smi')
    if found:
        return found
    if platform.system() == 'Windows':
        for path in WINDOWS_SMI:
            if os.path.isfile(path):
                return path
    return 'nvidia-smi'


def _mib(value: str) -> int | None:
    """nvidia-smi prints '[N/A]' or '[Not Supported]' for fields some drivers cannot read."""
    try:
        return int(float(value))
    except ValueError:
        return None


def parse_nvidia_smi(text: str) -> list[dict]:
    """Parse `nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv,noheader,nounits`."""
    devices = []
    for line in text.strip().splitlines():
        parts = [p.strip() for p in line.split(',')]
        if len(parts) != 4:
            raise ProbeError(f'unexpected nvidia-smi line: {line!r}')
        name, total, free, driver = parts
        devices.append({'name': name, 'vram_total_mb': _mib(total), 'vram_free_mb': _mib(free), 'driver': driver})
    return devices


def parse_driver_cuda(text: str) -> str | None:
    """The highest CUDA version the installed driver supports, from the plain `nvidia-smi` banner."""
    match = re.search(r'CUDA Version:\s*([0-9]+\.[0-9]+)', text)
    return match.group(1) if match else None


def _version(v: str | None) -> tuple[int, ...]:
    return tuple(int(x) for x in re.findall(r'\d+', v or '')[:2])


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
        info['device_name'] = torch.cuda.get_device_name(0)
        try:  # cudaMemGetInfo: reliable on Windows (WDDM) where nvidia-smi may print [N/A]
            free, total = torch.cuda.mem_get_info(0)
            info['vram_free_mb'], info['vram_total_mb'] = free // 2**20, total // 2**20
        except (RuntimeError, AttributeError):
            pass
    return info


def probe(run: Runner = run_command, torch=None) -> dict:
    """Returns {backend, devices, torch, errors, os, driver_cuda}. backend is cuda | rocm | mps | cpu."""
    errors: list[str] = []
    t = torch_info(torch)
    base = {'os': f'{platform.system()} {platform.release()}', 'driver_cuda': None}
    smi = nvidia_smi()
    try:
        devices = parse_nvidia_smi(run([smi, *NVIDIA_FIELDS]))
    except ProbeError as e:
        devices = []
        errors.append(f'nvidia-smi: {e}')
    if devices:
        try:
            base['driver_cuda'] = parse_driver_cuda(run([smi]))
        except ProbeError:
            pass
        if not t.get('installed'):
            errors.append('NVIDIA GPU present but PyTorch is not installed in this environment.')
            return {'backend': 'cpu', 'devices': devices, 'torch': t, 'errors': errors, **base}
        runtime, driver = t.get('cuda_runtime'), base['driver_cuda']
        if runtime and driver and _version(runtime) > _version(driver):
            errors.append(f'The NVIDIA driver supports CUDA {driver} but PyTorch was built for CUDA {runtime}. '
                          'Update the NVIDIA driver (or install a PyTorch build for an older CUDA).')
            return {'backend': 'cpu', 'devices': devices, 'torch': t, 'errors': errors, **base}
        if not t.get('cuda_available'):
            errors.append('NVIDIA GPU present but PyTorch cannot use CUDA (CPU-only PyTorch, or a driver/runtime mismatch).')
            return {'backend': 'cpu', 'devices': devices, 'torch': t, 'errors': errors, **base}
        for d in devices:
            d.update(cuda=t.get('cuda_runtime'), compute_capability=t.get('compute_capability'))
        if devices[0].get('vram_free_mb') is None and t.get('vram_free_mb') is not None:
            devices[0]['vram_free_mb'] = t['vram_free_mb']
        if devices[0].get('vram_total_mb') is None and t.get('vram_total_mb') is not None:
            devices[0]['vram_total_mb'] = t['vram_total_mb']
        return {'backend': 'cuda', 'devices': devices, 'torch': t, 'errors': errors, **base}
    try:
        devices = parse_rocm_smi(run(ROCM_QUERY))
    except (ProbeError, ValueError) as e:
        devices = []
        errors.append(f'rocm-smi: {e}')
    if devices and t.get('hip_runtime') and t.get('cuda_available'):  # torch exposes ROCm through torch.cuda
        return {'backend': 'rocm', 'devices': devices, 'torch': t, 'errors': errors, **base}
    if devices:
        errors.append('AMD GPU present but PyTorch was not built with ROCm.')
    if t.get('mps_available'):
        return {'backend': 'mps', 'devices': [{'name': 'Apple GPU (MPS)', 'vram_total_mb': None,
                                                'vram_free_mb': None, 'driver': None}], 'torch': t, 'errors': errors, **base}
    return {'backend': 'cpu', 'devices': devices, 'torch': t, 'errors': errors, **base}


if __name__ == '__main__':
    print(json.dumps(probe(), indent=2))
