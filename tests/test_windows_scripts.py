"""Run scripts/windows/*.ps1 under PowerShell 7 with FAKE Windows tools (nvidia-smi, py, git, node, vswhere,
nvcc are small shell scripts). This exercises the scripts' real decision logic; it is not a Windows run.
Skipped when pwsh is not installed (set PWSH=/path/to/pwsh)."""
import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PWSH = os.environ.get('PWSH') or shutil.which('pwsh')
pytestmark = pytest.mark.skipif(not PWSH or os.name == 'nt',
                                reason='needs pwsh on Linux/macOS (fake tools are /bin/sh scripts)')


def fake(path: Path, body: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('#!/bin/sh\n' + body + '\n', encoding='utf-8')
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


def windows_like(tmp_path, cuda='12.7', vram='6141', with_cuda_toolkit=True, with_vs=True):
    bin_dir = tmp_path / 'bin'
    banner = f'| NVIDIA-SMI 566.36   Driver Version: 566.36   CUDA Version: {cuda} |'
    fake(bin_dir / 'nvidia-smi', f'if [ "$#" -gt 0 ]; then echo "NVIDIA GeForce RTX 4050 Laptop GPU, {vram}, 566.36"; '
                                 f'else echo "{banner}"; fi')
    fake(bin_dir / 'py', 'echo "3.11 64"')
    fake(bin_dir / 'git', 'echo "git version 2.47.0.windows.1"')
    fake(bin_dir / 'node', 'echo "v22.11.0"')
    pf86, pf = tmp_path / 'pf86', tmp_path / 'pf'
    if with_vs:  # pwsh on Linux treats the scripts' backslashes as directory separators
        fake(pf86 / 'Microsoft Visual Studio' / 'Installer' / 'vswhere.exe', f'echo "{tmp_path / "vs"}"')
        tools = tmp_path / 'vs' / 'VC' / 'Auxiliary' / 'Build'
        tools.mkdir(parents=True)
        (tools / 'Microsoft.VCToolsVersion.default.txt').write_text('14.41.34120', encoding='utf-8')
    env = {'PATH': f'{bin_dir}:{os.environ["PATH"]}', 'OS': 'Windows_NT', 'ProgramFiles(x86)': str(pf86),
           'ProgramFiles': str(pf), 'SystemRoot': str(tmp_path / 'win'), 'HOME': str(tmp_path)}
    if with_cuda_toolkit:
        cuda_dir = tmp_path / 'cuda124'
        fake(cuda_dir / 'bin' / 'nvcc.exe', 'echo "Cuda compilation tools, release 12.4, V12.4.131"')
        env['CUDA_PATH_V12_4'] = str(cuda_dir)
    return env


def run(script, env, *args):
    return subprocess.run([PWSH, '-NoLogo', '-NoProfile', '-File', str(ROOT / 'scripts' / 'windows' / script), *args],
                          capture_output=True, text=True, encoding='utf-8', errors='replace', env=env, timeout=120)


@pytest.mark.parametrize('script', ['setup.ps1', 'start.ps1'])
def test_scripts_parse(script):
    cmd = (f"$e=$null;$t=$null;[void][System.Management.Automation.Language.Parser]::ParseFile("
           f"'{ROOT / 'scripts' / 'windows' / script}',[ref]$t,[ref]$e); exit $e.Count")
    assert subprocess.run([PWSH, '-NoProfile', '-Command', cmd], timeout=60).returncode == 0


def test_check_only_all_prerequisites_found(tmp_path):
    r = run('setup.ps1', windows_like(tmp_path), '-CheckOnly')
    assert r.returncode == 0, r.stdout + r.stderr
    out = r.stdout
    assert 'GPU: NVIDIA GeForce RTX 4050 Laptop GPU, 6141 MiB VRAM, driver 566.36, driver supports CUDA 12.7' in out
    assert "close to Stable Fast 3D's published ~6 GB" in out  # honest small-VRAM warning
    assert 'MSVC toolset 14.41.34120' in out and 'CUDA Toolkit 12.4' in out
    assert 'All prerequisites found' in out


def test_check_only_lists_missing_cuda_toolkit_and_build_tools(tmp_path):
    r = run('setup.ps1', windows_like(tmp_path, with_cuda_toolkit=False, with_vs=False), '-CheckOnly')
    assert r.returncode == 1
    assert 'CUDA Toolkit 12.4' in r.stdout and 'cuda-12-4-1-download-archive' in r.stdout
    assert 'Microsoft.VisualStudio.2022.BuildTools' in r.stdout


def test_driver_too_old_stops_with_fix(tmp_path):
    r = run('setup.ps1', windows_like(tmp_path, cuda='12.2'), '-CheckOnly')
    assert r.returncode != 0
    assert 'The driver supports CUDA 12.2, but PyTorch here needs CUDA 12.4' in r.stdout
    assert 'Update the NVIDIA driver' in r.stdout


def test_not_windows_refused(tmp_path):
    env = windows_like(tmp_path)
    env['OS'] = 'Linux'
    r = run('setup.ps1', env, '-CheckOnly')
    assert r.returncode != 0 and 'needs 64-bit Windows 10 or 11' in r.stdout


def test_start_without_setup_explains(tmp_path):
    r = run('start.ps1', windows_like(tmp_path))
    if (ROOT / '.venv-gpu').exists():
        pytest.skip('a real .venv-gpu exists in this checkout')
    assert r.returncode == 1 and 'setup.ps1' in r.stdout
