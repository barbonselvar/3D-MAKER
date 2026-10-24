"""Python side of scripts/windows/setup.ps1 (kept in Python so it is testable on any OS).

    python -m worker.windows.setup_helpers <command>

Commands print one JSON object on stdout and exit 0 on success, 2 on a user-fixable problem.
Secrets are read from the environment (HF_TOKEN_INPUT), never from the command line.
"""
from __future__ import annotations

import argparse
import json
import os
import secrets
import sys
from pathlib import Path

from worker.selftest import state_dir

SF3D_REPO = 'stabilityai/stable-fast-3d'
SF3D_FILES = ('config.yaml', 'model.safetensors')
GATED_HELP = ('Stable Fast 3D is a gated model. Sign in at https://huggingface.co, open '
              'https://huggingface.co/stabilityai/stable-fast-3d, accept the Stability AI Community License, '
              'then run setup again.')


def env_file() -> Path:
    return state_dir() / 'worker.env'


def read_env(path: Path | None = None) -> dict[str, str]:
    path = path or env_file()
    values = {}
    if path.exists():
        for line in path.read_text(encoding='utf-8').splitlines():
            if '=' in line and not line.lstrip().startswith('#'):
                key, value = line.split('=', 1)
                values[key.strip()] = value.strip()
    return values


def write_env(updates: dict[str, str], path: Path | None = None) -> Path:
    path = path or env_file()
    values = read_env(path) | {k: v for k, v in updates.items() if v is not None}
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ['# 3D-MAKER local worker settings, written by scripts/windows/setup.ps1. Do not share this file.']
    lines += [f'{k}={v}' for k, v in sorted(values.items())]
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return path


def load_env(path: Path | None = None) -> None:
    """Apply worker.env to os.environ without overriding variables the user set explicitly."""
    for key, value in read_env(path).items():
        os.environ.setdefault(key, value)


def _out(payload: dict, code: int = 0) -> int:
    print(json.dumps(payload))
    return code


def cmd_init_env() -> int:
    current = read_env()
    path = write_env({'MAKER_SECRET_KEY': current.get('MAKER_SECRET_KEY') or secrets.token_hex(32),
                      'MAKER_DATA': current.get('MAKER_DATA') or str(state_dir() / 'data'),
                      'WORKER_ENGINES': current.get('WORKER_ENGINES') or 'sf3d',
                      'SEGMENTER': current.get('SEGMENTER') or 'rembg-u2net',
                      'HF_HUB_DISABLE_TELEMETRY': '1'})
    return _out({'ok': True, 'env_file': str(path)})


def cmd_torch() -> int:
    try:
        import torch  # noqa: PLC0415
    except ImportError as e:
        return _out({'ok': False, 'error': f'PyTorch is not installed: {e}'}, 2)
    info = {'ok': True, 'version': torch.__version__, 'cuda': torch.version.cuda, 'cuda_available': torch.cuda.is_available()}
    if info['cuda_available']:
        major, minor = torch.cuda.get_device_capability(0)
        info |= {'device': torch.cuda.get_device_name(0), 'capability': f'{major}.{minor}',
                 'bf16': bool(torch.cuda.is_bf16_supported())}
    else:
        info |= {'ok': False, 'error': 'PyTorch cannot see the NVIDIA GPU (CPU-only build or driver problem).'}
    return _out(info, 0 if info['ok'] else 2)


def cmd_hf_login() -> int:
    from huggingface_hub import HfApi, login  # noqa: PLC0415

    token = os.environ.get('HF_TOKEN_INPUT')
    if token:
        login(token=token, add_to_git_credential=False)
    try:
        who = HfApi().whoami()
    except Exception as e:  # noqa: BLE001 - any failure means "not signed in"
        return _out({'ok': False, 'error': f'Not signed in to Hugging Face ({type(e).__name__}). Create a read token at '
                     'https://huggingface.co/settings/tokens and run setup again.'}, 2)
    return _out({'ok': True, 'user': who.get('name')})


def _is_gated(error: Exception) -> bool:
    name = type(error).__name__
    status = getattr(getattr(error, 'response', None), 'status_code', None)
    return name in ('GatedRepoError', 'RepositoryNotFoundError') or status in (401, 403)


def cmd_weights(revision: str | None) -> int:
    """Pin (if not given) and download the SF3D weights; record the revision in worker.env."""
    from huggingface_hub import HfApi, hf_hub_download  # noqa: PLC0415

    try:
        revision = revision or read_env().get('SF3D_WEIGHTS_REVISION') or HfApi().model_info(SF3D_REPO).sha
        paths = [hf_hub_download(SF3D_REPO, name, revision=revision) for name in SF3D_FILES]
    except Exception as e:  # noqa: BLE001
        if _is_gated(e):
            return _out({'ok': False, 'error': GATED_HELP}, 2)
        return _out({'ok': False, 'error': f'Download failed: {type(e).__name__}: {str(e)[:300]}'}, 2)
    # Weights are now cached at a pinned revision: generation never needs to contact Hugging Face again.
    write_env({'SF3D_WEIGHTS_REVISION': revision, 'HF_HUB_OFFLINE': '1'})
    return _out({'ok': True, 'revision': revision, 'files': {Path(p).name: os.path.getsize(p) for p in paths}})


def cmd_u2net() -> int:
    import rembg  # noqa: PLC0415

    rembg.new_session('u2net', providers=['CPUExecutionProvider'])  # downloads and MD5-checks u2net.onnx once
    return _out({'ok': True})


def cmd_check() -> int:
    load_env()
    from worker.engines.sf3d import Sf3dEngine  # noqa: PLC0415

    problem = Sf3dEngine.install_problem()
    return _out({'ok': problem is None, 'problem': problem}, 0 if problem is None else 2)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='command', required=True)
    for name in ('init-env', 'torch', 'hf-login', 'u2net', 'check'):
        sub.add_parser(name)
    weights = sub.add_parser('weights')
    weights.add_argument('--revision')
    args = ap.parse_args(argv)
    handlers = {'init-env': cmd_init_env, 'torch': cmd_torch, 'hf-login': cmd_hf_login, 'u2net': cmd_u2net,
                'check': cmd_check, 'weights': lambda: cmd_weights(args.revision)}
    return handlers[args.command]()


if __name__ == '__main__':
    sys.exit(main())
