"""worker/windows/setup_helpers.py: settings file and Hugging Face error handling (MOCK hub; no network)."""
import json
import sys
import types

import pytest

from worker.windows import setup_helpers as sh


@pytest.fixture(autouse=True)
def state(tmp_path, monkeypatch):
    monkeypatch.setenv('MAKER_WORKER_STATE', str(tmp_path))
    return tmp_path


def run(capsys, *argv):
    code = sh.main(list(argv))
    return code, json.loads(capsys.readouterr().out.strip().splitlines()[-1])


def test_init_env_creates_and_keeps_secret(capsys, state):
    code, out = run(capsys, 'init-env')
    first = sh.read_env()
    assert code == 0 and len(first['MAKER_SECRET_KEY']) == 64 and first['WORKER_ENGINES'] == 'sf3d'
    assert first['SEGMENTER'] == 'rembg-u2net' and first['MAKER_DATA'].endswith('data')
    run(capsys, 'init-env')
    assert sh.read_env()['MAKER_SECRET_KEY'] == first['MAKER_SECRET_KEY']  # stable across re-runs


def test_load_env_never_overrides_explicit(monkeypatch, state):
    sh.write_env({'WORKER_ENGINES': 'sf3d', 'MAKER_PORT': '8000'})
    monkeypatch.setenv('WORKER_ENGINES', 'trellis2-4b')
    monkeypatch.delenv('MAKER_PORT', raising=False)
    sh.load_env()
    import os
    assert os.environ['WORKER_ENGINES'] == 'trellis2-4b' and os.environ['MAKER_PORT'] == '8000'


def fake_hub(monkeypatch, model_info=None, download=None):
    hub = types.SimpleNamespace(
        HfApi=lambda: types.SimpleNamespace(model_info=model_info or (lambda repo: types.SimpleNamespace(sha='deadbeef')),
                                            whoami=lambda: {'name': 'me'}),
        hf_hub_download=download, login=lambda **kw: None)
    monkeypatch.setitem(sys.modules, 'huggingface_hub', hub)


def test_weights_pins_revision_and_records_it(capsys, monkeypatch, tmp_path):
    files = []

    def download(repo, name, revision):
        files.append((repo, name, revision))
        path = tmp_path / name
        path.write_bytes(b'x' * 10)
        return str(path)
    fake_hub(monkeypatch, download=download)
    code, out = run(capsys, 'weights')
    assert code == 0 and out['revision'] == 'deadbeef'
    assert files == [('stabilityai/stable-fast-3d', 'config.yaml', 'deadbeef'),
                     ('stabilityai/stable-fast-3d', 'model.safetensors', 'deadbeef')]
    assert sh.read_env()['SF3D_WEIGHTS_REVISION'] == 'deadbeef'
    run(capsys, 'weights')
    assert files[-1][2] == 'deadbeef'  # re-runs keep the recorded revision


def test_weights_gated_message(capsys, monkeypatch):
    class GatedRepoError(Exception):
        pass

    def gated(repo):
        raise GatedRepoError('403')
    fake_hub(monkeypatch, model_info=gated)
    code, out = run(capsys, 'weights')
    assert code == 2 and 'accept the Stability AI Community License' in out['error']


def test_check_reports_install_problem(capsys, monkeypatch):
    monkeypatch.setitem(sys.modules, 'torch', None)
    code, out = run(capsys, 'check')
    assert code == 2 and 'PyTorch is not installed' in out['problem']
