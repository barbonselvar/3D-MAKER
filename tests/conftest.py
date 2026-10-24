import os
import tempfile

# app.main reads MAKER_OUTPUT at import time, so point it at a throwaway directory first.
os.environ.setdefault('MAKER_OUTPUT', tempfile.mkdtemp(prefix='3d-maker-test-'))
os.environ.setdefault('MAKER_DATA', tempfile.mkdtemp(prefix='3d-maker-data-'))


import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.core.config import Settings  # noqa: E402


@pytest.fixture()
def settings(tmp_path):
    return Settings(data_dir=tmp_path / 'data', secret_key=b's' * 32,
                    worker_tokens={'tok-worker-one-0123456789': 'w1'})


@pytest.fixture()
def api(settings):
    """A fresh public app with its own database and storage."""
    from app.main import create_app

    return TestClient(create_app(settings))
