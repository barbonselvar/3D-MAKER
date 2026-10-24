"""P1-06: local storage backend and signed URLs."""
from urllib.parse import parse_qs, urlparse

import pytest

from app.core import clock
from app.core.storage import LocalStorage


@pytest.fixture()
def store(tmp_path):
    return LocalStorage(tmp_path / 'objects', b'secret' * 6, ttl=600)


def test_put_get_delete_roundtrip(store):
    key = store.new_key()
    store.put(key, b'hello', 'text/plain')
    assert store.get(key) == b'hello' and store.exists(key)
    assert store.content_type(key) == 'text/plain'
    store.delete(key)
    assert not store.exists(key)


@pytest.mark.parametrize('key', ['../../etc/passwd', 'short', 'a/b' * 10, '..' * 10, 'x' * 200])
def test_rejects_unsafe_keys(store, key):
    with pytest.raises(KeyError):
        store.path(key)
    assert store.exists(key) is False


def test_keys_are_random(store):
    assert len({store.new_key() for _ in range(100)}) == 100


def test_presigned_url_lifecycle(api, monkeypatch):
    ctx = api.app.state.ctx
    key = ctx.storage.new_key()
    ctx.storage.put(key, b'GLBDATA', 'model/gltf-binary')
    url = ctx.storage.presign(key, 'model.glb')
    r = api.get(url)
    assert r.status_code == 200 and r.content == b'GLBDATA'
    assert r.headers['content-type'] == 'model/gltf-binary'
    assert 'model.glb' in r.headers['content-disposition']

    q = {k: v[0] for k, v in parse_qs(urlparse(url).query).items()}
    tampered = url.replace(q['sig'], '0' * 64)
    assert api.get(tampered).json()['error']['code'] == 'SIGNATURE_INVALID'
    renamed = url.replace('name=model.glb', 'name=evil.html')
    assert api.get(renamed).status_code == 403
    other = ctx.storage.new_key()
    assert api.get(url.replace(key, other)).status_code == 403

    real_now = clock.now()
    monkeypatch.setattr(clock, 'now', lambda: real_now + 601)
    r = api.get(url)
    assert r.status_code == 403 and r.json()['error']['code'] == 'URL_EXPIRED'
