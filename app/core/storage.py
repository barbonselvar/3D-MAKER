"""Object storage interface and the local-directory backend (S3-compatible backend: P4-02).

Keys are random (never job ids). Downloads go through short-lived HMAC-signed
URLs, so artifacts are never publicly listable or guessable.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import secrets
import tempfile
from pathlib import Path
from typing import Protocol
from urllib.parse import quote, urlencode

from . import clock

KEY_RE = re.compile(r'^[A-Za-z0-9_-]{16,128}$')


class Storage(Protocol):
    def new_key(self) -> str: ...
    def put(self, key: str, data: bytes, content_type: str) -> None: ...
    def get(self, key: str) -> bytes: ...
    def exists(self, key: str) -> bool: ...
    def delete(self, key: str) -> None: ...
    def presign(self, key: str, filename: str, ttl: int | None = None) -> str: ...


class LocalStorage:
    def __init__(self, root: Path, secret: bytes, ttl: int = 600, url_prefix: str = '/api/v1/files'):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.secret, self.ttl, self.url_prefix = secret, ttl, url_prefix

    def new_key(self) -> str:
        return secrets.token_urlsafe(24)

    def path(self, key: str) -> Path:
        if not KEY_RE.match(key):
            raise KeyError('invalid storage key')
        return self.root / key[:2] / key

    def put(self, key: str, data: bytes, content_type: str) -> None:
        path = self.path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent)
        with os.fdopen(fd, 'wb') as fh:
            fh.write(data)
        os.replace(tmp, path)
        path.with_suffix('.meta').write_text(json.dumps({'content_type': content_type, 'size': len(data)}))

    def get(self, key: str) -> bytes:
        return self.path(key).read_bytes()

    def content_type(self, key: str) -> str:
        meta = self.path(key).with_suffix('.meta')
        return json.loads(meta.read_text(encoding='utf-8'))['content_type'] if meta.exists() else 'application/octet-stream'

    def exists(self, key: str) -> bool:
        try:
            return self.path(key).is_file()
        except KeyError:
            return False

    def delete(self, key: str) -> None:
        path = self.path(key)
        path.unlink(missing_ok=True)
        path.with_suffix('.meta').unlink(missing_ok=True)

    def _sign(self, key: str, exp: int, name: str) -> str:
        msg = f'{key}\n{exp}\n{name}'.encode()
        return hmac.new(self.secret, msg, hashlib.sha256).hexdigest()

    def presign(self, key: str, filename: str, ttl: int | None = None) -> str:
        self.path(key)  # validates the key
        exp = int(clock.now()) + (ttl or self.ttl)
        query = urlencode({'exp': exp, 'name': filename, 'sig': self._sign(key, exp, filename)})
        return f'{self.url_prefix}/{quote(key)}?{query}'

    def verify(self, key: str, exp: int, name: str, sig: str) -> str | None:
        """Returns None if the URL is valid, else an error code."""
        if not KEY_RE.match(key) or not hmac.compare_digest(self._sign(key, exp, name), sig):
            return 'SIGNATURE_INVALID'
        if clock.now() > exp:
            return 'URL_EXPIRED'
        return None
