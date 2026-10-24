"""Runtime settings, read from the environment (see .env.example)."""
from __future__ import annotations

import os
import secrets
from dataclasses import dataclass, field
from pathlib import Path


def _worker_tokens(raw: str) -> dict[str, str]:
    """Parse MAKER_WORKER_TOKENS="worker-id:token,other:token2" into {token: worker_id}."""
    tokens: dict[str, str] = {}
    for item in filter(None, (part.strip() for part in raw.split(','))):
        worker_id, sep, token = item.partition(':')
        if not sep or not worker_id or len(token) < 16:
            raise ValueError('MAKER_WORKER_TOKENS entries must be "<worker-id>:<token of 16+ chars>"')
        tokens[token] = worker_id
    return tokens


@dataclass
class Settings:
    data_dir: Path
    secret_key: bytes
    worker_tokens: dict[str, str] = field(default_factory=dict)
    api_token: str | None = None  # single-owner bearer token for /api/v1 (real auth: P4-01)
    signed_url_ttl: int = 600  # seconds (blueprint: presigned GET valid 10 minutes)
    heartbeat_fresh: int = 45  # a worker counts as online if it reported within this many seconds
    heartbeat_lost: int = 120  # an active job whose worker is silent this long is requeued once
    queue_expiry: int = 30 * 60  # queued jobs expire after 30 minutes without a worker
    wall_clock_limit: int = 15 * 60  # hard limit per job from start

    @classmethod
    def from_env(cls) -> Settings:
        secret = os.environ.get('MAKER_SECRET_KEY')
        return cls(
            data_dir=Path(os.environ.get('MAKER_DATA', 'var')).resolve(),
            # Without a configured key, signed URLs only survive until the process restarts.
            secret_key=secret.encode() if secret else secrets.token_bytes(32),
            worker_tokens=_worker_tokens(os.environ.get('MAKER_WORKER_TOKENS', '')),
            api_token=os.environ.get('MAKER_API_TOKEN') or None,
        )
