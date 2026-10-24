"""Per-app services shared by the routers (settings, database, storage)."""
from __future__ import annotations

import hmac
from dataclasses import dataclass

from fastapi import Request

from .config import Settings
from .db import Database
from .errors import ApiError
from .storage import LocalStorage


@dataclass
class Context:
    settings: Settings
    db: Database
    storage: LocalStorage

    @classmethod
    def create(cls, settings: Settings) -> Context:
        return cls(
            settings=settings,
            db=Database(settings.data_dir / 'maker.db'),
            storage=LocalStorage(settings.data_dir / 'objects', settings.secret_key, settings.signed_url_ttl),
        )


def ctx(request: Request) -> Context:
    return request.app.state.ctx


def current_owner(request: Request) -> str:
    """Single-owner identity until real auth lands (P4-01).

    With MAKER_API_TOKEN set, every /api/v1 call needs `Authorization: Bearer <token>`.
    Without it the API runs in local single-user mode and everything belongs to "local".
    """
    token = request.app.state.ctx.settings.api_token
    if not token:
        return 'local'
    supplied = request.headers.get('authorization', '').removeprefix('Bearer ').strip()
    if not hmac.compare_digest(supplied.encode(), token.encode()):
        raise ApiError(401, 'UNAUTHORIZED', 'A valid API token is required.')
    return 'owner'
