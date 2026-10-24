"""Structured API errors: {"error": {"code", "message", "retryable"}}."""
from __future__ import annotations

from fastapi import Request
from fastapi.responses import JSONResponse


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str, retryable: bool = False, headers: dict | None = None):
        super().__init__(message)
        self.status, self.code, self.message, self.retryable, self.headers = status, code, message, retryable, headers

    def body(self) -> dict:
        return {'error': {'code': self.code, 'message': self.message, 'retryable': self.retryable}}


async def api_error_handler(_: Request, exc: ApiError) -> JSONResponse:
    return JSONResponse(exc.body(), status_code=exc.status, headers=exc.headers)
