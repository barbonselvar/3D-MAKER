"""Structured API errors: {"error": {"code", "message", "retryable"}}."""
from __future__ import annotations

from fastapi import Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str, retryable: bool = False, headers: dict | None = None):
        super().__init__(message)
        self.status, self.code, self.message, self.retryable, self.headers = status, code, message, retryable, headers

    def body(self) -> dict:
        return {'error': {'code': self.code, 'message': self.message, 'retryable': self.retryable}}


async def api_error_handler(_: Request, exc: ApiError) -> JSONResponse:
    return JSONResponse(exc.body(), status_code=exc.status, headers=exc.headers)


async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Structured 422 for /api/v1; the legacy /api routes keep FastAPI's default body."""
    if not request.url.path.startswith('/api/v1'):
        return await request_validation_exception_handler(request, exc)
    first = exc.errors()[0] if exc.errors() else {}
    where = '.'.join(str(p) for p in first.get('loc', ()) if p != 'body')
    message = f"{where}: {first.get('msg', 'invalid value')}" if where else first.get('msg', 'Invalid request.')
    return JSONResponse({'error': {'code': 'REQUEST_INVALID', 'message': message, 'retryable': False}}, status_code=422)
