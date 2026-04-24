from time import perf_counter

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from app.core.logger import logger


class LoggingMiddleware(BaseHTTPMiddleware):
    """Log each request with method, path, client IP, status, and duration."""

    async def dispatch(self, request: Request, call_next):
        started_at = perf_counter()
        response = await call_next(request)
        duration = perf_counter() - started_at

        client_ip = request.client.host if request.client else "unknown"
        path = request.url.path
        if request.url.query:
            path = f"{path}?{request.url.query}"

        logger.info(
            f"{request.method} {path} | {client_ip} | {response.status_code} | {duration:.3f}s"
        )
        return response
