from time import perf_counter

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request


class TimingMiddleware(BaseHTTPMiddleware):
    """Attach request duration to the response headers."""

    async def dispatch(self, request: Request, call_next):
        started_at = perf_counter()
        response = await call_next(request)
        duration = perf_counter() - started_at
        response.headers["X-Response-Time"] = f"{duration:.3f}s"
        return response
