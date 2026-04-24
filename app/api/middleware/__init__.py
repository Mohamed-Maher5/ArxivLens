from fastapi import FastAPI

from app.api.middleware.logging import LoggingMiddleware
from app.api.middleware.timing import TimingMiddleware


def register_api_middleware(app: FastAPI) -> None:
    """Register the API middleware stack."""

    app.add_middleware(TimingMiddleware)
    app.add_middleware(LoggingMiddleware)
