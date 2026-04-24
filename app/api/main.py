from fastapi import FastAPI

from app.api.middleware import register_api_middleware
from app.api.routes.chat import router as chat_router
from app.api.routes.health import router as health_router
from app.api.routes.ingest import router as ingest_router
from app.api.routes.papers import router as papers_router


def create_app() -> FastAPI:
    app = FastAPI(
        title="ArxivLens API",
        version="0.1.0",
        description="Standalone backend for ArxivLens business logic.",
    )

    register_api_middleware(app)

    app.include_router(health_router)
    app.include_router(papers_router)
    app.include_router(ingest_router)
    app.include_router(chat_router)
    return app


app = create_app()
