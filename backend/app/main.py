"""
FastAPI Application Entry Point.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.app.core.config import settings
from backend.app.api.v1.router import api_v1_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle event handler for backend startup and shutdown."""
    print(f"[STARTUP] Initializing {settings.PROJECT_NAME} v{settings.VERSION}")
    print(f"[STARTUP] Running in {settings.ENVIRONMENT} mode")
    # Ensure temporary upload directory is ready
    settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    yield
    print(f"[SHUTDOWN] Terminating backend service cleanly.")


def create_application() -> FastAPI:
    """Application factory."""
    app = FastAPI(
        title=settings.PROJECT_NAME,
        version=settings.VERSION,
        description=(
            "AI-Powered Multimodal Vehicle Fault Detection and Diagnosis API. "
            "Provides endpoints for subsystem component localization and visible fault classification."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # Configure Cross-Origin Resource Sharing (CORS)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS if isinstance(settings.CORS_ORIGINS, list) else ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Attach API v1 Router
    app.include_router(api_v1_router, prefix=settings.API_V1_PREFIX)

    @app.get("/", tags=["Root"])
    async def root():
        return {
            "project": settings.PROJECT_NAME,
            "version": settings.VERSION,
            "docs": "/docs",
            "health": f"{settings.API_V1_PREFIX}/health",
        }

    return app


app = create_application()
