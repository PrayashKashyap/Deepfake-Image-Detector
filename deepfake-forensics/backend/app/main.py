"""FastAPI application factory.

Run (from the repository root, with the virtual environment active):
    uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000
OpenAPI / Swagger UI: http://127.0.0.1:8000/docs
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import analyses, evaluation, system
from app.core.config import APP_VERSION, Settings, get_settings
from app.core.container import Container
from app.core.errors import install_error_handlers
from app.core.security import require_api_key


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    logging.basicConfig(level=settings.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        app.state.container.jobs.shutdown()

    app = FastAPI(
        title="Deepfake Image Detection and Forensic Analysis API",
        version=APP_VERSION,
        description=(
            "REST API for model-based REAL/FAKE image classification (two tracks: AI-generated images and facial deepfakes), "
            "Grad-CAM explanations, image forensics, robustness demonstrations and research-result browsing.\n\n"
            "Class convention: 0 = REAL, 1 = FAKE. `fake_score` is the model's softmax output for FAKE — a model score, "
            "not a calibrated probability. Predictions are probabilistic and are not legal or forensic proof."
        ),
        lifespan=lifespan,
    )
    app.state.container = Container.build(settings)
    app.dependency_overrides[get_settings] = lambda: settings
    install_error_handlers(app)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=False,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["Content-Type", "X-API-Key"],
        expose_headers=["Content-Disposition"],
    )
    deps = [Depends(require_api_key)]
    app.include_router(system.router, prefix="/api", dependencies=deps)
    app.include_router(analyses.router, prefix="/api", dependencies=deps)
    app.include_router(evaluation.router, prefix="/api", dependencies=deps)

    @app.get("/", include_in_schema=False)
    def root():
        return {"name": settings.app_name, "docs": "/docs", "health": "/api/health"}

    return app


app = create_app()
