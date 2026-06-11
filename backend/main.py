"""MODULE 10 — FastAPI application entry point.

Run locally:  uvicorn backend.main:app --reload
Docs:         http://localhost:8000/docs
"""
from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routers import ALL_ROUTERS
from ml.utils import get_logger

logger = get_logger(__name__)


def create_app() -> FastAPI:
    app = FastAPI(
        title="NIFTY-50 Investment Intelligence API",
        description="Decision-support analytics over historical NIFTY-50 market data: "
                    "forecasting, portfolio construction, risk, anomalies, "
                    "explainable recommendations.",
        version="1.0.0",
    )

    origins = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    for router in ALL_ROUTERS:
        app.include_router(router)

    @app.get("/health", tags=["meta"])
    def health():
        return {"status": "ok"}

    @app.get("/", tags=["meta"])
    def root():
        return {"name": app.title, "version": app.version, "docs": "/docs"}

    return app


app = create_app()
