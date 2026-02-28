from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response
from starlette.types import ASGIApp

from app.config import Settings, get_settings

BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"


class UploadSizeLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp, *, settings: Settings) -> None:
        super().__init__(app)
        self._max_upload_size_bytes = settings.max_upload_size_bytes

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        if request.url.path == "/upload":
            content_length = request.headers.get("content-length")
            if content_length is not None:
                try:
                    content_length_value = int(content_length)
                except ValueError:
                    content_length_value = 0

                if content_length_value > self._max_upload_size_bytes:
                    return JSONResponse(status_code=413, content={"detail": "Payload too large."})
        return await call_next(request)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.settings = get_settings()
    # Startup hook reserved for model loading and temp-dir initialization.
    yield
    # Shutdown hook reserved for temp file cleanup and resource teardown.


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="DICOM Classifier Demo", lifespan=lifespan)
    app.add_middleware(UploadSizeLimitMiddleware, settings=settings)
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    app.state.templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
