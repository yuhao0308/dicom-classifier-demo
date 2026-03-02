from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.config import Settings, get_settings
from app.routes.jobs import router as jobs_router
from app.routes.results import router as results_router
from app.routes.upload import router as upload_router
from app.services.inference import load_model
from app.services.storage import ensure_temp_dir

BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"


class _UploadTooLargeError(Exception):
    pass


class UploadSizeLimitMiddleware:
    def __init__(self, app: ASGIApp, *, settings: Settings) -> None:
        self.app = app
        self._max_upload_size_bytes = settings.max_upload_size_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["path"] != "/upload":
            await self.app(scope, receive, send)
            return

        content_length = self._get_content_length(scope)
        if content_length > self._max_upload_size_bytes:
            response = JSONResponse(status_code=413, content={"detail": "Payload too large."})
            await response(scope, receive, send)
            return

        bytes_received = 0

        async def limited_receive() -> Message:
            nonlocal bytes_received
            message = await receive()
            if message["type"] == "http.request":
                body = message.get("body", b"")
                bytes_received += len(body)
                if bytes_received > self._max_upload_size_bytes:
                    raise _UploadTooLargeError
            return message

        try:
            await self.app(scope, limited_receive, send)
        except _UploadTooLargeError:
            response = JSONResponse(status_code=413, content={"detail": "Payload too large."})
            await response(scope, receive, send)

    @staticmethod
    def _get_content_length(scope: Scope) -> int:
        for header_name, header_value in scope["headers"]:
            if header_name.lower() == b"content-length":
                try:
                    return int(header_value.decode("latin-1"))
                except ValueError:
                    return 0
        return 0


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    app.state.settings = settings
    ensure_temp_dir(settings)
    app.state.model = load_model(settings.model_path, use_gpu=settings.use_gpu)
    app.state.inference_batch_size = settings.inference_batch_size
    # Startup hook reserved for model loading and temp-dir initialization.
    yield
    app.state.model = None
    # Shutdown hook reserved for temp file cleanup and resource teardown.


def create_app() -> FastAPI:
    settings = get_settings()
    ensure_temp_dir(settings)
    app = FastAPI(title="DICOM Classifier Demo", lifespan=lifespan)
    app.add_middleware(UploadSizeLimitMiddleware, settings=settings)
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    app.mount("/job-media", StaticFiles(directory=str(settings.temp_dir)), name="job-media")
    app.state.templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
    app.include_router(upload_router)
    app.include_router(jobs_router)
    app.include_router(results_router)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
