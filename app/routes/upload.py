from __future__ import annotations

import io
import logging
import shutil
import zipfile
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, Request, UploadFile, status
from fastapi.responses import HTMLResponse
from PIL import Image

from app.config import get_settings
from app.services.dicom_parser import parse_series
from app.services.inference import InferenceModel, load_model, run_inference
from app.services.postprocess import postprocess_results
from app.services.storage import (
    create_job_dir,
    save_dicom_slice,
    update_job_metadata,
    write_job_findings,
    write_job_metadata,
)

router = APIRouter()
LOGGER = logging.getLogger(__name__)

_DICOM_MAGIC_START = 128
_DICOM_MAGIC_END = 132
_DICOM_MAGIC = b"DICM"
_MEDIA_URL_PREFIX = "/job-media"
_DISCLAIMER = "For reference only. Not medical advice. Clinician must use their judgment."


def _has_dicom_magic(payload: bytes) -> bool:
    return (
        len(payload) >= _DICOM_MAGIC_END
        and payload[_DICOM_MAGIC_START:_DICOM_MAGIC_END] == _DICOM_MAGIC
    )


@router.get("/", response_class=HTMLResponse)
async def upload_page(request: Request) -> HTMLResponse:
    return request.app.state.templates.TemplateResponse(
        "index.html",
        {"request": request, "disclaimer": _DISCLAIMER},
    )


@router.post("/upload", status_code=status.HTTP_202_ACCEPTED)
async def upload(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile | None = File(default=None),
) -> dict[str, str | int]:
    if file is None:
        raise HTTPException(status_code=400, detail="No file was uploaded.")

    settings = (
        request.app.state.settings if hasattr(request.app.state, "settings") else get_settings()
    )
    archive_payload = await file.read()

    if not archive_payload:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    if len(archive_payload) > settings.max_upload_size_bytes:
        raise HTTPException(status_code=413, detail="Payload too large.")

    if not zipfile.is_zipfile(io.BytesIO(archive_payload)):
        raise HTTPException(status_code=400, detail="Only .zip archives are supported.")

    job_id = str(uuid4())
    job_dir = create_job_dir(settings, job_id)
    slice_count = 0

    try:
        with zipfile.ZipFile(io.BytesIO(archive_payload)) as archive:
            for entry in archive.infolist():
                if entry.is_dir():
                    continue

                with archive.open(entry) as content_file:
                    payload = content_file.read()

                if not _has_dicom_magic(payload):
                    raise HTTPException(
                        status_code=400,
                        detail="Archive contains non-DICOM content.",
                    )

                slice_count += 1
                if slice_count > settings.max_slices:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Archive exceeds MAX_SLICES ({settings.max_slices}).",
                    )
                save_dicom_slice(job_dir, index=slice_count, payload=payload)
    except zipfile.BadZipFile as exc:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail="Only .zip archives are supported.") from exc
    except HTTPException:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise
    except OSError as exc:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail="Failed to persist upload.") from exc

    if slice_count == 0:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail="Archive has no DICOM files.")

    write_job_metadata(
        job_dir,
        job_id=job_id,
        status="processing",
        slice_count=slice_count,
        progress=0,
    )
    model = getattr(
        request.app.state,
        "model",
        load_model(settings.model_path, use_gpu=settings.use_gpu),
    )
    batch_size = int(
        getattr(request.app.state, "inference_batch_size", settings.inference_batch_size)
    )
    background_tasks.add_task(_run_processing_pipeline, job_dir, model, batch_size)

    return {
        "job_id": job_id,
        "status": "processing",
        "slice_count": slice_count,
        "message": "Upload accepted. Processing started.",
    }


def _run_processing_pipeline(job_dir: Path, model: InferenceModel, batch_size: int) -> None:
    try:
        update_job_metadata(job_dir, status="processing", progress=25)
        slices = parse_series(job_dir)
        update_job_metadata(job_dir, progress=55, slice_count=len(slices))
        inference_results = run_inference(
            model,
            slices,
            batch_size=batch_size,
        )
        findings = postprocess_results(slices, inference_results)
        abnormal_slices: list[dict[str, object]] = []

        for finding in findings:
            image_name = f"slice_{finding.slice_index:04d}.png"
            image_path = job_dir / image_name
            Image.fromarray(finding.image, mode="RGB").save(image_path, format="PNG")
            abnormal_slices.append(
                {
                    "slice_index": finding.slice_index,
                    "confidence": round(float(finding.confidence), 4),
                    "finding": finding.finding,
                    "bbox": {
                        "x": finding.bbox.x,
                        "y": finding.bbox.y,
                        "width": finding.bbox.width,
                        "height": finding.bbox.height,
                    },
                    "image_url": f"{_MEDIA_URL_PREFIX}/{job_dir.name}/{image_name}",
                }
            )

        write_job_findings(
            job_dir,
            total_slices=len(slices),
            abnormal_slices=abnormal_slices,
        )
        update_job_metadata(job_dir, status="completed", progress=100, slice_count=len(slices))
    except Exception as exc:  # pragma: no cover - exercised by integration paths
        LOGGER.exception("pipeline_failed", extra={"job_dir": str(job_dir)})
        update_job_metadata(job_dir, status="failed", progress=100, error=str(exc))
