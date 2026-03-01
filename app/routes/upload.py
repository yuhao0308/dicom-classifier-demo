from __future__ import annotations

import io
import shutil
import zipfile
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.config import get_settings
from app.services.storage import create_job_dir, save_dicom_slice, write_job_metadata

router = APIRouter()

_DICOM_MAGIC_START = 128
_DICOM_MAGIC_END = 132
_DICOM_MAGIC = b"DICM"


def _has_dicom_magic(payload: bytes) -> bool:
    return (
        len(payload) >= _DICOM_MAGIC_END
        and payload[_DICOM_MAGIC_START:_DICOM_MAGIC_END] == _DICOM_MAGIC
    )


@router.post("/upload", status_code=status.HTTP_202_ACCEPTED)
async def upload(file: UploadFile | None = File(default=None)) -> dict[str, str | int]:
    if file is None:
        raise HTTPException(status_code=400, detail="No file was uploaded.")

    settings = get_settings()
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

    write_job_metadata(job_dir, job_id=job_id, status="processing", slice_count=slice_count)

    return {
        "job_id": job_id,
        "status": "processing",
        "slice_count": slice_count,
        "message": "Upload accepted. Processing started.",
    }
