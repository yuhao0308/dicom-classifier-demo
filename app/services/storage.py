from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import Settings

JOB_METADATA_FILENAME = "meta.json"


def ensure_temp_dir(settings: Settings) -> Path:
    settings.temp_dir.mkdir(parents=True, exist_ok=True)
    return settings.temp_dir


def create_job_dir(settings: Settings, job_id: str) -> Path:
    job_dir = ensure_temp_dir(settings) / job_id
    job_dir.mkdir(parents=True, exist_ok=False)
    return job_dir


def save_dicom_slice(job_dir: Path, *, index: int, payload: bytes) -> Path:
    file_path = job_dir / f"slice_{index:04d}.dcm"
    file_path.write_bytes(payload)
    return file_path


def write_job_metadata(job_dir: Path, *, job_id: str, status: str, slice_count: int) -> Path:
    created_at = (
        datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    )
    metadata = {
        "job_id": job_id,
        "status": status,
        "slice_count": slice_count,
        "created_at": created_at,
    }
    metadata_path = job_dir / JOB_METADATA_FILENAME
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata_path


def read_job_metadata(job_dir: Path) -> dict[str, Any]:
    metadata_path = job_dir / JOB_METADATA_FILENAME
    return json.loads(metadata_path.read_text(encoding="utf-8"))
