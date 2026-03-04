from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from app.config import Settings

JOB_METADATA_FILENAME = "meta.json"
JOB_FINDINGS_FILENAME = "findings.json"


def ensure_temp_dir(settings: Settings) -> Path:
    settings.temp_dir.mkdir(parents=True, exist_ok=True)
    return settings.temp_dir


def create_job_dir(settings: Settings, job_id: str) -> Path:
    job_dir = ensure_temp_dir(settings) / job_id
    job_dir.mkdir(parents=True, exist_ok=False)
    return job_dir


def get_job_dir(settings: Settings, job_id: str) -> Path:
    return settings.temp_dir / job_id


def save_dicom_slice(job_dir: Path, *, index: int, payload: bytes) -> Path:
    file_path = job_dir / f"slice_{index:04d}.dcm"
    file_path.write_bytes(payload)
    return file_path


def write_job_metadata(
    job_dir: Path,
    *,
    job_id: str,
    status: str,
    slice_count: int,
    progress: int = 0,
) -> Path:
    created_at = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    metadata = {
        "job_id": job_id,
        "status": status,
        "progress": progress,
        "slice_count": slice_count,
        "created_at": created_at,
    }
    metadata_path = job_dir / JOB_METADATA_FILENAME
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata_path


def read_job_metadata(job_dir: Path) -> dict[str, object]:
    metadata_path = job_dir / JOB_METADATA_FILENAME
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def update_job_metadata(job_dir: Path, **updates: object) -> Path:
    metadata = read_job_metadata(job_dir)
    metadata.update(updates)
    metadata_path = job_dir / JOB_METADATA_FILENAME
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata_path


def write_job_findings(
    job_dir: Path,
    *,
    total_slices: int,
    abnormal_slices: list[dict[str, object]],
    evaluation: dict[str, object] | None = None,
    missed_slices: list[dict[str, object]] | None = None,
    annotation_source: str | None = None,
) -> Path:
    findings_payload: dict[str, object] = {
        "total_slices": total_slices,
        "abnormal_slices": abnormal_slices,
    }
    if evaluation is not None:
        findings_payload["evaluation"] = evaluation
    if missed_slices is not None:
        findings_payload["missed_slices"] = missed_slices
    if annotation_source is not None:
        findings_payload["annotation_source"] = annotation_source
    findings_path = job_dir / JOB_FINDINGS_FILENAME
    findings_path.write_text(json.dumps(findings_payload, indent=2), encoding="utf-8")
    return findings_path


def read_job_findings(job_dir: Path) -> dict[str, object]:
    findings_path = job_dir / JOB_FINDINGS_FILENAME
    return json.loads(findings_path.read_text(encoding="utf-8"))
