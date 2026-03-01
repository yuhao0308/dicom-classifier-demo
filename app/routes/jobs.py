from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from app.services.storage import get_job_dir, read_job_metadata

router = APIRouter()


@router.get("/jobs/{job_id}")
async def get_job(job_id: str, request: Request) -> dict[str, object]:
    settings = request.app.state.settings
    job_dir = get_job_dir(settings, job_id)
    metadata_path = job_dir / "meta.json"
    if not metadata_path.exists():
        raise HTTPException(status_code=404, detail="Job not found.")

    metadata = read_job_metadata(job_dir)
    return {
        "job_id": metadata["job_id"],
        "status": metadata.get("status", "processing"),
        "progress": metadata.get("progress", 0),
        "created_at": metadata.get("created_at"),
    }
