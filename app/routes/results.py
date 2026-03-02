from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse

from app.services.storage import get_job_dir, read_job_findings, read_job_metadata

router = APIRouter()

RESULTS_DISCLAIMER = "For reference only. Not medical advice. Clinician must use their judgment."


@router.get("/results/{job_id}")
async def get_results(job_id: str, request: Request) -> dict[str, object]:
    return _load_results_payload(job_id, request)


@router.get("/results/{job_id}/view", response_class=HTMLResponse)
async def view_results(job_id: str, request: Request) -> HTMLResponse:
    payload = _load_results_payload(job_id, request)
    return request.app.state.templates.TemplateResponse(
        "results.html",
        {"request": request, **payload},
    )


def _load_results_payload(job_id: str, request: Request) -> dict[str, object]:
    settings = request.app.state.settings
    job_dir = get_job_dir(settings, job_id)
    meta_path = job_dir / "meta.json"
    findings_path = job_dir / "findings.json"

    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="Job not found.")

    metadata = read_job_metadata(job_dir)
    if metadata.get("status") != "completed":
        raise HTTPException(status_code=404, detail="Results not found.")

    if not findings_path.exists():
        raise HTTPException(status_code=404, detail="Results not found.")

    findings_payload = read_job_findings(job_dir)
    return {
        "job_id": job_id,
        "disclaimer": RESULTS_DISCLAIMER,
        "total_slices": findings_payload.get("total_slices", metadata.get("slice_count", 0)),
        "abnormal_slices": findings_payload.get("abnormal_slices", []),
    }
