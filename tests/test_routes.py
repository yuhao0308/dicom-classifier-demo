from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

from fastapi.testclient import TestClient

from app.routes.results import RESULTS_DISCLAIMER


def _dicom_payload() -> bytes:
    return (b"\x00" * 128) + b"DICM" + (b"\x00" * 32)


def _zip_payload(entries: dict[str, bytes]) -> bytes:
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for filename, content in entries.items():
            archive.writestr(filename, content)
    return payload.getvalue()


def test_health_returns_ok(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_upload_page_renders_form_and_disclaimer(client: TestClient) -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert "DICOM Classifier Demo" in response.text
    assert 'id="upload-form"' in response.text
    assert RESULTS_DISCLAIMER in response.text


def test_upload_valid_zip_returns_accepted(client: TestClient, tmp_path: Path) -> None:
    response = client.post(
        "/upload",
        files={
            "file": (
                "series.zip",
                _zip_payload(
                    {
                        "study/slice1.dcm": _dicom_payload(),
                        "study/slice2.dcm": _dicom_payload(),
                    }
                ),
                "application/zip",
            )
        },
    )

    assert response.status_code == 202
    response_body = response.json()
    assert response_body["status"] == "processing"
    assert response_body["slice_count"] == 2
    assert response_body["message"] == "Upload accepted. Processing started."

    job_id = response_body["job_id"]
    job_dir = tmp_path / job_id
    assert job_dir.exists()
    assert sorted(path.name for path in job_dir.glob("slice_*.dcm")) == [
        "slice_0001.dcm",
        "slice_0002.dcm",
    ]

    metadata = json.loads((job_dir / "meta.json").read_text(encoding="utf-8"))
    assert metadata["job_id"] == job_id
    assert metadata["status"] in {"processing", "completed", "failed"}
    assert metadata["slice_count"] == 2


def test_upload_non_dicom_returns_400(client: TestClient) -> None:
    response = client.post(
        "/upload",
        files={
            "file": (
                "series.zip",
                _zip_payload({"study/not_a_dicom.dcm": b"not-a-dicom"}),
                "application/zip",
            )
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Archive contains non-DICOM content."


def test_upload_without_file_returns_400(client: TestClient) -> None:
    response = client.post("/upload")
    assert response.status_code == 400
    assert response.json()["detail"] == "No file was uploaded."


def test_get_job_returns_status_payload(client: TestClient, tmp_path: Path) -> None:
    job_id = "job-123"
    job_dir = tmp_path / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "meta.json").write_text(
        json.dumps(
            {
                "job_id": job_id,
                "status": "processing",
                "progress": 35,
                "slice_count": 10,
                "created_at": "2026-03-01T00:00:00Z",
            }
        ),
        encoding="utf-8",
    )

    response = client.get(f"/jobs/{job_id}")

    assert response.status_code == 200
    assert response.json() == {
        "job_id": job_id,
        "status": "processing",
        "progress": 35,
        "created_at": "2026-03-01T00:00:00Z",
    }


def test_get_job_not_found_returns_404(client: TestClient) -> None:
    response = client.get("/jobs/missing-job")
    assert response.status_code == 404


def test_get_results_returns_contract_when_completed(client: TestClient, tmp_path: Path) -> None:
    job_id = "job-completed"
    job_dir = tmp_path / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "meta.json").write_text(
        json.dumps(
            {
                "job_id": job_id,
                "status": "completed",
                "progress": 100,
                "slice_count": 64,
                "created_at": "2026-03-01T00:00:00Z",
            }
        ),
        encoding="utf-8",
    )
    (job_dir / "findings.json").write_text(
        json.dumps(
            {
                "total_slices": 64,
                "abnormal_slices": [
                    {
                        "slice_index": 47,
                        "confidence": 0.82,
                        "finding": "Suspicious region detected in slice 47 (confidence: 0.82).",
                        "bbox": {"x": 120, "y": 95, "width": 45, "height": 38},
                        "image_url": f"/job-media/{job_id}/slice_0047.png",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    response = client.get(f"/results/{job_id}")

    assert response.status_code == 200
    payload = response.json()
    assert payload["job_id"] == job_id
    assert payload["disclaimer"] == RESULTS_DISCLAIMER
    assert payload["total_slices"] == 64
    assert len(payload["abnormal_slices"]) == 1
    assert payload["abnormal_slices"][0]["slice_index"] == 47


def test_get_results_for_processing_job_returns_404(client: TestClient, tmp_path: Path) -> None:
    job_id = "job-processing"
    job_dir = tmp_path / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "meta.json").write_text(
        json.dumps(
            {
                "job_id": job_id,
                "status": "processing",
                "progress": 60,
                "slice_count": 22,
                "created_at": "2026-03-01T00:00:00Z",
            }
        ),
        encoding="utf-8",
    )

    response = client.get(f"/results/{job_id}")
    assert response.status_code == 404


def test_results_html_page_renders_disclaimer_and_finding(
    client: TestClient,
    tmp_path: Path,
) -> None:
    job_id = "job-html"
    job_dir = tmp_path / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "meta.json").write_text(
        json.dumps(
            {
                "job_id": job_id,
                "status": "completed",
                "progress": 100,
                "slice_count": 12,
                "created_at": "2026-03-01T00:00:00Z",
            }
        ),
        encoding="utf-8",
    )
    (job_dir / "findings.json").write_text(
        json.dumps(
            {
                "total_slices": 12,
                "abnormal_slices": [
                    {
                        "slice_index": 3,
                        "confidence": 0.71,
                        "finding": "Suspicious region detected in slice 3 (confidence: 0.71).",
                        "bbox": {"x": 1, "y": 2, "width": 3, "height": 4},
                        "image_url": f"/job-media/{job_id}/slice_0003.png",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    response = client.get(f"/results/{job_id}/view")

    assert response.status_code == 200
    assert RESULTS_DISCLAIMER in response.text
    assert "Suspicious region detected in slice 3 (confidence: 0.71)." in response.text
