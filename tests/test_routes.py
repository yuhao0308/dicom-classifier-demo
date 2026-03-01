from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

from fastapi.testclient import TestClient


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
    assert metadata["status"] == "processing"
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
