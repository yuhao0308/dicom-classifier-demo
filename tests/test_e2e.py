from __future__ import annotations

import io
import time
import zipfile
from collections.abc import Callable

import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydicom import FileDataset

from app.routes.results import RESULTS_DISCLAIMER


def _zip_payload(entries: dict[str, bytes]) -> bytes:
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for filename, content in entries.items():
            archive.writestr(filename, content)
    return payload.getvalue()


def test_upload_poll_and_fetch_results(
    client: TestClient,
    build_test_dataset: Callable[..., FileDataset],
    dataset_to_bytes: Callable[[FileDataset], bytes],
) -> None:
    dataset_1 = build_test_dataset(
        np.array([[-900, -600], [-300, 120]], dtype=np.int16),
        instance_number=1,
    )
    dataset_2 = build_test_dataset(
        np.array([[-1200, -800], [-200, 400]], dtype=np.int16),
        instance_number=2,
    )
    archive_payload = _zip_payload(
        {
            "study/slice1.dcm": dataset_to_bytes(dataset_1),
            "study/slice2.dcm": dataset_to_bytes(dataset_2),
        }
    )

    upload_response = client.post(
        "/upload",
        files={"file": ("series.zip", archive_payload, "application/zip")},
    )
    assert upload_response.status_code == 202
    job_id = upload_response.json()["job_id"]

    results_response = client.get(f"/results/{job_id}")
    if results_response.status_code != 200:
        for _ in range(10):
            job_response = client.get(f"/jobs/{job_id}")
            assert job_response.status_code == 200
            job_status = job_response.json()["status"]
            if job_status == "completed":
                break
            if job_status == "failed":
                pytest.fail("Background pipeline marked the job as failed.")
            time.sleep(0.5)

        results_response = client.get(f"/results/{job_id}")

    assert results_response.status_code == 200
    results_payload = results_response.json()
    assert results_payload["job_id"] == job_id
    assert results_payload["disclaimer"] == RESULTS_DISCLAIMER
    assert results_payload["total_slices"] == 2
    assert isinstance(results_payload["abnormal_slices"], list)
