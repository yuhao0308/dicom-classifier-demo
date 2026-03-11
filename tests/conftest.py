from __future__ import annotations

import io
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np
import pytest
import torch
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydicom import FileDataset
from pydicom.dataset import FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid
from torch import nn
from torchvision.models import resnet18

from app.config import get_settings
from app.main import create_app


def _build_test_model() -> nn.Module:
    """Build a ResNet-18 with the same architecture as the training script.

    Must match _build_classifier() in app/services/inference.py:
      - conv1: 3x3/s1 (not 7x7/s2)
      - maxpool: Identity (removed)
      - fc: Linear(512, 2)
    """
    base_model = resnet18(weights=None)
    base_model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    base_model.bn1 = nn.BatchNorm2d(64)
    base_model.maxpool = nn.Identity()
    base_model.fc = nn.Linear(base_model.fc.in_features, 2)
    return base_model


@pytest.fixture
def app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[FastAPI]:
    model_path = tmp_path / "classifier.pt"
    base_model = _build_test_model()
    torch.save(base_model.state_dict(), model_path)

    monkeypatch.setenv("TEMP_DIR", str(tmp_path))
    monkeypatch.setenv("MODEL_PATH", str(model_path))
    monkeypatch.setenv("USE_GPU", "false")
    get_settings.cache_clear()
    test_app = create_app()
    yield test_app
    get_settings.cache_clear()


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def build_test_dataset() -> Callable[..., FileDataset]:
    def _build(
        pixel_array: np.ndarray,
        *,
        modality: str = "CT",
        instance_number: int | None = None,
        slice_location: float | None = None,
        rescale_slope: float = 1.0,
        rescale_intercept: float = 0.0,
    ) -> FileDataset:
        file_meta = FileMetaDataset()
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        file_meta.MediaStorageSOPClassUID = CTImageStorage
        file_meta.MediaStorageSOPInstanceUID = generate_uid()
        file_meta.ImplementationClassUID = generate_uid()

        dataset = FileDataset("", {}, file_meta=file_meta, preamble=b"\0" * 128)
        dataset.SOPClassUID = file_meta.MediaStorageSOPClassUID
        dataset.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
        dataset.Modality = modality
        dataset.PatientName = "Patient^Example"
        dataset.PatientID = "123456"
        dataset.PatientBirthDate = "19900101"
        dataset.PatientSex = "O"
        dataset.PatientAge = "034Y"
        dataset.InstitutionName = "Example Hospital"
        dataset.InstitutionAddress = "123 Main Street"
        dataset.ReferringPhysicianName = "Referrer^Doc"
        dataset.StudyDate = "20260228"
        dataset.StudyTime = "120000"
        dataset.AccessionNumber = "ACC-1"
        dataset.Rows = int(pixel_array.shape[0])
        dataset.Columns = int(pixel_array.shape[1])
        dataset.SamplesPerPixel = 1
        dataset.PhotometricInterpretation = "MONOCHROME2"
        dataset.BitsAllocated = 16
        dataset.BitsStored = 16
        dataset.HighBit = 15
        dataset.PixelRepresentation = 1
        dataset.RescaleSlope = rescale_slope
        dataset.RescaleIntercept = rescale_intercept
        if instance_number is not None:
            dataset.InstanceNumber = instance_number
        if slice_location is not None:
            dataset.SliceLocation = slice_location
        dataset.PixelData = pixel_array.astype(np.int16).tobytes()
        dataset.is_implicit_VR = False
        dataset.is_little_endian = True
        return dataset

    return _build


@pytest.fixture
def write_dataset() -> Callable[[Path, FileDataset], None]:
    def _write(path: Path, dataset: FileDataset) -> None:
        dataset.save_as(path, write_like_original=False)

    return _write


@pytest.fixture
def dataset_to_bytes() -> Callable[[FileDataset], bytes]:
    def _to_bytes(dataset: FileDataset) -> bytes:
        buffer = io.BytesIO()
        dataset.save_as(buffer, write_like_original=False)
        return buffer.getvalue()

    return _to_bytes
