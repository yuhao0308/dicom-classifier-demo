from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from pydicom import Dataset, FileDataset

from app.services.dicom_parser import (
    apply_windowing,
    parse_series,
    read_slices,
    sort_slices,
    strip_phi_tags,
    validate_modality,
)


def test_strip_phi_tags_removes_phi_and_logs_warning(caplog: pytest.LogCaptureFixture) -> None:
    dataset = Dataset()
    dataset.PatientName = "Patient^Example"
    dataset.PatientID = "123456"
    dataset.PatientComments = "Some comments"
    dataset.StudyDate = "20260228"
    dataset.AccessionNumber = "ACC-1"

    with caplog.at_level(logging.WARNING):
        strip_phi_tags(dataset)

    assert "PatientName" not in dataset
    assert "PatientID" not in dataset
    assert "StudyDate" not in dataset
    assert "AccessionNumber" not in dataset
    assert all(tag.group != 0x0010 for tag in dataset.keys())
    assert any(record.message == "phi_tags_detected" for record in caplog.records)


def test_validate_modality_rejects_non_ct() -> None:
    dataset = Dataset()
    dataset.Modality = "MR"

    with pytest.raises(ValueError, match="Only CT modality is supported."):
        validate_modality(dataset)


def test_sort_slices_uses_instance_number_then_slice_location_then_filename() -> None:
    dataset_a = Dataset()
    dataset_a.filename = "slice_0002.dcm"
    dataset_a.InstanceNumber = 2

    dataset_b = Dataset()
    dataset_b.filename = "slice_0001.dcm"
    dataset_b.InstanceNumber = 1

    dataset_c = Dataset()
    dataset_c.filename = "slice_0004.dcm"
    dataset_c.SliceLocation = -20.5

    dataset_d = Dataset()
    dataset_d.filename = "slice_0003.dcm"
    dataset_d.SliceLocation = -30.0

    dataset_e = Dataset()
    dataset_e.filename = "slice_0006.dcm"

    dataset_f = Dataset()
    dataset_f.filename = "slice_0005.dcm"

    sorted_datasets = sort_slices(
        [dataset_a, dataset_b, dataset_c, dataset_d, dataset_e, dataset_f]
    )

    assert [dataset.filename for dataset in sorted_datasets] == [
        "slice_0001.dcm",
        "slice_0002.dcm",
        "slice_0003.dcm",
        "slice_0004.dcm",
        "slice_0005.dcm",
        "slice_0006.dcm",
    ]


def test_apply_windowing_converts_to_uint8() -> None:
    pixel_array = np.array([[-2000, -600, 150, 1000]], dtype=np.float32)

    windowed = apply_windowing(pixel_array, window_width=1500, window_center=-600)

    assert windowed.dtype == np.uint8
    assert np.array_equal(windowed, np.array([[0, 128, 255, 255]], dtype=np.uint8))


def test_read_slices_reads_and_strips_phi(
    tmp_path: Path,
    build_test_dataset: Callable[..., FileDataset],
    write_dataset: Callable[[Path, FileDataset], None],
) -> None:
    dataset_1 = build_test_dataset(
        np.array([[-1000, -600], [0, 200]], dtype=np.int16),
        instance_number=1,
    )
    dataset_2 = build_test_dataset(
        np.array([[-1200, -500], [50, 150]], dtype=np.int16),
        instance_number=2,
    )

    write_dataset(tmp_path / "slice_0001.dcm", dataset_1)
    write_dataset(tmp_path / "slice_0002.dcm", dataset_2)

    datasets = read_slices(tmp_path)

    assert len(datasets) == 2
    assert all(dataset.Modality == "CT" for dataset in datasets)
    assert all("PatientName" not in dataset for dataset in datasets)
    assert all("PatientID" not in dataset for dataset in datasets)


def test_read_slices_rejects_non_ct(
    tmp_path: Path,
    build_test_dataset: Callable[..., FileDataset],
    write_dataset: Callable[[Path, FileDataset], None],
) -> None:
    dataset = build_test_dataset(np.array([[-1000, 100]], dtype=np.int16), modality="MR")
    write_dataset(tmp_path / "slice_0001.dcm", dataset)

    with pytest.raises(ValueError, match="Only CT modality is supported."):
        read_slices(tmp_path)


def test_parse_series_sorts_and_windows(
    tmp_path: Path,
    build_test_dataset: Callable[..., FileDataset],
    write_dataset: Callable[[Path, FileDataset], None],
) -> None:
    first = build_test_dataset(
        np.array([[-1000, 0], [500, 1000]], dtype=np.int16),
        instance_number=2,
    )
    second = build_test_dataset(
        np.array([[-1300, -600], [150, 300]], dtype=np.int16),
        instance_number=1,
    )

    write_dataset(tmp_path / "slice_0001.dcm", first)
    write_dataset(tmp_path / "slice_0002.dcm", second)

    parsed = parse_series(tmp_path)

    assert len(parsed) == 2
    assert all(image.dtype == np.uint8 for image in parsed)
    assert np.array_equal(parsed[0], np.array([[8, 128], [255, 255]], dtype=np.uint8))
    assert np.array_equal(parsed[1], np.array([[60, 230], [255, 255]], dtype=np.uint8))
