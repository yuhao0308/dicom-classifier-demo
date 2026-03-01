from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pydicom
from pydicom import Dataset
from pydicom.datadict import tag_for_keyword
from pydicom.errors import InvalidDicomError
from pydicom.tag import Tag

LOGGER = logging.getLogger(__name__)
_PATIENT_GROUP = 0x0010
_DEFAULT_WINDOW_WIDTH = 1500.0
_DEFAULT_WINDOW_CENTER = -600.0
_COMMON_PHI_KEYWORDS = (
    "PatientName",
    "PatientID",
    "PatientBirthDate",
    "PatientSex",
    "PatientAge",
    "InstitutionName",
    "InstitutionAddress",
    "ReferringPhysicianName",
    "StudyDate",
    "StudyTime",
    "AccessionNumber",
)


def strip_phi_tags(dataset: Dataset) -> None:
    detected_tags: set[str] = set()

    for keyword in _COMMON_PHI_KEYWORDS:
        tag_value = tag_for_keyword(keyword)
        if tag_value is None:
            continue
        tag = Tag(tag_value)
        if tag in dataset:
            detected_tags.add(keyword)
            del dataset[tag]

    for tag in list(dataset.keys()):
        if tag.group == _PATIENT_GROUP:
            detected_tags.add(str(tag))
            del dataset[tag]

    if detected_tags:
        LOGGER.warning("phi_tags_detected", extra={"phi_tags": sorted(detected_tags)})


def validate_modality(dataset: Dataset) -> None:
    modality = str(getattr(dataset, "Modality", "")).upper()
    if modality != "CT":
        raise ValueError("Only CT modality is supported.")


def read_slices(job_dir: Path) -> list[Dataset]:
    slice_paths = sorted(job_dir.glob("slice_*.dcm"))
    if not slice_paths:
        raise ValueError("No DICOM slices found in job directory.")

    datasets: list[Dataset] = []
    for path in slice_paths:
        try:
            dataset = pydicom.dcmread(path)
        except InvalidDicomError as exc:
            raise ValueError(f"Invalid DICOM file: {path.name}") from exc

        strip_phi_tags(dataset)
        validate_modality(dataset)
        datasets.append(dataset)

    return datasets


def sort_slices(datasets: list[Dataset]) -> list[Dataset]:
    def sort_key(dataset: Dataset) -> tuple[int, float | str, str]:
        instance_number = _to_float(getattr(dataset, "InstanceNumber", None))
        if instance_number is not None:
            return (0, instance_number, _filename(dataset))

        slice_location = _to_float(getattr(dataset, "SliceLocation", None))
        if slice_location is not None:
            return (1, slice_location, _filename(dataset))

        return (2, _filename(dataset), _filename(dataset))

    return sorted(datasets, key=sort_key)


def apply_windowing(
    pixel_array: np.ndarray,
    window_width: float = _DEFAULT_WINDOW_WIDTH,
    window_center: float = _DEFAULT_WINDOW_CENTER,
) -> np.ndarray:
    if window_width <= 0:
        raise ValueError("window_width must be > 0.")

    image = np.asarray(pixel_array, dtype=np.float32)
    lower = window_center - (window_width / 2.0)
    upper = window_center + (window_width / 2.0)
    clipped = np.clip(image, lower, upper)
    scaled = (clipped - lower) / (upper - lower)
    return np.rint(scaled * 255.0).astype(np.uint8)


def parse_series(
    job_dir: Path,
    *,
    window_width: float = _DEFAULT_WINDOW_WIDTH,
    window_center: float = _DEFAULT_WINDOW_CENTER,
) -> list[np.ndarray]:
    datasets = sort_slices(read_slices(job_dir))
    parsed_slices: list[np.ndarray] = []

    for dataset in datasets:
        try:
            pixel_array = np.asarray(dataset.pixel_array, dtype=np.float32)
        except Exception as exc:  # pragma: no cover - pydicom backend errors vary by plugin
            raise ValueError("Failed to decode DICOM pixel data.") from exc

        if pixel_array.ndim != 2:
            raise ValueError("Expected a 2D pixel array for each CT slice.")

        slope = _to_float(getattr(dataset, "RescaleSlope", 1.0)) or 1.0
        intercept = _to_float(getattr(dataset, "RescaleIntercept", 0.0)) or 0.0
        hu_array = (pixel_array * slope) + intercept
        parsed_slices.append(
            apply_windowing(
                hu_array,
                window_width=window_width,
                window_center=window_center,
            )
        )

    return parsed_slices


def _to_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _filename(dataset: Dataset) -> str:
    filename = getattr(dataset, "filename", "")
    if not filename:
        return ""
    return Path(str(filename)).name
