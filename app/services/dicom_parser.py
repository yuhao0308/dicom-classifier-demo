from __future__ import annotations

import logging
from dataclasses import dataclass
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


@dataclass(frozen=True, slots=True)
class SliceMetadata:
    """Metadata extracted from a DICOM slice for annotation mapping."""

    sop_instance_uid: str
    slice_index: int
    image_position_z: float | None
    patient_id: str


def _extract_sop_uid(dataset: Dataset) -> str:
    """Extract SOP Instance UID from a DICOM dataset (before PHI stripping)."""
    return str(getattr(dataset, "SOPInstanceUID", ""))


def _extract_patient_id(dataset: Dataset) -> str:
    """Extract Patient ID from a DICOM dataset (before PHI stripping)."""
    return str(getattr(dataset, "PatientID", ""))


def _extract_image_position_z(dataset: Dataset) -> float | None:
    """Extract the Z coordinate from ImagePositionPatient, if present."""
    ipp = getattr(dataset, "ImagePositionPatient", None)
    if ipp is not None and len(ipp) >= 3:
        try:
            return float(ipp[2])
        except (TypeError, ValueError, IndexError):
            return None
    return _to_float(getattr(dataset, "SliceLocation", None))


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


def read_slices_with_metadata(job_dir: Path) -> tuple[list[Dataset], list[dict[str, object]]]:
    """Read DICOM slices and extract metadata *before* PHI stripping.

    Returns the datasets (with PHI stripped) and a parallel list of raw
    metadata dicts (sop_uid, patient_id, image_position_z) captured before
    the PHI tags were removed.
    """
    slice_paths = sorted(job_dir.glob("slice_*.dcm"))
    if not slice_paths:
        raise ValueError("No DICOM slices found in job directory.")

    datasets: list[Dataset] = []
    raw_metadata: list[dict[str, object]] = []

    for path in slice_paths:
        try:
            dataset = pydicom.dcmread(path)
        except InvalidDicomError as exc:
            raise ValueError(f"Invalid DICOM file: {path.name}") from exc

        # Extract metadata BEFORE stripping PHI
        raw_metadata.append(
            {
                "sop_uid": _extract_sop_uid(dataset),
                "patient_id": _extract_patient_id(dataset),
                "image_position_z": _extract_image_position_z(dataset),
            }
        )

        strip_phi_tags(dataset)
        validate_modality(dataset)
        datasets.append(dataset)

    return datasets, raw_metadata


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


def _sort_with_metadata(
    datasets: list[Dataset],
    raw_metadata: list[dict[str, object]],
) -> tuple[list[Dataset], list[dict[str, object]]]:
    """Sort datasets and keep raw_metadata in sync."""
    paired = list(zip(datasets, raw_metadata, strict=True))

    def sort_key(pair: tuple[Dataset, dict[str, object]]) -> tuple[int, float | str, str]:
        dataset = pair[0]
        instance_number = _to_float(getattr(dataset, "InstanceNumber", None))
        if instance_number is not None:
            return (0, instance_number, _filename(dataset))
        slice_location = _to_float(getattr(dataset, "SliceLocation", None))
        if slice_location is not None:
            return (1, slice_location, _filename(dataset))
        return (2, _filename(dataset), _filename(dataset))

    paired.sort(key=sort_key)
    sorted_datasets = [p[0] for p in paired]
    sorted_meta = [p[1] for p in paired]
    return sorted_datasets, sorted_meta


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


def _process_dataset(
    dataset: Dataset,
    window_width: float,
    window_center: float,
) -> np.ndarray:
    """Convert a DICOM dataset to a windowed uint8 pixel array."""
    try:
        pixel_array = np.asarray(dataset.pixel_array, dtype=np.float32)
    except Exception as exc:  # pragma: no cover
        raise ValueError("Failed to decode DICOM pixel data.") from exc

    if pixel_array.ndim != 2:
        raise ValueError("Expected a 2D pixel array for each CT slice.")

    slope = _to_float(getattr(dataset, "RescaleSlope", 1.0)) or 1.0
    intercept = _to_float(getattr(dataset, "RescaleIntercept", 0.0)) or 0.0
    hu_array = (pixel_array * slope) + intercept
    return apply_windowing(hu_array, window_width=window_width, window_center=window_center)


def parse_series(
    job_dir: Path,
    *,
    window_width: float = _DEFAULT_WINDOW_WIDTH,
    window_center: float = _DEFAULT_WINDOW_CENTER,
) -> list[np.ndarray]:
    datasets = sort_slices(read_slices(job_dir))
    return [_process_dataset(ds, window_width, window_center) for ds in datasets]


def parse_series_with_metadata(
    job_dir: Path,
    *,
    window_width: float = _DEFAULT_WINDOW_WIDTH,
    window_center: float = _DEFAULT_WINDOW_CENTER,
) -> tuple[list[np.ndarray], list[SliceMetadata]]:
    """Parse DICOM series returning pixel arrays and per-slice metadata.

    Unlike `parse_series`, this also returns `SliceMetadata` for each slice
    (SOP Instance UID, patient ID, Z position) needed for annotation mapping.
    """
    datasets, raw_meta = read_slices_with_metadata(job_dir)
    datasets, raw_meta = _sort_with_metadata(datasets, raw_meta)

    parsed_slices: list[np.ndarray] = []
    metadata: list[SliceMetadata] = []

    for idx, (dataset, meta) in enumerate(zip(datasets, raw_meta, strict=True)):
        parsed_slices.append(_process_dataset(dataset, window_width, window_center))
        metadata.append(
            SliceMetadata(
                sop_instance_uid=str(meta["sop_uid"]),
                slice_index=idx,
                image_position_z=(
                    float(meta["image_position_z"])
                    if meta["image_position_z"] is not None
                    else None
                ),
                patient_id=str(meta["patient_id"]),
            )
        )

    return parsed_slices, metadata


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
