"""Parse LIDC-IDRI XML annotation files into structured nodule data.

The LIDC XML format contains multiple ``readingSession`` elements (one per
radiologist, typically 4).  Each session contains ``unblindedReadNodule``
elements (nodules ≥3 mm) and ``nonNodule`` elements.

This module extracts per-slice bounding boxes from the contour edge-map
coordinates and groups nodules across reading sessions by spatial proximity
to produce a consensus view.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from xml.etree import ElementTree

from app.services.postprocess import BBox

LOGGER = logging.getLogger(__name__)

# LIDC XML namespace
_NS = {"lidc": "http://www.nih.gov"}

# Default distance threshold (in pixels) for matching nodules across readers.
_CENTROID_MATCH_THRESHOLD_PX = 40.0


@dataclass(frozen=True, slots=True)
class NoduleSlice:
    """A nodule annotation on a single CT slice."""

    sop_uid: str
    z_position: float
    contour_points: list[tuple[int, int]]
    bbox: BBox


@dataclass(slots=True)
class NoduleAnnotation:
    """A distinct nodule, potentially annotated by multiple radiologists."""

    nodule_id: str
    reading_session_count: int = 1
    slices: dict[str, NoduleSlice] = field(default_factory=dict)

    @property
    def centroid_x(self) -> float:
        """Mean X across all contour points on all slices."""
        all_x = [p[0] for s in self.slices.values() for p in s.contour_points]
        return sum(all_x) / len(all_x) if all_x else 0.0

    @property
    def centroid_y(self) -> float:
        """Mean Y across all contour points on all slices."""
        all_y = [p[1] for s in self.slices.values() for p in s.contour_points]
        return sum(all_y) / len(all_y) if all_y else 0.0

    @property
    def overall_bbox(self) -> BBox | None:
        """Axis-aligned bounding box spanning all slices."""
        if not self.slices:
            return None
        all_x = [p[0] for s in self.slices.values() for p in s.contour_points]
        all_y = [p[1] for s in self.slices.values() for p in s.contour_points]
        x_min, x_max = min(all_x), max(all_x)
        y_min, y_max = min(all_y), max(all_y)
        return BBox(
            x=x_min,
            y=y_min,
            width=max(x_max - x_min + 1, 1),
            height=max(y_max - y_min + 1, 1),
        )


def _contour_to_bbox(points: list[tuple[int, int]]) -> BBox:
    """Compute axis-aligned bounding box from contour edge-map points."""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    return BBox(
        x=x_min,
        y=y_min,
        width=max(x_max - x_min + 1, 1),
        height=max(y_max - y_min + 1, 1),
    )


def _parse_roi(roi_elem: ElementTree.Element) -> NoduleSlice | None:
    """Parse a single ``<roi>`` element into a NoduleSlice."""
    z_elem = roi_elem.find("lidc:imageZposition", _NS)
    sop_elem = roi_elem.find("lidc:imageSOP_UID", _NS)
    if z_elem is None or z_elem.text is None:
        return None
    if sop_elem is None or sop_elem.text is None:
        return None

    z_position = float(z_elem.text.strip())
    sop_uid = sop_elem.text.strip()

    contour_points: list[tuple[int, int]] = []
    for edge_map in roi_elem.findall("lidc:edgeMap", _NS):
        x_elem = edge_map.find("lidc:xCoord", _NS)
        y_elem = edge_map.find("lidc:yCoord", _NS)
        if x_elem is not None and y_elem is not None:
            x_text = x_elem.text
            y_text = y_elem.text
            if x_text is not None and y_text is not None:
                contour_points.append((int(x_text.strip()), int(y_text.strip())))

    if not contour_points:
        return None

    return NoduleSlice(
        sop_uid=sop_uid,
        z_position=z_position,
        contour_points=contour_points,
        bbox=_contour_to_bbox(contour_points),
    )


def _parse_nodule(
    nodule_elem: ElementTree.Element,
) -> tuple[str, list[NoduleSlice]]:
    """Parse a single ``<unblindedReadNodule>`` element."""
    id_elem = nodule_elem.find("lidc:noduleID", _NS)
    nodule_id = id_elem.text.strip() if id_elem is not None and id_elem.text else "Unknown"

    slices: list[NoduleSlice] = []
    for roi_elem in nodule_elem.findall("lidc:roi", _NS):
        nodule_slice = _parse_roi(roi_elem)
        if nodule_slice is not None:
            slices.append(nodule_slice)

    return nodule_id, slices


def _nodule_centroid(slices: list[NoduleSlice]) -> tuple[float, float]:
    """Compute mean centroid of a nodule across its slices."""
    all_x = [p[0] for s in slices for p in s.contour_points]
    all_y = [p[1] for s in slices for p in s.contour_points]
    return (
        sum(all_x) / len(all_x) if all_x else 0.0,
        sum(all_y) / len(all_y) if all_y else 0.0,
    )


def _euclidean_dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def _merge_nodules(
    raw_nodules: list[tuple[str, list[NoduleSlice]]],
    match_threshold: float = _CENTROID_MATCH_THRESHOLD_PX,
) -> list[NoduleAnnotation]:
    """Merge nodules from multiple reading sessions by spatial proximity.

    If two nodules from different readers have centroids within
    *match_threshold* pixels, they are considered the same physical nodule
    and merged into a single `NoduleAnnotation` with an incremented reader
    count.
    """
    merged: list[NoduleAnnotation] = []

    for nodule_id, slices in raw_nodules:
        if not slices:
            continue

        centroid = _nodule_centroid(slices)
        matched = False

        for existing in merged:
            existing_centroid = (existing.centroid_x, existing.centroid_y)
            if _euclidean_dist(centroid, existing_centroid) <= match_threshold:
                existing.reading_session_count += 1
                for s in slices:
                    if s.sop_uid not in existing.slices:
                        existing.slices[s.sop_uid] = s
                matched = True
                break

        if not matched:
            annotation = NoduleAnnotation(nodule_id=nodule_id)
            for s in slices:
                annotation.slices[s.sop_uid] = s
            merged.append(annotation)

    return merged


def parse_annotation_xml(xml_path: Path) -> list[NoduleAnnotation]:
    """Parse an LIDC-IDRI XML annotation file and return merged nodules.

    Parameters
    ----------
    xml_path:
        Path to the XML annotation file (e.g. ``data/LIDC-IDRI-0078/annotations.xml``).

    Returns
    -------
    list[NoduleAnnotation]
        Merged nodules with per-slice bounding boxes and reader counts.
    """
    if not xml_path.exists():
        LOGGER.warning("annotation_file_not_found", extra={"path": str(xml_path)})
        return []

    try:
        tree = ElementTree.parse(xml_path)  # noqa: S314
    except ElementTree.ParseError as exc:
        LOGGER.warning(
            "annotation_xml_parse_error",
            extra={"path": str(xml_path), "error": str(exc)},
        )
        return []

    root = tree.getroot()
    raw_nodules: list[tuple[str, list[NoduleSlice]]] = []

    for session in root.findall("lidc:readingSession", _NS):
        for nodule_elem in session.findall("lidc:unblindedReadNodule", _NS):
            nodule_id, slices = _parse_nodule(nodule_elem)
            if slices:
                raw_nodules.append((nodule_id, slices))

    nodules = _merge_nodules(raw_nodules)
    LOGGER.info(
        "annotations_parsed",
        extra={"path": str(xml_path), "nodule_count": len(nodules)},
    )
    return nodules


def find_annotation_for_patient(data_dir: Path, patient_id: str) -> Path | None:
    """Locate the LIDC XML annotation file for a patient in the data directory.

    Checks for:
    1. ``data/<patient_id>/annotations.xml`` (downloaded by our script)
    2. ``data/annotations/<NNN>.xml`` (extracted from full zip)
    """
    candidate_1 = data_dir / patient_id / "annotations.xml"
    if candidate_1.exists():
        return candidate_1

    import re

    match = re.search(r"(\d+)$", patient_id)
    if match:
        numeric_name = f"{int(match.group(1)):03d}.xml"
        candidate_2 = data_dir / "annotations" / numeric_name
        if candidate_2.exists():
            return candidate_2

    return None


def get_ground_truth_for_slice(
    nodules: list[NoduleAnnotation],
    sop_uid: str,
) -> list[tuple[NoduleAnnotation, NoduleSlice]]:
    """Find all nodule annotations that appear on a given slice (by SOP UID).

    Returns a list of (nodule, slice_annotation) tuples for the given SOP UID.
    """
    matches: list[tuple[NoduleAnnotation, NoduleSlice]] = []
    for nodule in nodules:
        if sop_uid in nodule.slices:
            matches.append((nodule, nodule.slices[sop_uid]))
    return matches


def build_z_position_index(
    nodules: list[NoduleAnnotation],
) -> dict[float, list[tuple[NoduleAnnotation, NoduleSlice]]]:
    """Build an index mapping Z position → list of (nodule, slice) tuples.

    Used as a fallback when SOP UIDs don't match (e.g. different series of
    the same patient).
    """
    index: dict[float, list[tuple[NoduleAnnotation, NoduleSlice]]] = {}
    for nodule in nodules:
        for nodule_slice in nodule.slices.values():
            z = nodule_slice.z_position
            index.setdefault(z, []).append((nodule, nodule_slice))
    return index


def get_ground_truth_for_z_position(
    z_index: dict[float, list[tuple[NoduleAnnotation, NoduleSlice]]],
    z_position: float,
    tolerance: float = 0.5,
) -> list[tuple[NoduleAnnotation, NoduleSlice]]:
    """Find annotations matching a Z position within a tolerance (mm).

    Fallback for when SOP UIDs don't match across series.
    """
    matches: list[tuple[NoduleAnnotation, NoduleSlice]] = []
    for z, entries in z_index.items():
        if abs(z - z_position) <= tolerance:
            matches.extend(entries)
    return matches
