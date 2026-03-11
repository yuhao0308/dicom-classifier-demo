from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.services.inference import PATCH_SIZE, SliceResult


@dataclass(frozen=True, slots=True)
class BBox:
    x: int
    y: int
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class SliceFinding:
    slice_index: int
    confidence: float
    finding: str
    bbox: BBox
    image: np.ndarray


def generate_finding(slice_index: int, confidence: float) -> str:
    return f"Suspicious region detected in slice {slice_index} (confidence: {confidence:.2f})."


def _draw_bbox(rgb: np.ndarray, bbox: BBox, color: tuple[int, int, int]) -> None:
    """Draw an axis-aligned bounding box on an RGB image (in-place)."""
    x_min = max(0, bbox.x)
    y_min = max(0, bbox.y)
    x_max = min(rgb.shape[1] - 1, bbox.x + bbox.width - 1)
    y_max = min(rgb.shape[0] - 1, bbox.y + bbox.height - 1)

    if x_min > x_max or y_min > y_max:
        return

    c = np.array(color, dtype=np.uint8)
    rgb[y_min, x_min : x_max + 1] = c
    rgb[y_max, x_min : x_max + 1] = c
    rgb[y_min : y_max + 1, x_min] = c
    rgb[y_min : y_max + 1, x_max] = c


def _to_rgb(slice_array: np.ndarray) -> np.ndarray:
    """Convert a 2D grayscale array to an RGB uint8 image."""
    if slice_array.ndim != 2:
        raise ValueError("slice_array must be a 2D array.")
    grayscale = np.asarray(np.clip(slice_array, 0, 255), dtype=np.uint8)
    return np.stack((grayscale, grayscale, grayscale), axis=-1)


_COLOR_RED = (255, 0, 0)
_COLOR_GREEN = (0, 200, 0)


def render_overlay(slice_array: np.ndarray, bbox: BBox) -> np.ndarray:
    rgb = _to_rgb(slice_array)
    _draw_bbox(rgb, bbox, _COLOR_RED)
    return rgb


def render_ground_truth_overlay(slice_array: np.ndarray, gt_bbox: BBox) -> np.ndarray:
    """Render a CT slice with a green bounding box for ground truth."""
    rgb = _to_rgb(slice_array)
    _draw_bbox(rgb, gt_bbox, _COLOR_GREEN)
    return rgb


def render_comparison_overlay(
    slice_array: np.ndarray,
    pred_bbox: BBox | None,
    gt_bbox: BBox | None,
) -> np.ndarray:
    """Render a CT slice with both prediction (red) and ground truth (green) boxes."""
    rgb = _to_rgb(slice_array)
    if gt_bbox is not None:
        _draw_bbox(rgb, gt_bbox, _COLOR_GREEN)
    if pred_bbox is not None:
        _draw_bbox(rgb, pred_bbox, _COLOR_RED)
    return rgb


def _bbox_from_candidate(x: int, y: int, patch_size: int = PATCH_SIZE) -> BBox:
    """Create a bounding box centered on the candidate position."""
    half = patch_size // 2
    return BBox(x=x - half, y=y - half, width=patch_size, height=patch_size)


def _nms(
    results: list[SliceResult],
    radius: int = PATCH_SIZE // 2,
) -> list[SliceResult]:
    """Non-maximum suppression: keep only the highest-scoring detection per region.

    For each result (sorted by score descending), suppress any lower-scoring
    result on the same slice whose center is within `radius` pixels.
    """
    kept: list[SliceResult] = []
    for result in results:
        suppressed = False
        for kept_result in kept:
            if kept_result.slice_index != result.slice_index:
                continue
            dx = result.x - kept_result.x
            dy = result.y - kept_result.y
            if dx * dx + dy * dy <= radius * radius:
                suppressed = True
                break
        if not suppressed:
            kept.append(result)
    return kept


def postprocess_results(
    slice_arrays: list[np.ndarray],
    inference_results: list[SliceResult],
    *,
    confidence_threshold: float = 0.15,
    top_k: int = 10,
) -> list[SliceFinding]:
    """Convert inference results to findings with bounding boxes and overlays.

    Each SliceResult has a candidate position (x, y) — the bbox is derived
    directly from the patch coordinates centered on that position.

    Applies non-maximum suppression to merge overlapping detections from
    the sliding window candidate generator.
    """
    if top_k <= 0:
        return []

    selected = [result for result in inference_results if result.score >= confidence_threshold]
    selected.sort(key=lambda result: result.score, reverse=True)
    selected = _nms(selected)
    selected = selected[:top_k]

    findings: list[SliceFinding] = []

    for result in selected:
        if result.slice_index < 0 or result.slice_index >= len(slice_arrays):
            raise ValueError("Slice index out of range for provided slice arrays.")

        bbox = _bbox_from_candidate(result.x, result.y)
        finding_text = generate_finding(result.slice_index, result.score)
        overlay = render_overlay(slice_arrays[result.slice_index], bbox)
        findings.append(
            SliceFinding(
                slice_index=result.slice_index,
                confidence=result.score,
                finding=finding_text,
                bbox=bbox,
                image=overlay,
            )
        )

    return findings
