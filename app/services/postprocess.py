from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import label

from app.services.inference import SliceResult


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


def threshold_cam(cam: np.ndarray, percentile: float = 90.0) -> np.ndarray:
    if cam.ndim != 2:
        raise ValueError("cam must be a 2D array.")
    if not 0.0 <= percentile <= 100.0:
        raise ValueError("percentile must be between 0 and 100.")

    threshold_value = float(np.percentile(cam, percentile))
    return (cam > threshold_value).astype(np.uint8)


def extract_bbox(mask: np.ndarray) -> BBox | None:
    if mask.ndim != 2:
        raise ValueError("mask must be a 2D array.")

    binary_mask = np.asarray(mask, dtype=bool)
    labeled_mask, num_components = label(binary_mask)
    if num_components == 0:
        return None

    component_sizes = np.bincount(labeled_mask.ravel())
    largest_component = int(np.argmax(component_sizes[1:]) + 1)
    ys, xs = np.where(labeled_mask == largest_component)

    x_min = int(xs.min())
    x_max = int(xs.max())
    y_min = int(ys.min())
    y_max = int(ys.max())

    return BBox(
        x=x_min,
        y=y_min,
        width=(x_max - x_min + 1),
        height=(y_max - y_min + 1),
    )


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


def postprocess_results(
    slice_arrays: list[np.ndarray],
    inference_results: list[SliceResult],
    *,
    confidence_threshold: float = 0.5,
    top_k: int = 10,
) -> list[SliceFinding]:
    if top_k <= 0:
        return []

    selected = [result for result in inference_results if result.score >= confidence_threshold]
    selected.sort(key=lambda result: result.score, reverse=True)
    selected = selected[:top_k]

    findings: list[SliceFinding] = []

    for result in selected:
        if result.slice_index < 0 or result.slice_index >= len(slice_arrays):
            raise ValueError("Slice index out of range for provided slice arrays.")

        mask = threshold_cam(result.cam)
        bbox = extract_bbox(mask)
        if bbox is None:
            continue

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
