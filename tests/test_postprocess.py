from __future__ import annotations

import numpy as np

from app.services.inference import PATCH_SIZE, SliceResult
from app.services.postprocess import (
    BBox,
    _nms,
    generate_finding,
    postprocess_results,
    render_overlay,
)

HALF = PATCH_SIZE // 2


def test_generate_finding_matches_template() -> None:
    finding = generate_finding(47, 0.824)
    assert finding == "Suspicious region detected in slice 47 (confidence: 0.82)."


def test_render_overlay_outputs_rgb_with_expected_shape() -> None:
    slice_array = np.full((5, 6), 120, dtype=np.uint8)
    bbox = BBox(x=1, y=1, width=3, height=2)

    overlay = render_overlay(slice_array, bbox)

    assert overlay.shape == (5, 6, 3)
    assert overlay.dtype == np.uint8
    assert np.array_equal(overlay[1, 1], np.array([255, 0, 0], dtype=np.uint8))
    assert np.array_equal(overlay[2, 3], np.array([255, 0, 0], dtype=np.uint8))


def test_render_overlay_clamps_bbox_out_of_bounds() -> None:
    slice_array = np.full((5, 6), 80, dtype=np.uint8)
    bbox = BBox(x=-2, y=-1, width=4, height=4)

    overlay = render_overlay(slice_array, bbox)

    assert overlay.shape == (5, 6, 3)
    assert np.array_equal(overlay[0, 0], np.array([255, 0, 0], dtype=np.uint8))
    assert np.array_equal(overlay[2, 1], np.array([255, 0, 0], dtype=np.uint8))


def test_postprocess_filters_by_threshold_and_top_k() -> None:
    slice_arrays = [
        np.full((64, 64), 50, dtype=np.uint8),
        np.full((64, 64), 70, dtype=np.uint8),
        np.full((64, 64), 90, dtype=np.uint8),
    ]

    inference_results = [
        SliceResult(slice_index=0, score=0.4, x=20, y=20),
        SliceResult(slice_index=1, score=0.95, x=30, y=30),
        SliceResult(slice_index=2, score=0.8, x=25, y=25),
    ]

    findings = postprocess_results(
        slice_arrays,
        inference_results,
        confidence_threshold=0.5,
        top_k=2,
    )

    assert len(findings) == 2
    # Sorted by score descending
    assert findings[0].slice_index == 1
    assert findings[0].confidence == 0.95
    assert findings[0].bbox == BBox(x=30 - HALF, y=30 - HALF, width=PATCH_SIZE, height=PATCH_SIZE)
    assert findings[0].finding == "Suspicious region detected in slice 1 (confidence: 0.95)."
    assert findings[0].image.shape == (64, 64, 3)

    assert findings[1].slice_index == 2
    assert findings[1].confidence == 0.8


def test_postprocess_returns_empty_when_top_k_is_zero() -> None:
    slice_arrays = [np.full((64, 64), 100, dtype=np.uint8)]
    inference_results = [SliceResult(slice_index=0, score=0.9, x=32, y=32)]

    findings = postprocess_results(slice_arrays, inference_results, top_k=0)

    assert findings == []


def test_postprocess_bbox_derived_from_candidate_position() -> None:
    """Verify bbox is centered on the candidate's (x, y) position."""
    slice_arrays = [np.full((100, 100), 128, dtype=np.uint8)]
    inference_results = [SliceResult(slice_index=0, score=0.9, x=50, y=60)]

    findings = postprocess_results(slice_arrays, inference_results)

    assert len(findings) == 1
    bbox = findings[0].bbox
    assert bbox.x == 50 - HALF
    assert bbox.y == 60 - HALF
    assert bbox.width == PATCH_SIZE
    assert bbox.height == PATCH_SIZE


def test_nms_suppresses_nearby_detections() -> None:
    """NMS should keep only the highest-scoring detection in a region."""
    results = [
        SliceResult(slice_index=0, score=0.9, x=100, y=100),
        SliceResult(slice_index=0, score=0.8, x=105, y=105),  # nearby, suppress
        SliceResult(slice_index=0, score=0.7, x=200, y=200),  # far, keep
    ]
    kept = _nms(results, radius=24)

    assert len(kept) == 2
    assert kept[0].score == 0.9
    assert kept[1].score == 0.7


def test_nms_keeps_detections_on_different_slices() -> None:
    """NMS should not suppress detections on different slices."""
    results = [
        SliceResult(slice_index=0, score=0.9, x=100, y=100),
        SliceResult(slice_index=1, score=0.8, x=100, y=100),  # same position, different slice
    ]
    kept = _nms(results, radius=24)

    assert len(kept) == 2


def test_nms_empty_input() -> None:
    assert _nms([], radius=24) == []


def test_postprocess_applies_nms_to_sliding_window_results() -> None:
    """Overlapping sliding window detections should be merged by NMS."""
    slice_arrays = [np.full((128, 128), 100, dtype=np.uint8)]

    # Simulate sliding window: many nearby high-score results within NMS radius
    inference_results = [
        SliceResult(slice_index=0, score=0.95, x=60, y=60),
        SliceResult(slice_index=0, score=0.90, x=63, y=63),
        SliceResult(slice_index=0, score=0.85, x=57, y=57),
        SliceResult(slice_index=0, score=0.80, x=66, y=66),
    ]

    findings = postprocess_results(
        slice_arrays,
        inference_results,
        confidence_threshold=0.5,
    )

    # NMS with radius=HALF should merge nearby detections (all within ~8px of each other)
    assert len(findings) == 1
    assert findings[0].confidence == 0.95
