from __future__ import annotations

import numpy as np

from app.services.inference import SliceResult
from app.services.postprocess import (
    BBox,
    extract_bbox,
    generate_finding,
    postprocess_results,
    render_overlay,
    threshold_cam,
)


def test_threshold_cam_returns_expected_binary_mask() -> None:
    cam = np.array([[0.1, 0.4], [0.6, 0.9]], dtype=np.float32)

    mask = threshold_cam(cam, percentile=50.0)

    assert np.array_equal(mask, np.array([[0, 0], [1, 1]], dtype=np.uint8))


def test_extract_bbox_returns_largest_component_bbox() -> None:
    mask = np.zeros((6, 7), dtype=np.uint8)
    mask[1, 1] = 1
    mask[2:5, 3:6] = 1

    bbox = extract_bbox(mask)

    assert bbox == BBox(x=3, y=2, width=3, height=3)


def test_extract_bbox_returns_none_for_empty_mask() -> None:
    mask = np.zeros((4, 4), dtype=np.uint8)
    assert extract_bbox(mask) is None


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
        np.full((8, 8), 50, dtype=np.uint8),
        np.full((8, 8), 70, dtype=np.uint8),
        np.full((8, 8), 90, dtype=np.uint8),
    ]
    hotspot_cam = np.zeros((8, 8), dtype=np.float32)
    hotspot_cam[2:4, 3:5] = 1.0

    inference_results = [
        SliceResult(slice_index=0, score=0.4, cam=hotspot_cam),
        SliceResult(slice_index=1, score=0.95, cam=hotspot_cam),
        SliceResult(slice_index=2, score=0.8, cam=hotspot_cam),
        SliceResult(slice_index=0, score=0.9, cam=np.zeros((8, 8), dtype=np.float32)),
    ]

    findings = postprocess_results(
        slice_arrays,
        inference_results,
        confidence_threshold=0.5,
        top_k=2,
    )

    assert len(findings) == 1
    assert findings[0].slice_index == 1
    assert findings[0].confidence == 0.95
    assert findings[0].bbox == BBox(x=3, y=2, width=2, height=2)
    assert findings[0].finding == "Suspicious region detected in slice 1 (confidence: 0.95)."
    assert findings[0].image.shape == (8, 8, 3)


def test_postprocess_returns_empty_when_top_k_is_zero() -> None:
    slice_arrays = [np.full((8, 8), 100, dtype=np.uint8)]
    hotspot_cam = np.zeros((8, 8), dtype=np.float32)
    hotspot_cam[2:4, 2:4] = 1.0
    inference_results = [SliceResult(slice_index=0, score=0.9, cam=hotspot_cam)]

    findings = postprocess_results(slice_arrays, inference_results, top_k=0)

    assert findings == []
