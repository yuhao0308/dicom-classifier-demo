from __future__ import annotations

from app.services.evaluation import EvaluationSummary, compute_iou, evaluate_results
from app.services.postprocess import BBox


def test_compute_iou_identical_boxes() -> None:
    bbox = BBox(x=10, y=10, width=20, height=20)
    assert compute_iou(bbox, bbox) == 1.0


def test_compute_iou_no_overlap() -> None:
    a = BBox(x=0, y=0, width=10, height=10)
    b = BBox(x=20, y=20, width=10, height=10)
    assert compute_iou(a, b) == 0.0


def test_compute_iou_partial_overlap() -> None:
    a = BBox(x=0, y=0, width=10, height=10)
    b = BBox(x=5, y=5, width=10, height=10)
    iou = compute_iou(a, b)
    # Intersection: 5x5=25, Union: 100+100-25=175
    assert abs(iou - 25 / 175) < 1e-6


def test_compute_iou_one_pixel_box() -> None:
    a = BBox(x=5, y=5, width=1, height=1)
    b = BBox(x=5, y=5, width=1, height=1)
    assert compute_iou(a, b) == 1.0


def test_evaluate_results_with_true_positive() -> None:
    predictions = [
        {
            "slice_index": 5,
            "confidence": 0.9,
            "bbox": {"x": 10, "y": 10, "width": 20, "height": 20},
        }
    ]
    gt = {
        5: [
            {
                "bbox": {"x": 10, "y": 10, "width": 20, "height": 20},
                "nodule_id": "N1",
                "reader_count": 3,
            }
        ]
    }

    result = evaluate_results(predictions, gt, total_slices=50)

    assert isinstance(result, EvaluationSummary)
    assert result.true_positives == 1
    assert result.false_positives == 0
    assert result.missed == 0
    assert result.has_ground_truth is True


def test_evaluate_results_with_false_positive() -> None:
    predictions = [
        {
            "slice_index": 5,
            "confidence": 0.8,
            "bbox": {"x": 100, "y": 100, "width": 20, "height": 20},
        }
    ]
    gt = {
        10: [
            {
                "bbox": {"x": 200, "y": 200, "width": 30, "height": 30},
                "nodule_id": "N1",
                "reader_count": 2,
            }
        ]
    }

    result = evaluate_results(predictions, gt, total_slices=50)

    assert result.false_positives == 1
    assert result.true_positives == 0
    assert result.missed == 1


def test_evaluate_results_with_missed_nodule() -> None:
    predictions: list[dict[str, object]] = []
    gt = {
        7: [
            {
                "bbox": {"x": 50, "y": 50, "width": 15, "height": 15},
                "nodule_id": "N1",
                "reader_count": 4,
            }
        ]
    }

    result = evaluate_results(predictions, gt, total_slices=50)

    assert result.missed == 1
    assert result.true_positives == 0
    assert result.false_positives == 0
    assert result.model_flagged == 0


def test_evaluate_results_no_ground_truth() -> None:
    predictions = [
        {
            "slice_index": 3,
            "confidence": 0.7,
            "bbox": {"x": 10, "y": 10, "width": 10, "height": 10},
        }
    ]

    result = evaluate_results(predictions, {}, total_slices=50)

    assert result.has_ground_truth is False
    assert len(result.per_slice) == 1
    assert result.per_slice[0].match_type == "unverified"


def test_evaluate_results_missed_counts_unique_nodules() -> None:
    """A nodule spanning multiple slices should count as 1 missed, not N."""
    predictions: list[dict[str, object]] = []
    gt = {
        5: [
            {
                "bbox": {"x": 50, "y": 50, "width": 15, "height": 15},
                "nodule_id": "N1",
                "reader_count": 3,
            }
        ],
        6: [
            {
                "bbox": {"x": 50, "y": 50, "width": 16, "height": 16},
                "nodule_id": "N1",
                "reader_count": 3,
            }
        ],
        7: [
            {
                "bbox": {"x": 50, "y": 50, "width": 14, "height": 14},
                "nodule_id": "N1",
                "reader_count": 3,
            }
        ],
    }

    result = evaluate_results(predictions, gt, total_slices=50)

    assert result.gt_nodule_count == 1
    assert result.missed == 1
    # Per-slice detail still has 3 entries for the detail view
    missed_entries = [s for s in result.per_slice if s.match_type == "missed"]
    assert len(missed_entries) == 3


def test_evaluate_results_mixed_tp_fp_missed() -> None:
    predictions = [
        {
            "slice_index": 5,
            "confidence": 0.9,
            "bbox": {"x": 10, "y": 10, "width": 20, "height": 20},
        },
        {
            "slice_index": 8,
            "confidence": 0.6,
            "bbox": {"x": 200, "y": 200, "width": 10, "height": 10},
        },
    ]
    gt = {
        5: [
            {
                "bbox": {"x": 10, "y": 10, "width": 20, "height": 20},
                "nodule_id": "N1",
                "reader_count": 3,
            }
        ],
        12: [
            {
                "bbox": {"x": 300, "y": 300, "width": 25, "height": 25},
                "nodule_id": "N2",
                "reader_count": 2,
            }
        ],
    }

    result = evaluate_results(predictions, gt, total_slices=50)

    assert result.true_positives == 1
    assert result.false_positives == 1
    assert result.missed == 1
    assert result.model_flagged == 2
    assert result.gt_nodule_count == 2
