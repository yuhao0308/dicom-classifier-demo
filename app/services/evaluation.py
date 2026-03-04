"""Evaluate model predictions against ground truth annotations.

Computes IoU between predicted and ground truth bounding boxes and
classifies each prediction/ground-truth pair as true positive, false
positive, or missed.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.services.postprocess import BBox


@dataclass(frozen=True, slots=True)
class SliceEvaluation:
    """Evaluation result for a single slice."""

    slice_index: int
    match_type: str  # "tp", "fp", "missed", "unverified"
    iou: float
    prediction_bbox: BBox | None = None
    prediction_confidence: float | None = None
    gt_bbox: BBox | None = None
    gt_nodule_id: str | None = None
    gt_reader_count: int | None = None


@dataclass(frozen=True, slots=True)
class EvaluationSummary:
    """Top-level evaluation summary across all slices."""

    total_slices: int
    model_flagged: int
    gt_nodule_count: int
    true_positives: int
    false_positives: int
    missed: int
    has_ground_truth: bool
    per_slice: list[SliceEvaluation]


def compute_iou(bbox_a: BBox, bbox_b: BBox) -> float:
    """Compute intersection-over-union between two axis-aligned bounding boxes.

    Returns a value in [0.0, 1.0].
    """
    x_left = max(bbox_a.x, bbox_b.x)
    y_top = max(bbox_a.y, bbox_b.y)
    x_right = min(bbox_a.x + bbox_a.width, bbox_b.x + bbox_b.width)
    y_bottom = min(bbox_a.y + bbox_a.height, bbox_b.y + bbox_b.height)

    if x_right <= x_left or y_bottom <= y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    area_a = bbox_a.width * bbox_a.height
    area_b = bbox_b.width * bbox_b.height
    union_area = area_a + area_b - intersection_area

    if union_area <= 0:
        return 0.0

    return intersection_area / union_area


def evaluate_results(
    predictions: list[dict[str, object]],
    gt_by_slice_index: dict[int, list[dict[str, object]]],
    total_slices: int,
    *,
    iou_threshold: float = 0.1,
) -> EvaluationSummary:
    """Compare model predictions against ground truth annotations.

    Parameters
    ----------
    predictions:
        List of abnormal slice dicts from the postprocessing pipeline, each
        having ``slice_index``, ``confidence``, ``bbox`` keys.
    gt_by_slice_index:
        Dict mapping slice index → list of ground truth annotation dicts,
        each with ``bbox``, ``nodule_id``, ``reader_count`` keys.
    total_slices:
        Total number of slices in the series.
    iou_threshold:
        Minimum IoU to consider a prediction–GT pair a true positive.

    Returns
    -------
    EvaluationSummary
    """
    has_gt = bool(gt_by_slice_index)
    per_slice: list[SliceEvaluation] = []
    true_positives = 0
    false_positives = 0

    # Track which GT nodule-slice entries have been matched
    matched_gt_keys: set[tuple[int, str]] = set()

    for pred in predictions:
        slice_idx = int(pred["slice_index"])  # type: ignore[arg-type]
        pred_bbox_dict = pred["bbox"]
        pred_bbox = BBox(
            x=int(pred_bbox_dict["x"]),  # type: ignore[index]
            y=int(pred_bbox_dict["y"]),  # type: ignore[index]
            width=int(pred_bbox_dict["width"]),  # type: ignore[index]
            height=int(pred_bbox_dict["height"]),  # type: ignore[index]
        )
        pred_conf = float(pred["confidence"])  # type: ignore[arg-type]

        if not has_gt:
            per_slice.append(
                SliceEvaluation(
                    slice_index=slice_idx,
                    match_type="unverified",
                    iou=0.0,
                    prediction_bbox=pred_bbox,
                    prediction_confidence=pred_conf,
                )
            )
            continue

        gt_list = gt_by_slice_index.get(slice_idx, [])
        best_iou = 0.0
        best_gt: dict[str, object] | None = None
        best_gt_bbox: BBox | None = None

        for gt in gt_list:
            gt_bbox_dict = gt["bbox"]
            gt_bbox = BBox(
                x=int(gt_bbox_dict["x"]),  # type: ignore[index]
                y=int(gt_bbox_dict["y"]),  # type: ignore[index]
                width=int(gt_bbox_dict["width"]),  # type: ignore[index]
                height=int(gt_bbox_dict["height"]),  # type: ignore[index]
            )
            iou = compute_iou(pred_bbox, gt_bbox)
            if iou > best_iou:
                best_iou = iou
                best_gt = gt
                best_gt_bbox = gt_bbox

        if best_iou >= iou_threshold and best_gt is not None:
            true_positives += 1
            gt_nid = str(best_gt.get("nodule_id", ""))
            matched_gt_keys.add((slice_idx, gt_nid))
            per_slice.append(
                SliceEvaluation(
                    slice_index=slice_idx,
                    match_type="tp",
                    iou=round(best_iou, 4),
                    prediction_bbox=pred_bbox,
                    prediction_confidence=pred_conf,
                    gt_bbox=best_gt_bbox,
                    gt_nodule_id=gt_nid,
                    gt_reader_count=int(best_gt.get("reader_count", 0)),  # type: ignore[arg-type]
                )
            )
        else:
            false_positives += 1
            per_slice.append(
                SliceEvaluation(
                    slice_index=slice_idx,
                    match_type="fp",
                    iou=round(best_iou, 4),
                    prediction_bbox=pred_bbox,
                    prediction_confidence=pred_conf,
                    gt_bbox=best_gt_bbox,
                    gt_nodule_id=str(best_gt.get("nodule_id", "")) if best_gt else None,
                    gt_reader_count=(
                        int(best_gt.get("reader_count", 0))  # type: ignore[arg-type]
                        if best_gt
                        else None
                    ),
                )
            )

    # Collect all unique GT nodule IDs and which were matched
    all_gt_ids: set[str] = set()
    matched_gt_nodule_ids: set[str] = {nid for _, nid in matched_gt_keys}
    for gt_list in gt_by_slice_index.values():
        for gt in gt_list:
            all_gt_ids.add(str(gt.get("nodule_id", "")))

    # Find missed GT entries (present in GT but not matched by any prediction).
    # We still emit per-slice entries for the detail view, but count missed as
    # unique nodule IDs to stay consistent with gt_nodule_count.
    missed_nodule_ids: set[str] = all_gt_ids - matched_gt_nodule_ids
    for slice_idx, gt_list in gt_by_slice_index.items():
        for gt in gt_list:
            gt_nid = str(gt.get("nodule_id", ""))
            key = (slice_idx, gt_nid)
            if key not in matched_gt_keys:
                gt_bbox_dict = gt["bbox"]
                per_slice.append(
                    SliceEvaluation(
                        slice_index=slice_idx,
                        match_type="missed",
                        iou=0.0,
                        gt_bbox=BBox(
                            x=int(gt_bbox_dict["x"]),  # type: ignore[index]
                            y=int(gt_bbox_dict["y"]),  # type: ignore[index]
                            width=int(gt_bbox_dict["width"]),  # type: ignore[index]
                            height=int(gt_bbox_dict["height"]),  # type: ignore[index]
                        ),
                        gt_nodule_id=gt_nid,
                        gt_reader_count=int(gt.get("reader_count", 0)),  # type: ignore[arg-type]
                    )
                )

    per_slice.sort(key=lambda s: s.slice_index)

    return EvaluationSummary(
        total_slices=total_slices,
        model_flagged=len(predictions),
        gt_nodule_count=len(all_gt_ids),
        true_positives=len(matched_gt_nodule_ids),
        false_positives=false_positives,
        missed=len(missed_nodule_ids),
        has_ground_truth=has_gt,
        per_slice=per_slice,
    )
