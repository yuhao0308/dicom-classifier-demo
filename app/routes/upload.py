from __future__ import annotations

import io
import logging
import shutil
import zipfile
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, Request, UploadFile, status
from fastapi.responses import HTMLResponse
from PIL import Image

from app.config import get_settings
from app.services.annotation_parser import (
    build_z_position_index,
    find_annotation_for_patient,
    get_ground_truth_for_slice,
    get_ground_truth_for_z_position,
    parse_annotation_xml,
)
from app.services.dicom_parser import parse_series_with_metadata
from app.services.evaluation import evaluate_results
from app.services.inference import InferenceModel, load_model, run_inference
from app.services.postprocess import (
    BBox,
    postprocess_results,
    render_comparison_overlay,
    render_ground_truth_overlay,
)
from app.services.storage import (
    create_job_dir,
    save_dicom_slice,
    update_job_metadata,
    write_job_findings,
    write_job_metadata,
)

router = APIRouter()
LOGGER = logging.getLogger(__name__)

_DICOM_MAGIC_START = 128
_DICOM_MAGIC_END = 132
_DICOM_MAGIC = b"DICM"
_MEDIA_URL_PREFIX = "/job-media"
_DISCLAIMER = "For reference only. Not medical advice. Clinician must use their judgment."


def _has_dicom_magic(payload: bytes) -> bool:
    return (
        len(payload) >= _DICOM_MAGIC_END
        and payload[_DICOM_MAGIC_START:_DICOM_MAGIC_END] == _DICOM_MAGIC
    )


@router.get("/", response_class=HTMLResponse)
async def upload_page(request: Request) -> HTMLResponse:
    return request.app.state.templates.TemplateResponse(
        "index.html",
        {"request": request, "disclaimer": _DISCLAIMER},
    )


@router.post("/upload", status_code=status.HTTP_202_ACCEPTED)
async def upload(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile | None = File(default=None),
    annotation: UploadFile | None = File(default=None),
) -> dict[str, str | int]:
    if file is None:
        raise HTTPException(status_code=400, detail="No file was uploaded.")

    settings = (
        request.app.state.settings if hasattr(request.app.state, "settings") else get_settings()
    )
    archive_payload = await file.read()

    if not archive_payload:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    if len(archive_payload) > settings.max_upload_size_bytes:
        raise HTTPException(status_code=413, detail="Payload too large.")

    if not zipfile.is_zipfile(io.BytesIO(archive_payload)):
        raise HTTPException(status_code=400, detail="Only .zip archives are supported.")

    job_id = str(uuid4())
    job_dir = create_job_dir(settings, job_id)
    slice_count = 0

    try:
        with zipfile.ZipFile(io.BytesIO(archive_payload)) as archive:
            for entry in archive.infolist():
                if entry.is_dir():
                    continue

                with archive.open(entry) as content_file:
                    payload = content_file.read()

                if not _has_dicom_magic(payload):
                    raise HTTPException(
                        status_code=400,
                        detail="Archive contains non-DICOM content.",
                    )

                slice_count += 1
                if slice_count > settings.max_slices:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Archive exceeds MAX_SLICES ({settings.max_slices}).",
                    )
                save_dicom_slice(job_dir, index=slice_count, payload=payload)
    except zipfile.BadZipFile as exc:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail="Only .zip archives are supported.") from exc
    except HTTPException:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise
    except OSError as exc:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail="Failed to persist upload.") from exc

    if slice_count == 0:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail="Archive has no DICOM files.")

    write_job_metadata(
        job_dir,
        job_id=job_id,
        status="processing",
        slice_count=slice_count,
        progress=0,
    )
    model = getattr(
        request.app.state,
        "model",
        load_model(settings.model_path, use_gpu=settings.use_gpu),
    )
    batch_size = int(
        getattr(
            request.app.state,
            "inference_batch_size",
            settings.inference_batch_size,
        )
    )
    annotation_dir = Path(getattr(request.app.state, "annotation_dir", settings.annotation_dir))

    # Save user-uploaded annotation XML if provided
    annotation_xml_path: Path | None = None
    if annotation is not None:
        annotation_payload = await annotation.read()
        if annotation_payload:
            annotation_xml_path = job_dir / "annotations.xml"
            annotation_xml_path.write_bytes(annotation_payload)

    background_tasks.add_task(
        _run_processing_pipeline, job_dir, model, batch_size, annotation_dir, annotation_xml_path
    )

    return {
        "job_id": job_id,
        "status": "processing",
        "slice_count": slice_count,
        "message": "Upload accepted. Processing started.",
    }


def _lookup_gt_for_slice(
    nodules: list,
    sop_uid: str,
    z_position: float | None,
    z_index: dict | None,
    use_z_fallback: bool,
) -> list:
    """Look up ground truth for a slice (SOP UID first, then Z fallback)."""
    matches = get_ground_truth_for_slice(nodules, sop_uid) if sop_uid else []
    if not matches and use_z_fallback and z_index and z_position is not None:
        matches = get_ground_truth_for_z_position(z_index, z_position)
    return matches


def _run_processing_pipeline(
    job_dir: Path,
    model: InferenceModel,
    batch_size: int,
    annotation_dir: Path,
    annotation_xml_path: Path | None = None,
) -> None:
    try:
        update_job_metadata(job_dir, status="processing", progress=25)

        # Parse DICOM series WITH metadata (SOP UIDs, patient ID)
        slices, slice_metadata = parse_series_with_metadata(job_dir)
        update_job_metadata(job_dir, progress=40, slice_count=len(slices))

        # Attempt to load ground truth annotations
        # Priority: user-uploaded XML > auto-discovered XML
        patient_id = slice_metadata[0].patient_id if slice_metadata else ""
        nodules: list = []
        annotation_source = "none"

        if annotation_xml_path is not None and annotation_xml_path.exists():
            nodules = parse_annotation_xml(annotation_xml_path)
            annotation_source = "user_upload"
            LOGGER.info(
                "annotations_loaded_from_upload",
                extra={"nodule_count": len(nodules)},
            )
        elif patient_id:
            xml_path = find_annotation_for_patient(annotation_dir, patient_id)
            if xml_path is not None:
                nodules = parse_annotation_xml(xml_path)
                annotation_source = f"auto:{xml_path.name}"
                LOGGER.info(
                    "annotations_loaded",
                    extra={
                        "patient_id": patient_id,
                        "nodule_count": len(nodules),
                        "source": annotation_source,
                    },
                )

        update_job_metadata(job_dir, progress=55)

        # Run model inference
        inference_results = run_inference(model, slices, batch_size=batch_size)

        # Log score distribution for diagnostics
        if inference_results:
            scores = [r.score for r in inference_results]
            scores_sorted = sorted(scores, reverse=True)
            LOGGER.info(
                "inference_score_distribution",
                extra={
                    "total_slices": len(scores),
                    "min": round(min(scores), 4),
                    "max": round(max(scores), 4),
                    "mean": round(sum(scores) / len(scores), 4),
                    "top_10_scores": [round(s, 4) for s in scores_sorted[:10]],
                    "above_0.5": sum(1 for s in scores if s >= 0.5),
                    "above_0.3": sum(1 for s in scores if s >= 0.3),
                    "above_0.2": sum(1 for s in scores if s >= 0.2),
                },
            )

        findings = postprocess_results(slices, inference_results)

        # ── Build GT mapping (SOP UID first, Z-position fallback) ──
        sop_to_index: dict[str, int] = {m.sop_instance_uid: m.slice_index for m in slice_metadata}

        # Detect if we need Z-position fallback
        use_z_fallback = False
        z_index = None
        if nodules:
            sop_matched = sum(1 for n in nodules for sop in n.slices if sop in sop_to_index)
            if sop_matched == 0:
                LOGGER.info(
                    "sop_uid_mismatch_using_z_fallback",
                    extra={"patient_id": patient_id},
                )
                use_z_fallback = True
                z_index = build_z_position_index(nodules)

        # Build Z-rank mapping for fallback (handles different coordinate frames)
        ann_z_to_dicom_idx: dict[float, int] = {}
        if use_z_fallback:
            # Sort DICOM Z positions
            dicom_zs = sorted(
                (
                    (m.image_position_z, m.slice_index)
                    for m in slice_metadata
                    if m.image_position_z is not None
                ),
                key=lambda x: x[0],
            )
            # Collect unique annotation Z positions
            ann_zs = sorted({ns.z_position for n in nodules for ns in n.slices.values()})
            if dicom_zs and ann_zs:
                # Compute spacing in both coordinate frames
                dicom_z_values = [z for z, _ in dicom_zs]
                dicom_min = dicom_z_values[0]
                dicom_max = dicom_z_values[-1]
                ann_min = ann_zs[0]
                ann_max = ann_zs[-1]
                dicom_span = dicom_max - dicom_min
                ann_span = ann_max - ann_min
                if ann_span > 0 and dicom_span > 0:
                    # Map annotation Z to DICOM slice by linear
                    # interpolation of relative position
                    for az in ann_zs:
                        frac = (az - ann_min) / ann_span
                        target_z = dicom_min + frac * dicom_span
                        # Find closest DICOM slice to target_z
                        best_idx = dicom_zs[0][1]
                        best_dist = abs(dicom_z_values[0] - target_z)
                        for dz, idx in dicom_zs:
                            d = abs(dz - target_z)
                            if d < best_dist:
                                best_dist = d
                                best_idx = idx
                        ann_z_to_dicom_idx[az] = best_idx
                LOGGER.info(
                    "z_rank_mapping_built",
                    extra={
                        "ann_z_count": len(ann_zs),
                        "dicom_z_count": len(dicom_zs),
                        "mapped_count": len(ann_z_to_dicom_idx),
                    },
                )

        # Build ground truth mapped by slice index
        gt_by_slice_index: dict[int, list[dict[str, object]]] = {}
        for nodule in nodules:
            for sop_uid, nodule_slice in nodule.slices.items():
                slice_idx = sop_to_index.get(sop_uid)

                # Z-rank fallback
                if slice_idx is None and use_z_fallback:
                    slice_idx = ann_z_to_dicom_idx.get(nodule_slice.z_position)

                if slice_idx is not None:
                    entry = {
                        "bbox": {
                            "x": nodule_slice.bbox.x,
                            "y": nodule_slice.bbox.y,
                            "width": nodule_slice.bbox.width,
                            "height": nodule_slice.bbox.height,
                        },
                        "nodule_id": nodule.nodule_id,
                        "reader_count": nodule.reading_session_count,
                    }
                    gt_by_slice_index.setdefault(slice_idx, []).append(entry)

        LOGGER.info(
            "gt_mapping_complete",
            extra={
                "gt_slices_matched": len(gt_by_slice_index),
                "use_z_fallback": use_z_fallback,
            },
        )

        # ── Build prediction list ──
        abnormal_slices: list[dict[str, object]] = []
        for finding in findings:
            image_name = f"slice_{finding.slice_index:04d}.png"
            image_path = job_dir / image_name

            meta = (
                slice_metadata[finding.slice_index]
                if finding.slice_index < len(slice_metadata)
                else None
            )
            gt_matches = _lookup_gt_for_slice(
                nodules,
                meta.sop_instance_uid if meta else "",
                meta.image_position_z if meta else None,
                z_index,
                use_z_fallback,
            )

            gt_bbox_for_overlay = gt_matches[0][1].bbox if gt_matches else None
            comparison_img = render_comparison_overlay(
                slices[finding.slice_index],
                finding.bbox,
                gt_bbox_for_overlay,
            )
            Image.fromarray(comparison_img, mode="RGB").save(image_path, format="PNG")

            slice_data: dict[str, object] = {
                "slice_index": finding.slice_index,
                "confidence": round(float(finding.confidence), 4),
                "finding": finding.finding,
                "bbox": {
                    "x": finding.bbox.x,
                    "y": finding.bbox.y,
                    "width": finding.bbox.width,
                    "height": finding.bbox.height,
                },
                "image_url": (f"{_MEDIA_URL_PREFIX}/{job_dir.name}/{image_name}"),
            }

            if gt_matches:
                nodule_ann, nodule_sl = gt_matches[0]
                slice_data["gt_bbox"] = {
                    "x": nodule_sl.bbox.x,
                    "y": nodule_sl.bbox.y,
                    "width": nodule_sl.bbox.width,
                    "height": nodule_sl.bbox.height,
                }
                slice_data["gt_nodule_id"] = nodule_ann.nodule_id
                slice_data["gt_reader_count"] = nodule_ann.reading_session_count

            abnormal_slices.append(slice_data)

        # ── Generate images for missed GT nodules ──
        predicted_slice_indices = {f.slice_index for f in findings}
        missed_slices: list[dict[str, object]] = []
        for slice_idx, gt_list in gt_by_slice_index.items():
            if slice_idx not in predicted_slice_indices and slice_idx < len(slices):
                for gt_entry in gt_list:
                    gt_bbox_dict = gt_entry["bbox"]
                    gt_bbox = BBox(
                        x=int(gt_bbox_dict["x"]),  # type: ignore[index]
                        y=int(gt_bbox_dict["y"]),  # type: ignore[index]
                        width=int(gt_bbox_dict["width"]),  # type: ignore[index]
                        height=int(gt_bbox_dict["height"]),  # type: ignore[index]
                    )
                    missed_name = f"missed_{slice_idx:04d}.png"
                    missed_path = job_dir / missed_name
                    missed_img = render_ground_truth_overlay(slices[slice_idx], gt_bbox)
                    Image.fromarray(missed_img, mode="RGB").save(missed_path, format="PNG")
                    missed_slices.append(
                        {
                            "slice_index": slice_idx,
                            "image_url": (f"{_MEDIA_URL_PREFIX}/{job_dir.name}/{missed_name}"),
                            "gt_bbox": gt_bbox_dict,
                            "gt_nodule_id": str(gt_entry.get("nodule_id", "")),
                            "gt_reader_count": int(
                                gt_entry.get("reader_count", 0)  # type: ignore[arg-type]
                            ),
                        }
                    )

        # ── Run evaluation ──
        evaluation_data = evaluate_results(
            abnormal_slices,
            gt_by_slice_index,
            total_slices=len(slices),
        )

        evaluation_payload: dict[str, object] = {
            "has_ground_truth": evaluation_data.has_ground_truth,
            "total_slices": evaluation_data.total_slices,
            "model_flagged": evaluation_data.model_flagged,
            "gt_nodule_count": evaluation_data.gt_nodule_count,
            "true_positives": evaluation_data.true_positives,
            "false_positives": evaluation_data.false_positives,
            "missed": evaluation_data.missed,
            "per_slice": [
                {
                    "slice_index": s.slice_index,
                    "match_type": s.match_type,
                    "iou": s.iou,
                }
                for s in evaluation_data.per_slice
            ],
        }

        write_job_findings(
            job_dir,
            total_slices=len(slices),
            abnormal_slices=abnormal_slices,
            evaluation=evaluation_payload,
            missed_slices=missed_slices,
            annotation_source=annotation_source,
        )
        update_job_metadata(
            job_dir,
            status="completed",
            progress=100,
            slice_count=len(slices),
        )
    except Exception as exc:  # pragma: no cover
        LOGGER.exception("pipeline_failed", extra={"job_dir": str(job_dir)})
        update_job_metadata(job_dir, status="failed", progress=100, error=str(exc))
