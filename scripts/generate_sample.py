"""Generate sample DICOM datasets for the demo.

Creates 3 sample cases:
  1. LIDC-IDRI-0078 (real, trimmed to 10 slices around nodules, PHI stripped)
  2. Synthetic-Chest-001 (synthetic CT-like data with a fake nodule)
  3. Synthetic-Chest-002 (synthetic CT-like data with two fake nodules)

Each case is saved as:
  app/static/samples/<case_id>.zip        — DICOM .zip archive
  app/static/samples/<case_id>_ann.xml    — LIDC-style annotation XML

Usage:
  python scripts/generate_sample.py
"""

from __future__ import annotations

import io
import shutil
import zipfile
from pathlib import Path

import numpy as np
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

ROOT = Path(__file__).resolve().parent.parent
SAMPLES_DIR = ROOT / "app" / "static" / "samples"
REAL_DATA_DIR = ROOT / "data" / "LIDC-IDRI-0078"

# PHI tags to strip
PHI_TAGS = [
    "PatientName",
    "PatientID",
    "PatientBirthDate",
    "PatientSex",
    "PatientAge",
    "PatientWeight",
    "InstitutionName",
    "InstitutionAddress",
    "ReferringPhysicianName",
    "PerformingPhysicianName",
    "StudyDate",
    "StudyTime",
    "AccessionNumber",
    "OtherPatientIDs",
]


def _make_dicom(
    pixel_array: np.ndarray,
    *,
    instance_number: int,
    slice_location: float,
    sop_uid: str | None = None,
    series_uid: str | None = None,
    study_uid: str | None = None,
    patient_id: str = "SYNTHETIC",
) -> bytes:
    """Create a minimal DICOM CT file in memory."""
    file_meta = FileMetaDataset()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.MediaStorageSOPClassUID = CTImageStorage
    file_meta.MediaStorageSOPInstanceUID = sop_uid or generate_uid()
    file_meta.ImplementationClassUID = generate_uid()

    ds = FileDataset("", {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = study_uid or generate_uid()
    ds.SeriesInstanceUID = series_uid or generate_uid()
    ds.Modality = "CT"
    ds.PatientID = patient_id
    ds.PatientName = "Anonymous"
    ds.InstanceNumber = instance_number
    ds.SliceLocation = slice_location
    ds.ImagePositionPatient = [0.0, 0.0, slice_location]
    ds.Rows = pixel_array.shape[0]
    ds.Columns = pixel_array.shape[1]
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 1
    ds.RescaleSlope = 1.0
    ds.RescaleIntercept = -1024.0
    ds.PixelData = pixel_array.astype(np.int16).tobytes()
    ds.is_implicit_VR = False
    ds.is_little_endian = True

    buf = io.BytesIO()
    ds.save_as(buf, write_like_original=False)
    return buf.getvalue()


def _generate_ct_slice(
    rng: np.random.Generator,
    size: int = 512,
    *,
    nodules: list[tuple[int, int, int]] | None = None,
) -> np.ndarray:
    """Generate a synthetic CT-like slice in HU-offset values.

    Returns int16 values where 0 = -1024 HU (air) after RescaleIntercept.
    So lung tissue (~-700 HU) = 324, soft tissue (~40 HU) = 1064.
    """
    # Start with air background
    img = np.full((size, size), 0, dtype=np.int16)

    # Body ellipse (soft tissue ~1064 = 40 HU)
    cy, cx = size // 2, size // 2
    yy, xx = np.ogrid[:size, :size]
    body_mask = ((xx - cx) / 180) ** 2 + ((yy - cy) / 220) ** 2 < 1.0
    img[body_mask] = 1064 + rng.integers(-20, 20, size=img[body_mask].shape).astype(np.int16)

    # Left lung (air-ish ~324 = -700 HU)
    lung_l = ((xx - (cx - 70)) / 90) ** 2 + ((yy - (cy - 10)) / 140) ** 2 < 1.0
    img[lung_l] = 324 + rng.integers(-30, 30, size=img[lung_l].shape).astype(np.int16)

    # Right lung
    lung_r = ((xx - (cx + 70)) / 90) ** 2 + ((yy - (cy - 10)) / 140) ** 2 < 1.0
    img[lung_r] = 324 + rng.integers(-30, 30, size=img[lung_r].shape).astype(np.int16)

    # Spine (bone ~1800 = 776 HU)
    spine = ((xx - cx) / 25) ** 2 + ((yy - (cy + 80)) / 35) ** 2 < 1.0
    img[spine] = 1800 + rng.integers(-50, 50, size=img[spine].shape).astype(np.int16)

    # Add nodules if specified: (x_center, y_center, radius)
    if nodules:
        for nx, ny, nr in nodules:
            nodule_mask = ((xx - nx) ** 2 + (yy - ny) ** 2) < nr**2
            img[nodule_mask] = 1064 + rng.integers(-30, 30, size=img[nodule_mask].shape).astype(
                np.int16
            )

    return img


def _make_annotation_xml(
    case_id: str,
    series_uid: str,
    nodules: list[dict],
) -> str:
    """Generate a minimal LIDC-style annotation XML."""
    rois = []
    for nod in nodules:
        edge_maps = []
        cx, cy, r = nod["x"], nod["y"], nod["radius"]
        # Generate circular contour points
        for angle in range(0, 360, 15):
            rad = angle * np.pi / 180
            ex = int(round(cx + r * np.cos(rad)))
            ey = int(round(cy + r * np.sin(rad)))
            edge_maps.append(
                f"          <edgeMap>\n"
                f"            <xCoord>{ex}</xCoord>\n"
                f"            <yCoord>{ey}</yCoord>\n"
                f"          </edgeMap>"
            )

        for sop_uid, z_pos in zip(nod["sop_uids"], nod["z_positions"], strict=True):
            rois.append(
                f"      <roi>\n"
                f"        <imageZposition>{z_pos:.6f}</imageZposition>\n"
                f"        <imageSOP_UID>{sop_uid}</imageSOP_UID>\n"
                f"        <inclusion>TRUE</inclusion>\n"
                + "\n".join(edge_maps)
                + "\n      </roi>"
            )

    nodule_blocks = []
    for i, nod in enumerate(nodules):
        nodule_blocks.append(
            f"    <unblindedReadNodule>\n"
            f"      <noduleID>Nodule {i + 1:03d}</noduleID>\n"
            + "\n".join(r for r in rois[i * len(nodules[0]["sop_uids"]) :])
            + "\n    </unblindedReadNodule>"
        )

    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<LidcReadMessage uid="{generate_uid()}" '
        'xmlns="http://www.nih.gov">\n'
        "<ResponseHeader>\n"
        "  <Version>1.8.1</Version>\n"
        f"  <SeriesInstanceUid>{series_uid}</SeriesInstanceUid>\n"
        "</ResponseHeader>\n"
        "<readingSession>\n"
        "    <annotationVersion>3.12</annotationVersion>\n"
        + "\n".join(nodule_blocks)
        + "\n</readingSession>\n"
        "</LidcReadMessage>\n"
    )


def generate_synthetic_case(
    case_id: str,
    patient_id: str,
    n_slices: int,
    nodule_specs: list[dict],
    rng: np.random.Generator,
) -> None:
    """Generate a synthetic DICOM series with annotation XML."""
    series_uid = generate_uid()
    study_uid = generate_uid()

    zip_path = SAMPLES_DIR / f"{case_id}.zip"
    ann_path = SAMPLES_DIR / f"{case_id}_ann.xml"

    # Determine which slices have nodules
    for nod in nodule_specs:
        nod["sop_uids"] = []
        nod["z_positions"] = []

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for i in range(n_slices):
            z_pos = -200.0 + i * 2.5
            sop_uid = generate_uid()

            # Check which nodules appear on this slice
            slice_nodules = []
            for nod in nodule_specs:
                if nod["slice_start"] <= i <= nod["slice_end"]:
                    slice_nodules.append((nod["x"], nod["y"], nod["radius"]))
                    nod["sop_uids"].append(sop_uid)
                    nod["z_positions"].append(z_pos)

            pixel_array = _generate_ct_slice(
                rng,
                nodules=slice_nodules if slice_nodules else None,
            )
            dcm_bytes = _make_dicom(
                pixel_array,
                instance_number=i + 1,
                slice_location=z_pos,
                sop_uid=sop_uid,
                series_uid=series_uid,
                study_uid=study_uid,
                patient_id=patient_id,
            )
            zf.writestr(f"{i + 1:08d}.dcm", dcm_bytes)

    # Generate annotation XML
    xml_content = _make_annotation_xml(case_id, series_uid, nodule_specs)
    ann_path.write_text(xml_content)

    print(f"  Created {zip_path.name} ({zip_path.stat().st_size / 1024:.0f} KB, {n_slices} slices)")
    print(f"  Created {ann_path.name} ({len(nodule_specs)} nodule(s))")


def prepare_real_case() -> None:
    """Trim LIDC-IDRI-0078 to 10 slices around nodules, strip PHI."""
    import pydicom

    case_id = "LIDC-IDRI-0078"
    zip_path = SAMPLES_DIR / f"{case_id}.zip"
    ann_src = REAL_DATA_DIR / "annotations.xml"
    ann_dst = SAMPLES_DIR / f"{case_id}_ann.xml"

    if not REAL_DATA_DIR.exists():
        print(f"  Skipping {case_id}: source data not found at {REAL_DATA_DIR}")
        return

    # Read all slices, sort by instance number
    dcm_files = sorted(REAL_DATA_DIR.glob("*.dcm"))
    if not dcm_files:
        print(f"  Skipping {case_id}: no .dcm files found")
        return

    # Pick 10 slices from the middle (where nodules typically are)
    total = len(dcm_files)
    mid = total // 2
    start = max(0, mid - 5)
    selected = dcm_files[start : start + 10]

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for dcm_path in selected:
            ds = pydicom.dcmread(dcm_path)
            # Strip PHI
            for tag in PHI_TAGS:
                if hasattr(ds, tag):
                    delattr(ds, tag)
            ds.PatientName = "Anonymous"
            ds.PatientID = case_id

            buf = io.BytesIO()
            ds.save_as(buf, write_like_original=False)
            zf.writestr(dcm_path.name, buf.getvalue())

    print(
        f"  Created {zip_path.name} ({zip_path.stat().st_size / 1024:.0f} KB, "
        f"{len(selected)} slices from {total})"
    )

    # Copy annotation XML
    if ann_src.exists():
        shutil.copy2(ann_src, ann_dst)
        print(f"  Copied {ann_dst.name}")
    else:
        print(f"  Warning: no annotations.xml found for {case_id}")


def main() -> None:
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    print("Generating sample datasets...")
    print()

    # Case 1: Real LIDC-IDRI-0078 (trimmed)
    print("[1/3] LIDC-IDRI-0078 (real, trimmed)")
    prepare_real_case()
    print()

    rng = np.random.default_rng(42)

    # Case 2: Synthetic with 1 nodule
    print("[2/3] Synthetic-Chest-001 (1 nodule)")
    generate_synthetic_case(
        case_id="Synthetic-Chest-001",
        patient_id="SYNTH-001",
        n_slices=10,
        nodule_specs=[
            {"x": 190, "y": 250, "radius": 8, "slice_start": 3, "slice_end": 6},
        ],
        rng=rng,
    )
    print()

    # Case 3: Synthetic with 2 nodules
    print("[3/3] Synthetic-Chest-002 (2 nodules)")
    generate_synthetic_case(
        case_id="Synthetic-Chest-002",
        patient_id="SYNTH-002",
        n_slices=10,
        nodule_specs=[
            {"x": 320, "y": 230, "radius": 6, "slice_start": 2, "slice_end": 4},
            {"x": 185, "y": 280, "radius": 10, "slice_start": 5, "slice_end": 8},
        ],
        rng=rng,
    )
    print()
    print("Done! Sample files saved to app/static/samples/")


if __name__ == "__main__":
    main()
