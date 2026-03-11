"""
Pre-extract LUNA16 patches for sliding-window training.
========================================================
Paste into Cell 2 (after kaggle_prepare_data.py, before kaggle_train.py).
Internet NOT required.

Strategy — sliding-window-aligned training:
  POSITIVES: 24x24 patches centered on true nodule locations from
  candidates.csv (class=1).  Also extracts offset-augmented copies
  at ±8px to improve robustness.

  NEGATIVES: random 24x24 patches sampled from lung tissue regions
  (intensity 10-240 in lung window) that do NOT overlap with any
  nodule.  This matches the inference-time sliding window which
  proposes patches across the entire lung.

  Output: /kaggle/working/preextracted.npz
    - images:       (N, 24, 24) uint8
    - labels:       (N,) int64
    - series_uids:  (N,) str

Expects two Kaggle input datasets attached:
  - avc0706/luna16       (subsets 0-4 + annotations + candidates)
  - vafaeii/luna16       (subsets 0-9)
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import SimpleITK as sitk

# ──────────────────────────────────────────────────────────────────────────────
# Configuration (must match training script and app inference)
# ──────────────────────────────────────────────────────────────────────────────

PATCH_SIZE = 24
HALF = PATCH_SIZE // 2  # 12
WINDOW_CENTER = -600.0
WINDOW_WIDTH = 1500.0
NEG_PER_POS_RATIO = 10.0
# Offset augmentation: also extract patches shifted ±OFFSET from nodule center
# ±4px for 24x24 patches (1/6 of patch size)
POS_OFFSETS = [(0, 0), (-4, 0), (4, 0), (0, -4), (0, 4)]
SEED = 42
OUTPUT_PATH = Path("/kaggle/working/preextracted.npz")

SKIP_DIRS = {"seg-lungs-LUNA16", "seg-lungs-luna16"}

# Lung mask thresholds (must match app/services/inference.py)
LUNG_LO = 10
LUNG_HI = 240
# Minimum fraction of lung tissue in a patch to qualify as a candidate
LUNG_COVERAGE = 0.25
MIN_LUNG_PIXELS = int(LUNG_COVERAGE * PATCH_SIZE * PATCH_SIZE)

# Minimum distance (pixels) between a random negative and any nodule center
NODULE_EXCLUSION_RADIUS = PATCH_SIZE  # 24px — one full patch away from nodule center


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def apply_lung_window_uint8(img: np.ndarray) -> np.ndarray:
    lo = WINDOW_CENTER - WINDOW_WIDTH / 2
    hi = WINDOW_CENTER + WINDOW_WIDTH / 2
    clipped = np.clip(img, lo, hi)
    scaled = (clipped - lo) / (hi - lo)
    return np.rint(scaled * 255.0).astype(np.uint8)


def extract_patch(
    volume_slice: np.ndarray,
    cy: int,
    cx: int,
) -> np.ndarray:
    """Extract a PATCH_SIZE x PATCH_SIZE patch centered at (cy, cx).

    Zero-pads if the patch extends beyond the image boundary.
    volume_slice is a 2D array (H, W) in uint8 after windowing.
    """
    h, w = volume_slice.shape
    y0 = max(0, cy - HALF)
    y1 = min(h, cy + HALF)
    x0 = max(0, cx - HALF)
    x1 = min(w, cx + HALF)

    if y1 <= y0 or x1 <= x0:
        return np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=volume_slice.dtype)

    patch = np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=volume_slice.dtype)

    dy0 = y0 - (cy - HALF)
    dx0 = x0 - (cx - HALF)
    dy1 = dy0 + (y1 - y0)
    dx1 = dx0 + (x1 - x0)

    patch[dy0:dy1, dx0:dx1] = volume_slice[y0:y1, x0:x1]
    return patch


def _sample_random_lung_positions(
    windowed_slice: np.ndarray,
    n_samples: int,
    rng: np.random.RandomState,
    nodule_positions: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Sample random patch centers within lung tissue, excluding nodule regions.

    Uses the same lung mask as inference (pixels 10-240) and checks that
    at least LUNG_COVERAGE of the patch falls within lung tissue.
    """
    h, w = windowed_slice.shape
    img = windowed_slice.astype(np.float32)
    lung_mask = (img >= LUNG_LO) & (img <= LUNG_HI)

    # Build integral image for fast patch-level coverage checks
    integral = np.cumsum(np.cumsum(lung_mask.astype(np.int32), axis=0), axis=1)

    # Collect all valid positions
    valid_positions: list[tuple[int, int]] = []
    stride = HALF  # stride=24 to keep candidate pool manageable
    for cy in range(HALF, h - HALF + 1, stride):
        for cx in range(HALF, w - HALF + 1, stride):
            y0, y1 = cy - HALF, cy + HALF
            x0, x1 = cx - HALF, cx + HALF

            # Integral image sum
            s = integral[y1 - 1, x1 - 1]
            if y0 > 0:
                s -= integral[y0 - 1, x1 - 1]
            if x0 > 0:
                s -= integral[y1 - 1, x0 - 1]
            if y0 > 0 and x0 > 0:
                s += integral[y0 - 1, x0 - 1]

            if s < MIN_LUNG_PIXELS:
                continue

            # Exclude positions near any nodule
            too_close = False
            for nx, ny in nodule_positions:
                dx = cx - nx
                dy = cy - ny
                if dx * dx + dy * dy < NODULE_EXCLUSION_RADIUS * NODULE_EXCLUSION_RADIUS:
                    too_close = True
                    break
            if too_close:
                continue

            valid_positions.append((cx, cy))

    if not valid_positions:
        return []

    # Sample with replacement if needed
    if len(valid_positions) <= n_samples:
        return valid_positions

    indices = rng.choice(len(valid_positions), size=n_samples, replace=False)
    return [valid_positions[i] for i in indices]


def extract_patches_from_scan(
    mhd_path: Path,
    scan_candidates: list[dict],
    rng: np.random.RandomState,
) -> tuple[list[np.ndarray], list[int], list[str]]:
    """Load one scan, extract positive + random negative patches."""
    img = sitk.ReadImage(str(mhd_path))
    volume = sitk.GetArrayFromImage(img).astype(np.float32)
    origin = img.GetOrigin()  # (x, y, z) in mm
    spacing = img.GetSpacing()  # (x, y, z) in mm
    n_slices = volume.shape[0]

    # Sanity check: skip segmentation masks
    hu_min, hu_max = volume.min(), volume.max()
    if hu_max - hu_min < 100:
        raise ValueError(f"Likely segmentation mask (HU range [{hu_min:.0f}, {hu_max:.0f}])")

    uid = mhd_path.stem

    # Only keep positive candidates for this scan
    pos_candidates = [c for c in scan_candidates if c["cls"] == 1]

    patches: list[np.ndarray] = []
    labels: list[int] = []
    uids: list[str] = []

    # ── Extract positive patches (with offset augmentation) ──
    # Track nodule voxel positions per slice for exclusion zones
    nodule_positions_by_slice: dict[int, list[tuple[int, int]]] = {}

    for cand in pos_candidates:
        vx = int(round((cand["coordX"] - origin[0]) / spacing[0]))
        vy = int(round((cand["coordY"] - origin[1]) / spacing[1]))
        vz = int(round((cand["coordZ"] - origin[2]) / spacing[2]))

        if vz < 0 or vz >= n_slices:
            continue

        nodule_positions_by_slice.setdefault(vz, []).append((vx, vy))
        windowed = apply_lung_window_uint8(volume[vz])

        for dx, dy in POS_OFFSETS:
            cx_off = vx + dx
            cy_off = vy + dy
            patch = extract_patch(windowed, cy=cy_off, cx=cx_off)
            patches.append(patch)
            labels.append(1)
            uids.append(uid)

    n_pos = len(patches)
    if n_pos == 0:
        return patches, labels, uids

    # ── Extract random negative patches from lung regions ──
    n_neg_needed = int(n_pos * NEG_PER_POS_RATIO / len(POS_OFFSETS))

    # Sample negatives from random slices in this scan
    neg_collected = 0
    # Pick random slices (prefer slices with some lung tissue)
    slice_indices = rng.permutation(n_slices)

    for sz in slice_indices:
        if neg_collected >= n_neg_needed:
            break

        windowed = apply_lung_window_uint8(volume[sz])

        # Get nodule positions on this specific slice
        nodule_pos_this_slice = nodule_positions_by_slice.get(sz, [])

        # How many negatives to sample from this slice
        n_from_slice = min(3, n_neg_needed - neg_collected)

        positions = _sample_random_lung_positions(
            windowed, n_from_slice, rng, nodule_pos_this_slice
        )

        for cx, cy in positions:
            patch = extract_patch(windowed, cy=cy, cx=cx)
            patches.append(patch)
            labels.append(0)
            uids.append(uid)
            neg_collected += 1

    return patches, labels, uids


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

if OUTPUT_PATH.exists():
    OUTPUT_PATH.unlink()
    print("Deleted old preextracted.npz — re-extracting with random-negative approach")

if not OUTPUT_PATH.exists():
    import pandas as pd

    # Load candidates CSV (we only need positives from it now)
    candidates_csv = None
    for name in ("candidates.csv", "candidates_V2.csv"):
        for p in sorted(Path("/kaggle/input").rglob(name)):
            if "evaluationScript" in str(p):
                continue
            candidates_csv = p
            break
        if candidates_csv is not None:
            break
    if candidates_csv is None:
        raise FileNotFoundError("candidates.csv / candidates_V2.csv not found")
    print(f"Using candidates: {candidates_csv}")

    cand_df = pd.read_csv(candidates_csv)
    cls_col = "class" if "class" in cand_df.columns else "label"
    n_total = len(cand_df)
    n_pos_total = int((cand_df[cls_col] == 1).sum())
    print(f"  Total candidates: {n_total:,}")
    print(f"  Positive (used): {n_pos_total:,}")
    print(f"  Negative (ignored — using random lung patches instead): {n_total - n_pos_total:,}")

    # Group candidates by series UID
    cand_by_uid: dict[str, list[dict]] = {}
    for _, row in cand_df.iterrows():
        uid = row["seriesuid"]
        cand_by_uid.setdefault(uid, []).append(
            {
                "coordX": row["coordX"],
                "coordY": row["coordY"],
                "coordZ": row["coordZ"],
                "cls": int(row[cls_col]),
            }
        )

    # ── Discover all .mhd files across all input datasets, deduplicate by UID ──

    print("=" * 60)
    print("Discovering CT scans across all Kaggle input datasets")
    print("=" * 60)

    all_mhd: dict[str, Path] = {}
    for mhd in sorted(Path("/kaggle/input").rglob("*.mhd")):
        if any(skip in mhd.parts for skip in SKIP_DIRS):
            continue
        uid = mhd.stem
        raw_path = mhd.with_suffix(".raw")
        if uid not in all_mhd:
            all_mhd[uid] = mhd
        elif raw_path.exists() and not all_mhd[uid].with_suffix(".raw").exists():
            all_mhd[uid] = mhd

    print(f"  Found {len(all_mhd)} unique scans (deduplicated by series UID)")
    n_with_cand = sum(1 for uid in all_mhd if uid in cand_by_uid)
    print(f"  Scans with candidates: {n_with_cand}")

    # ── Extract patches ──────────────────────────────────────────────────────

    print()
    print("=" * 60)
    print("Extracting patches (48x48) — positives from annotations, negatives from random lung")
    print("=" * 60)

    all_images: list[np.ndarray] = []
    all_labels: list[int] = []
    all_uids: list[str] = []
    rng = np.random.RandomState(SEED)
    t0 = time.time()
    skipped = 0

    for i, (uid, mhd_path) in enumerate(all_mhd.items(), 1):
        scan_cands = cand_by_uid.get(uid, [])
        if not scan_cands:
            continue  # No candidates for this scan
        # Only proceed if there are positive candidates
        if not any(c["cls"] == 1 for c in scan_cands):
            continue
        try:
            imgs, lbls, uids_list = extract_patches_from_scan(mhd_path, scan_cands, rng)
            all_images.extend(imgs)
            all_labels.extend(lbls)
            all_uids.extend(uids_list)
        except Exception as e:
            skipped += 1
            if skipped <= 10:
                print(f"  Warning: skipping {uid[:40]}: {e}")
            elif skipped == 11:
                print("  (suppressing further warnings...)")

        if i % 50 == 0:
            n_pos = sum(1 for lbl in all_labels if lbl == 1)
            n_neg = sum(1 for lbl in all_labels if lbl == 0)
            elapsed = time.time() - t0
            print(
                f"  Processed {i}/{len(all_mhd)} scans "
                f"({len(all_images)} patches, pos={n_pos}, neg={n_neg}, "
                f"{elapsed:.0f}s elapsed)"
            )

    n_pos = sum(1 for lbl in all_labels if lbl == 1)
    n_neg = sum(1 for lbl in all_labels if lbl == 0)
    elapsed = time.time() - t0
    print(
        f"\n  Done: {len(all_images)} patches from "
        f"{len(all_mhd) - skipped} scans (skipped {skipped})"
    )
    print(f"  Positive: {n_pos} (with {len(POS_OFFSETS)}x offset augmentation)")
    print(f"  Negative: {n_neg} (random lung patches)")
    print(f"  Time: {elapsed:.0f}s ({elapsed / 60:.1f} min)")

    # ── Save to disk ──────────────────────────────────────────────────────

    print()
    print("=" * 60)
    print("Saving pre-extracted data")
    print("=" * 60)

    images_array = np.stack(all_images)  # (N, 48, 48) uint8
    labels_array = np.array(all_labels, dtype=np.int64)
    uids_array = np.array(all_uids, dtype=object)

    np.savez_compressed(
        OUTPUT_PATH,
        images=images_array,
        labels=labels_array,
        series_uids=uids_array,
    )

    file_mb = OUTPUT_PATH.stat().st_size / 1e6
    print(f"  Saved: {OUTPUT_PATH} ({file_mb:.0f} MB)")
    print(f"  Shape: {images_array.shape}")
    print(f"  Positive: {n_pos}, Negative: {n_neg}, Total: {len(labels_array)}")

print()
print("Pre-extracted data ready. Proceed to the training cell.")
