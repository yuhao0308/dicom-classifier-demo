"""
LUNA16 Dataset Preparation & Verification — Kaggle Cell
========================================================
Paste this into the FIRST cell of your Kaggle notebook and run it
BEFORE the training script (kaggle_train.py).

It will:
  1. Install missing pip packages
  2. Discover which LUNA16 datasets you've attached
  3. Verify .mhd/.raw scan files are accessible
  4. Verify annotations & candidates CSVs exist
  5. Print a summary with dataset statistics
  6. Flag any problems before training starts

Kaggle Dataset Setup (do this BEFORE running)
----------------------------------------------
In the Kaggle notebook sidebar → "Add data" → search and add these:

  OPTION A — Single dataset (easiest, most popular):
    • Search: "luna16"
    • Add: "Luna16" by avc0706  (34.5 GB, ~6800 downloads)
      → Contains subset0–subset9 (.mhd/.raw) + annotations.csv + candidates.csv

  OPTION B — Separate datasets (if Option A is missing CSVs):
    • Scans:  "Luna16" by avc0706
    • CSVs:   "luna16-lung-cancer-dataset" by fanbyprinciple  (346 MB)
              → Contains annotations.csv, candidates_V2.csv

  OPTION C — Official from Zenodo (full 66 GB, upload as private dataset):
    Download from https://zenodo.org/records/2595813 (Part 1)
                + https://zenodo.org/records/2596479 (Part 2)
    Upload to Kaggle as a private dataset.

After adding datasets, they appear read-only at:
  /kaggle/input/<dataset-slug>/...
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# 0. INSTALL DEPENDENCIES
# ──────────────────────────────────────────────────────────────────────────────

REQUIRED_PACKAGES = [
    "SimpleITK",
    "grad-cam",
    "scikit-learn",
]

print("=" * 60)
print("STEP 0: Installing dependencies")
print("=" * 60)
for pkg in REQUIRED_PACKAGES:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", pkg],
        stdout=subprocess.DEVNULL,
    )
    print(f"  ✓ {pkg}")
print()

# ──────────────────────────────────────────────────────────────────────────────
# 1. DISCOVER ATTACHED DATASETS
# ──────────────────────────────────────────────────────────────────────────────

INPUT_ROOT = Path("/kaggle/input")

print("=" * 60)
print("STEP 1: Discovering attached datasets")
print("=" * 60)

if not INPUT_ROOT.exists():
    print("ERROR: /kaggle/input does not exist. Are you running on Kaggle?")
    sys.exit(1)

datasets = sorted(p for p in INPUT_ROOT.iterdir() if p.is_dir())
if not datasets:
    print("ERROR: No datasets attached!")
    print("       Go to sidebar → 'Add data' → search 'luna16' → add dataset")
    sys.exit(1)

for ds in datasets:
    # Count files and estimate size
    files = list(ds.rglob("*"))
    file_count = sum(1 for f in files if f.is_file())
    total_bytes = sum(f.stat().st_size for f in files if f.is_file())
    print(f"  📁 {ds.name}  ({file_count} files, {total_bytes / 1e9:.1f} GB)")

print()

# ──────────────────────────────────────────────────────────────────────────────
# 2. FIND .mhd SCAN FILES
# ──────────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("STEP 2: Finding CT scan files (.mhd)")
print("=" * 60)

mhd_files = sorted(INPUT_ROOT.rglob("*.mhd"))
raw_files = sorted(INPUT_ROOT.rglob("*.raw"))

# Group by parent directory to show subset structure
subset_counts: dict[str, int] = {}
for mhd in mhd_files:
    parent = mhd.parent.name
    subset_counts[parent] = subset_counts.get(parent, 0) + 1

if mhd_files:
    print(f"  Found {len(mhd_files)} .mhd files, {len(raw_files)} .raw files")
    print(f"  Across {len(subset_counts)} directories:")
    for dirname, count in sorted(subset_counts.items()):
        print(f"    • {dirname}: {count} scans")

    if len(mhd_files) != len(raw_files):
        print(f"  ⚠️  WARNING: .mhd count ({len(mhd_files)}) != .raw count ({len(raw_files)})")
        print("     Some scans may be incomplete.")
else:
    print("  ❌ No .mhd files found!")
    print("     The LUNA16 dataset uses MetaImage format (.mhd + .raw pairs).")
    print("     Make sure you added the correct dataset (e.g., 'Luna16' by avc0706).")
    print()
    # Check if there are .zip files that need extracting
    zips = list(INPUT_ROOT.rglob("subset*.zip"))
    if zips:
        print("  💡 Found compressed subsets that need extracting:")
        for z in zips:
            print(f"     {z}")
        print()
        print("  Extracting now (this may take several minutes)...")

        import zipfile

        extract_dir = Path("/kaggle/working/luna16_extracted")
        extract_dir.mkdir(parents=True, exist_ok=True)

        for z in sorted(zips):
            print(f"    Extracting {z.name}...", end=" ", flush=True)
            with zipfile.ZipFile(z, "r") as zf:
                zf.extractall(extract_dir)
            print("done")

        # Re-scan
        mhd_files = sorted(extract_dir.rglob("*.mhd"))
        raw_files = sorted(extract_dir.rglob("*.raw"))
        print(f"\n  After extraction: {len(mhd_files)} .mhd, {len(raw_files)} .raw")

        if mhd_files:
            print("  ✓ Scans extracted successfully!")
            print(f"  📍 Location: {extract_dir}")
            print()
            print("  ⚠️  IMPORTANT: Update the training script's input_root:")
            print(f'     input_root: Path = Path("{extract_dir}")')
        else:
            print("  ❌ Still no .mhd files after extraction. Check dataset contents.")

print()

# ──────────────────────────────────────────────────────────────────────────────
# 3. FIND ANNOTATION CSVs
# ──────────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("STEP 3: Finding annotation CSV files")
print("=" * 60)

csv_names = [
    "annotations.csv",
    "candidates.csv",
    "candidates_V2.csv",
    "annotations_excluded.csv",
    "sampleSubmission.csv",
]

found_csvs: dict[str, Path] = {}
for name in csv_names:
    matches = list(INPUT_ROOT.rglob(name))
    # Also check extracted dir
    extract_dir = Path("/kaggle/working/luna16_extracted")
    if extract_dir.exists():
        matches.extend(extract_dir.rglob(name))

    if matches:
        found_csvs[name] = matches[0]
        # Show row count
        with open(matches[0]) as f:
            row_count = sum(1 for _ in f) - 1  # subtract header
        print(f"  ✓ {name:<30s}  {row_count:>8,} rows  → {matches[0]}")
    else:
        tag = "❌ REQUIRED" if name in ("annotations.csv", "candidates.csv") else "  (optional)"
        print(f"  {tag}: {name} not found")

# Accept candidates_V2.csv as fallback for candidates.csv
if "candidates.csv" not in found_csvs and "candidates_V2.csv" in found_csvs:
    print("\n  💡 candidates.csv not found, but candidates_V2.csv is available.")
    print("     The training script will use candidates_V2.csv automatically.")

print()

# ──────────────────────────────────────────────────────────────────────────────
# 4. QUICK DATA INTEGRITY CHECK
# ──────────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("STEP 4: Data integrity check")
print("=" * 60)

if mhd_files and ("annotations.csv" in found_csvs):
    import pandas as pd

    ann = pd.read_csv(found_csvs["annotations.csv"])
    scan_uids = {p.stem for p in mhd_files}
    ann_uids = set(ann["seriesuid"].unique())

    matched = ann_uids & scan_uids
    missing = ann_uids - scan_uids

    print(f"  Annotations reference {len(ann_uids)} unique scans")
    print(f"  Scans on disk:        {len(scan_uids)}")
    print(f"  Matched:              {len(matched)} ({100 * len(matched) / len(ann_uids):.0f}%)")
    if missing:
        print(f"  Missing:              {len(missing)} (annotations without scan files)")

    # Check a random scan can be read
    print("\n  Reading a sample scan to verify format...", end=" ", flush=True)
    try:
        import SimpleITK as sitk

        test_mhd = mhd_files[0]
        img = sitk.ReadImage(str(test_mhd))
        arr = sitk.GetArrayFromImage(img)
        print(f"OK")
        print(f"    Scan:    {test_mhd.stem}")
        print(f"    Shape:   {arr.shape} (slices × H × W)")
        print(f"    Spacing: {img.GetSpacing()}")
        print(f"    Origin:  {img.GetOrigin()}")
        print(f"    HU range: [{arr.min()}, {arr.max()}]")
    except Exception as e:
        print(f"FAILED")
        print(f"    Error: {e}")
else:
    print("  ⚠️  Skipped — missing scan files or annotations.csv")

print()

# ──────────────────────────────────────────────────────────────────────────────
# 5. CANDIDATE STATISTICS
# ──────────────────────────────────────────────────────────────────────────────

cand_csv = found_csvs.get("candidates.csv") or found_csvs.get("candidates_V2.csv")
if cand_csv and mhd_files:
    print("=" * 60)
    print("STEP 5: Candidate statistics")
    print("=" * 60)

    cand = pd.read_csv(cand_csv)
    scan_uids = {p.stem for p in mhd_files}
    cand_available = cand[cand["seriesuid"].isin(scan_uids)]

    n_pos = (cand_available["class"] == 1).sum()
    n_neg = (cand_available["class"] == 0).sum()

    print(f"  Total candidates:       {len(cand):>10,}")
    print(f"  With available scans:   {len(cand_available):>10,}")
    print(f"    Positive (nodule):    {n_pos:>10,}")
    print(f"    Negative (normal):    {n_neg:>10,}")
    print(f"    Ratio (neg:pos):      {n_neg / max(n_pos, 1):>10.0f}:1")
    print()

# ──────────────────────────────────────────────────────────────────────────────
# 6. SUMMARY & NEXT STEPS
# ──────────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("SUMMARY")
print("=" * 60)

has_scans = len(mhd_files) > 0
has_annotations = "annotations.csv" in found_csvs
has_candidates = "candidates.csv" in found_csvs or "candidates_V2.csv" in found_csvs

checks = [
    ("CT scans (.mhd/.raw)", has_scans, f"{len(mhd_files)} scans"),
    ("annotations.csv", has_annotations, "found"),
    ("candidates.csv / V2", has_candidates, "found"),
]

all_ok = True
for label, ok, detail in checks:
    status = "✓" if ok else "✗"
    print(f"  [{status}] {label:<30s}  {detail if ok else 'MISSING'}")
    if not ok:
        all_ok = False

print()
if all_ok:
    print("  ✅ All data ready! Proceed to the training cell.")

    # Print the input_root the training script should use
    # Check if scans are in /kaggle/input or extracted to /working
    scan_root = mhd_files[0].parent
    while scan_root.parent != INPUT_ROOT and scan_root.parent != Path("/kaggle/working"):
        scan_root = scan_root.parent

    if str(scan_root).startswith("/kaggle/working"):
        print()
        print("  ⚠️  Scans were extracted to /kaggle/working/luna16_extracted")
        print("     Update Config in the training script:")
        print('     input_root: Path = Path("/kaggle/working/luna16_extracted")')
    print()
    print("  Copy the training script (kaggle_train.py) into the next cell and run.")
else:
    print("  ❌ Some data is missing. Check the instructions at the top of this script.")
    print("     Add the missing datasets via sidebar → 'Add data'")
print("=" * 60)
