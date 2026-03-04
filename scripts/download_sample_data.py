"""Download a sample DICOM CT series from the TCIA LIDC-IDRI collection.

Fetches a single de-identified lung CT series (LIDC-IDRI-0078, 87 slices, ~46 MB)
via the public TCIA REST API.  No API key or registration required.

The downloaded .dcm files are saved to ``data/`` (gitignored) and can be zipped
for upload to the demo app.

Usage::

    python scripts/download_sample_data.py              # default series
    python scripts/download_sample_data.py --list       # list available series for patient
    python scripts/download_sample_data.py --patient LIDC-IDRI-0001  # different patient
    python scripts/download_sample_data.py --zip        # also create a ready-to-upload zip
"""

from __future__ import annotations

import argparse
import io
import sys
import zipfile
from pathlib import Path

import httpx

TCIA_BASE = "https://services.cancerimagingarchive.net/nbia-api/services/v1"
COLLECTION = "LIDC-IDRI"

# LIDC-IDRI-0078: 87 slices, 4 annotated nodules, ~46 MB, CC BY 3.0
DEFAULT_PATIENT = "LIDC-IDRI-0078"
DEFAULT_SERIES_UID = "1.3.6.1.4.1.14519.5.2.1.6279.6001.303494235102183795724852353824"

TIMEOUT_META = 30
TIMEOUT_DOWNLOAD = 600


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a sample DICOM CT series from the TCIA LIDC-IDRI collection.",
    )
    parser.add_argument(
        "--patient",
        default=DEFAULT_PATIENT,
        help=f"Patient ID to download (default: {DEFAULT_PATIENT}).",
    )
    parser.add_argument(
        "--series-uid",
        default=None,
        help="Specific SeriesInstanceUID.  If omitted, the first CT series is used.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data"),
        help="Output directory for extracted DICOM files (default: data/).",
    )
    parser.add_argument(
        "--zip",
        action="store_true",
        dest="create_zip",
        help="Also create a ready-to-upload .zip archive in the output directory.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        dest="list_series",
        help="List available CT series for the patient and exit.",
    )
    return parser.parse_args()


def fetch_series_list(patient_id: str) -> list[dict[str, object]]:
    response = httpx.get(
        f"{TCIA_BASE}/getSeries",
        params={
            "Collection": COLLECTION,
            "PatientID": patient_id,
            "Modality": "CT",
            "format": "json",
        },
        timeout=TIMEOUT_META,
    )
    response.raise_for_status()
    return response.json()


def download_series(series_uid: str, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading series {series_uid} ...")
    response = httpx.get(
        f"{TCIA_BASE}/getImage",
        params={"SeriesInstanceUID": series_uid},
        timeout=TIMEOUT_DOWNLOAD,
    )
    response.raise_for_status()

    content = io.BytesIO(response.content)
    if not zipfile.is_zipfile(content):
        print("Error: TCIA response is not a valid ZIP archive.", file=sys.stderr)
        sys.exit(1)

    content.seek(0)
    with zipfile.ZipFile(content) as archive:
        archive.extractall(output_dir)

    return sorted(output_dir.glob("**/*.dcm"))


def create_upload_zip(dcm_files: list[Path], output_path: Path) -> Path:
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for dcm_file in dcm_files:
            archive.write(dcm_file, arcname=dcm_file.name)
    return output_path


def main() -> None:
    args = parse_args()

    series_list = fetch_series_list(args.patient)
    if not series_list:
        print(f"No CT series found for patient {args.patient}.", file=sys.stderr)
        sys.exit(1)

    if args.list_series:
        print(f"CT series for {args.patient}:\n")
        for series in series_list:
            print(f"  UID:    {series.get('SeriesInstanceUID')}")
            print(f"  Slices: {series.get('ImageCount')}")
            size_bytes = int(series.get("FileSize", 0))
            print(f"  Size:   {size_bytes / 1024 / 1024:.1f} MB")
            print()
        return

    series_uid = args.series_uid
    if series_uid is None:
        if args.patient == DEFAULT_PATIENT:
            series_uid = DEFAULT_SERIES_UID
        else:
            series_uid = str(series_list[0]["SeriesInstanceUID"])
            print(f"Using first available series: {series_uid}")

    case_dir = args.output / args.patient
    dcm_files = download_series(series_uid, case_dir)
    print(f"Extracted {len(dcm_files)} DICOM files to {case_dir}")

    if args.create_zip:
        zip_path = args.output / f"{args.patient}.zip"
        create_upload_zip(dcm_files, zip_path)
        print(f"Created upload archive: {zip_path}")

    print("\nDone. To use with the demo app:")
    if not args.create_zip:
        print("  python scripts/download_sample_data.py --zip")
        print(f"  # Then upload data/{args.patient}.zip at http://localhost:8000")
    else:
        print(f"  Upload {zip_path} at http://localhost:8000")


if __name__ == "__main__":
    main()
