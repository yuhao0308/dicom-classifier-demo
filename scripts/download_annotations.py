"""Download LIDC-IDRI XML annotation files from TCIA.

Fetches the ``LIDC-XML-only.zip`` archive (8.62 MB, CC BY 3.0) from the
Cancer Imaging Archive and extracts the annotation XML for a single patient.

Usage::

    python scripts/download_annotations.py                       # default (LIDC-IDRI-0078)
    python scripts/download_annotations.py --patient LIDC-IDRI-0001
"""

from __future__ import annotations

import argparse
import io
import re
import sys
import zipfile
from pathlib import Path

import httpx

LIDC_XML_ZIP_URL = (
    "https://wiki.cancerimagingarchive.net/download/attachments/1966254/"
    "LIDC-XML-only.zip?version=1&modificationDate=1530215018015&api=v2"
)
DEFAULT_PATIENT = "LIDC-IDRI-0078"
TIMEOUT_DOWNLOAD = 120


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download LIDC-IDRI XML annotations from TCIA.",
    )
    parser.add_argument(
        "--patient",
        default=DEFAULT_PATIENT,
        help=f"Patient ID to extract annotations for (default: {DEFAULT_PATIENT}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data"),
        help="Output base directory (default: data/).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        dest="extract_all",
        help="Extract all annotation XML files, not just the target patient.",
    )
    return parser.parse_args()


def _patient_id_to_xml_name(patient_id: str) -> str:
    """Convert e.g. ``LIDC-IDRI-0078`` → ``078.xml``."""
    match = re.search(r"(\d+)$", patient_id)
    if not match:
        print(f"Error: cannot parse numeric ID from '{patient_id}'.", file=sys.stderr)
        sys.exit(1)
    return f"{int(match.group(1)):03d}.xml"


def download_xml_zip() -> bytes:
    """Download the full LIDC-XML-only.zip archive."""
    print("Downloading LIDC-XML-only.zip from TCIA ...")
    response = httpx.get(LIDC_XML_ZIP_URL, timeout=TIMEOUT_DOWNLOAD, follow_redirects=True)
    response.raise_for_status()
    print(f"Downloaded {len(response.content) / 1024 / 1024:.1f} MB.")
    return response.content


def extract_patient_xml(
    zip_content: bytes,
    patient_id: str,
    output_dir: Path,
) -> Path:
    """Extract the annotation XML for a single patient from the zip archive."""
    target_name = _patient_id_to_xml_name(patient_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    dest_path = output_dir / "annotations.xml"

    with zipfile.ZipFile(io.BytesIO(zip_content)) as archive:
        for entry in archive.infolist():
            if entry.filename.endswith(target_name):
                dest_path.write_bytes(archive.read(entry))
                print(f"Extracted {entry.filename} → {dest_path}")
                return dest_path

    print(f"Error: annotation file '{target_name}' not found in archive.", file=sys.stderr)
    sys.exit(1)


def extract_all_xml(zip_content: bytes, output_dir: Path) -> list[Path]:
    """Extract all annotation XML files from the zip archive."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    with zipfile.ZipFile(io.BytesIO(zip_content)) as archive:
        for entry in archive.infolist():
            if entry.is_dir() or not entry.filename.endswith(".xml"):
                continue
            basename = Path(entry.filename).name
            dest_path = output_dir / basename
            dest_path.write_bytes(archive.read(entry))
            paths.append(dest_path)

    print(f"Extracted {len(paths)} XML files to {output_dir}")
    return paths


def main() -> None:
    args = parse_args()
    zip_content = download_xml_zip()

    if args.extract_all:
        annotations_dir = args.output / "annotations"
        extract_all_xml(zip_content, annotations_dir)
    else:
        case_dir = args.output / args.patient
        extract_patient_xml(zip_content, args.patient, case_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
