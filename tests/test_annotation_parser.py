from __future__ import annotations

from pathlib import Path

import pytest

from app.services.annotation_parser import (
    NoduleAnnotation,
    NoduleSlice,
    find_annotation_for_patient,
    get_ground_truth_for_slice,
    parse_annotation_xml,
)
from app.services.postprocess import BBox

# Minimal valid LIDC XML with 2 reading sessions annotating the same nodule
_SAMPLE_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<LidcReadMessage xmlns="http://www.nih.gov">
  <readingSession>
    <unblindedReadNodule>
      <noduleID>Nodule 001</noduleID>
      <roi>
        <imageZposition>-120.0</imageZposition>
        <imageSOP_UID>1.2.3.4.5.001</imageSOP_UID>
        <edgeMap><xCoord>100</xCoord><yCoord>200</yCoord></edgeMap>
        <edgeMap><xCoord>110</xCoord><yCoord>200</yCoord></edgeMap>
        <edgeMap><xCoord>110</xCoord><yCoord>210</yCoord></edgeMap>
        <edgeMap><xCoord>100</xCoord><yCoord>210</yCoord></edgeMap>
      </roi>
      <roi>
        <imageZposition>-117.5</imageZposition>
        <imageSOP_UID>1.2.3.4.5.002</imageSOP_UID>
        <edgeMap><xCoord>102</xCoord><yCoord>202</yCoord></edgeMap>
        <edgeMap><xCoord>108</xCoord><yCoord>208</yCoord></edgeMap>
      </roi>
    </unblindedReadNodule>
  </readingSession>
  <readingSession>
    <unblindedReadNodule>
      <noduleID>Nodule 001b</noduleID>
      <roi>
        <imageZposition>-120.0</imageZposition>
        <imageSOP_UID>1.2.3.4.5.001</imageSOP_UID>
        <edgeMap><xCoord>101</xCoord><yCoord>201</yCoord></edgeMap>
        <edgeMap><xCoord>112</xCoord><yCoord>212</yCoord></edgeMap>
      </roi>
    </unblindedReadNodule>
    <unblindedReadNodule>
      <noduleID>Nodule 002</noduleID>
      <roi>
        <imageZposition>-80.0</imageZposition>
        <imageSOP_UID>1.2.3.4.5.010</imageSOP_UID>
        <edgeMap><xCoord>300</xCoord><yCoord>300</yCoord></edgeMap>
        <edgeMap><xCoord>320</xCoord><yCoord>320</yCoord></edgeMap>
      </roi>
    </unblindedReadNodule>
  </readingSession>
</LidcReadMessage>
"""


@pytest.fixture
def sample_xml(tmp_path: Path) -> Path:
    xml_path = tmp_path / "annotations.xml"
    xml_path.write_text(_SAMPLE_XML, encoding="utf-8")
    return xml_path


def test_parse_annotation_xml_returns_merged_nodules(sample_xml: Path) -> None:
    nodules = parse_annotation_xml(sample_xml)

    # Nodule 001 and 001b should merge (same region), Nodule 002 is separate
    assert len(nodules) == 2

    # Find the merged nodule (the one with reader_count > 1)
    merged = [n for n in nodules if n.reading_session_count == 2]
    assert len(merged) == 1
    assert merged[0].reading_session_count == 2
    assert "1.2.3.4.5.001" in merged[0].slices

    # Check the separate nodule
    separate = [n for n in nodules if n.reading_session_count == 1]
    assert len(separate) == 1
    assert separate[0].nodule_id == "Nodule 002"
    assert "1.2.3.4.5.010" in separate[0].slices


def test_parse_annotation_xml_extracts_correct_bbox(sample_xml: Path) -> None:
    nodules = parse_annotation_xml(sample_xml)

    # Find the nodule on SOP UID 1.2.3.4.5.010
    for nodule in nodules:
        if "1.2.3.4.5.010" in nodule.slices:
            ns = nodule.slices["1.2.3.4.5.010"]
            assert ns.bbox.x == 300
            assert ns.bbox.y == 300
            assert ns.bbox.width == 21
            assert ns.bbox.height == 21


def test_parse_annotation_xml_returns_empty_for_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.xml"
    nodules = parse_annotation_xml(missing)
    assert nodules == []


def test_parse_annotation_xml_returns_empty_for_malformed_xml(tmp_path: Path) -> None:
    bad_xml = tmp_path / "bad.xml"
    bad_xml.write_text("this is not xml", encoding="utf-8")
    nodules = parse_annotation_xml(bad_xml)
    assert nodules == []


def test_find_annotation_for_patient_finds_annotations_xml(tmp_path: Path) -> None:
    patient_dir = tmp_path / "LIDC-IDRI-0078"
    patient_dir.mkdir()
    ann_file = patient_dir / "annotations.xml"
    ann_file.write_text("<xml/>", encoding="utf-8")

    result = find_annotation_for_patient(tmp_path, "LIDC-IDRI-0078")
    assert result == ann_file


def test_find_annotation_for_patient_finds_numbered_xml(tmp_path: Path) -> None:
    ann_dir = tmp_path / "annotations"
    ann_dir.mkdir()
    ann_file = ann_dir / "078.xml"
    ann_file.write_text("<xml/>", encoding="utf-8")

    result = find_annotation_for_patient(tmp_path, "LIDC-IDRI-0078")
    assert result == ann_file


def test_find_annotation_for_patient_returns_none_when_missing(tmp_path: Path) -> None:
    result = find_annotation_for_patient(tmp_path, "LIDC-IDRI-9999")
    assert result is None


def test_get_ground_truth_for_slice_returns_matching_entries(sample_xml: Path) -> None:
    nodules = parse_annotation_xml(sample_xml)
    matches = get_ground_truth_for_slice(nodules, "1.2.3.4.5.001")
    assert len(matches) >= 1
    assert all(isinstance(m[1], NoduleSlice) for m in matches)


def test_get_ground_truth_for_slice_returns_empty_for_unknown_sop() -> None:
    nodule = NoduleAnnotation(nodule_id="test")
    nodule.slices["known_sop"] = NoduleSlice(
        sop_uid="known_sop",
        z_position=0.0,
        contour_points=[(10, 10)],
        bbox=BBox(x=10, y=10, width=1, height=1),
    )
    matches = get_ground_truth_for_slice([nodule], "unknown_sop")
    assert matches == []


def test_nodule_annotation_centroid_and_overall_bbox() -> None:
    nodule = NoduleAnnotation(nodule_id="test")
    nodule.slices["sop1"] = NoduleSlice(
        sop_uid="sop1",
        z_position=0.0,
        contour_points=[(10, 20), (30, 40)],
        bbox=BBox(x=10, y=20, width=21, height=21),
    )
    assert nodule.centroid_x == 20.0
    assert nodule.centroid_y == 30.0

    overall = nodule.overall_bbox
    assert overall is not None
    assert overall.x == 10
    assert overall.y == 20
