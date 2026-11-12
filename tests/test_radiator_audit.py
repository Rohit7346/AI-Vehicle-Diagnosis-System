"""
Unit and Integration Tests for Stage 2.7: Radiator Dataset Audit and Quality Analysis.
Validates source dataset archives, ontology mapping, class ID resolution, split distributions,
image-label pairing, YOLO bounding box geometry, duplicate/leakage detection, visual review samples,
and class distribution CSV integrity.
"""

import zipfile
import csv
from pathlib import Path
import pytest

ZIP_DIR = Path("D:/AI-Vehicle-Diagnosis-System/data/raw/stage2_7_radiator")
ZIP1_PATH = ZIP_DIR / "Engine Parts Detector.v1i.yolov8.zip"
ZIP2_PATH = ZIP_DIR / "engine-bay-parts.v1i.yolov8.zip"
DIST_CSV_PATH = Path("D:/AI-Vehicle-Diagnosis-System/docs/stage2_7_radiator_class_distribution.csv")
VISUAL_REVIEW_DIR = Path("D:/AI-Vehicle-Diagnosis-System/docs/stage2_7_radiator_visual_review")


@pytest.fixture(scope="module")
def source1_zip():
    assert ZIP1_PATH.exists(), f"Source 1 archive missing: {ZIP1_PATH}"
    with zipfile.ZipFile(ZIP1_PATH, "r") as zf:
        yield zf


@pytest.fixture(scope="module")
def source2_zip():
    assert ZIP2_PATH.exists(), f"Source 2 archive missing: {ZIP2_PATH}"
    with zipfile.ZipFile(ZIP2_PATH, "r") as zf:
        yield zf


def test_zip_archives_exist_and_readable(source1_zip, source2_zip):
    """Verify both source zip archives are readable and contain expected files."""
    assert len(source1_zip.namelist()) > 0
    assert len(source2_zip.namelist()) > 0


def test_class_ontology_and_radiator_id(source1_zip, source2_zip):
    """
    Verify class ontology and radiator class ID mapping:
    - Source 1 (Engine Parts Detector): class 4 = 'Radiator'
    - Source 2 (engine-bay-parts): class 9 = 'Radiator', 10 = 'Radiator Cap', 3 = 'Engine Coolant Reservoir'
    - Target Prototype Class ID = 3 ('radiator')
    """
    s1_yaml = source1_zip.read("data.yaml").decode("utf-8")
    assert "Radiator" in s1_yaml
    # Check lines of names in Source 1
    # names: ['Air Coolant', 'Battery', 'Fuse-Box', 'Main Engine', 'Radiator', 'Windscreen Washer']
    assert "'Radiator'" in s1_yaml

    s2_yaml = source2_zip.read("data.yaml").decode("utf-8")
    assert "Radiator" in s2_yaml
    assert "Radiator Cap" in s2_yaml
    assert "Engine Coolant Reservoir" in s2_yaml


def test_source1_split_counts_and_cardinality(source1_zip):
    """Verify exact image, label, and radiator bounding box counts in Source 1."""
    imgs = [n for n in source1_zip.namelist() if n.lower().endswith(('.jpg', '.jpeg', '.png'))]
    assert len(imgs) == 400

    train_imgs = [n for n in imgs if n.startswith("train/images/")]
    valid_imgs = [n for n in imgs if n.startswith("valid/images/")]
    test_imgs = [n for n in imgs if n.startswith("test/images/")]

    assert len(train_imgs) == 265
    assert len(valid_imgs) == 86
    assert len(test_imgs) == 49

    rad_imgs = 0
    rad_boxes = 0
    multi_box_count = 0

    for ip in imgs:
        lp = ip.replace('/images/', '/labels/').rsplit('.', 1)[0] + '.txt'
        if lp in source1_zip.namelist():
            raw = source1_zip.read(lp).decode("utf-8").strip()
            boxes = [line for line in raw.splitlines() if line.strip().startswith("4 ")]
            if len(boxes) > 0:
                rad_imgs += 1
                rad_boxes += len(boxes)
                if len(boxes) > 1:
                    multi_box_count += 1

    assert rad_imgs == 260
    assert rad_boxes == 269
    assert multi_box_count == 9


def test_source2_split_counts_and_cardinality(source2_zip):
    """Verify exact image, label, and radiator bounding box counts in Source 2."""
    imgs = [n for n in source2_zip.namelist() if n.lower().endswith(('.jpg', '.jpeg', '.png'))]
    assert len(imgs) == 200

    # Source 2 has all images in train/
    train_imgs = [n for n in imgs if n.startswith("train/images/")]
    assert len(train_imgs) == 200

    rad_imgs = 0
    rad_boxes = 0
    multi_box_count = 0

    for ip in imgs:
        lp = ip.replace('/images/', '/labels/').rsplit('.', 1)[0] + '.txt'
        if lp in source2_zip.namelist():
            raw = source2_zip.read(lp).decode("utf-8").strip()
            boxes = [line for line in raw.splitlines() if line.strip().startswith("9 ")]
            if len(boxes) > 0:
                rad_imgs += 1
                rad_boxes += len(boxes)
                if len(boxes) > 1:
                    multi_box_count += 1

    assert rad_imgs == 118
    assert rad_boxes == 119
    assert multi_box_count == 1  # 819_jpg collage


def test_image_label_pairing(source1_zip, source2_zip):
    """Verify 100% 1-to-1 pairing between images and labels in both sources."""
    for zf, name in [(source1_zip, "Source 1"), (source2_zip, "Source 2")]:
        imgs = [n for n in zf.namelist() if n.lower().endswith(('.jpg', '.jpeg', '.png'))]
        for ip in imgs:
            lp = ip.replace('/images/', '/labels/').rsplit('.', 1)[0] + '.txt'
            assert lp in zf.namelist(), f"{name}: Missing label file for image {ip}"


def test_normalized_coordinates_validity(source1_zip, source2_zip):
    """Verify that all radiator bounding box coordinates are within [0, 1]."""
    for zf, rad_id in [(source1_zip, 4), (source2_zip, 9)]:
        labels = [n for n in zf.namelist() if n.endswith('.txt') and not n.endswith('.dataset.txt') and not n.endswith('.roboflow.txt')]
        for lp in labels:
            raw = zf.read(lp).decode("utf-8").strip()
            for line in raw.splitlines():
                toks = line.strip().split()
                if len(toks) == 5 and toks[0] == str(rad_id):
                    xc, yc, w, h = map(float, toks[1:])
                    assert 0.0 <= xc <= 1.0, f"xc out of bounds: {xc} in {lp}"
                    assert 0.0 <= yc <= 1.0, f"yc out of bounds: {yc} in {lp}"
                    assert 0.0 < w <= 1.0, f"w out of bounds: {w} in {lp}"
                    assert 0.0 < h <= 1.0, f"h out of bounds: {h} in {lp}"


def test_radiator_geometry_discrepancy(source1_zip, source2_zip):
    """
    Validate the core Stage 2.7 Task 12 discovery:
    - Source 1 annotates radiator caps/fillers (mean area ~0.007, aspect ratio ~0.98, >50% tiny boxes).
    - Source 2 annotates the wide radiator core/support (mean area ~0.045, aspect ratio ~8.34, <1% tiny boxes).
    """
    s1_areas, s1_aspects = [], []
    for lp in [n for n in source1_zip.namelist() if n.endswith('.txt') and not n.endswith('.dataset.txt') and not n.endswith('.roboflow.txt')]:
        for line in source1_zip.read(lp).decode("utf-8").strip().splitlines():
            toks = line.strip().split()
            if len(toks) == 5 and toks[0] == "4":
                _, _, w, h = map(float, toks[1:])
                s1_areas.append(w * h)
                s1_aspects.append(w / h)

    s2_areas, s2_aspects = [], []
    for lp in [n for n in source2_zip.namelist() if n.endswith('.txt') and not n.endswith('.dataset.txt') and not n.endswith('.roboflow.txt')]:
        for line in source2_zip.read(lp).decode("utf-8").strip().splitlines():
            toks = line.strip().split()
            if len(toks) == 5 and toks[0] == "9":
                _, _, w, h = map(float, toks[1:])
                s2_areas.append(w * h)
                s2_aspects.append(w / h)

    # Source 1 tiny boxes (< 0.005) exceeds 50%
    tiny_s1 = sum(1 for a in s1_areas if a < 0.005)
    assert tiny_s1 == 139
    assert tiny_s1 / len(s1_areas) > 0.50
    assert sum(s1_aspects) / len(s1_aspects) < 1.5  # near 1.0

    # Source 2 tiny boxes (< 0.005) is <= 1 box
    tiny_s2 = sum(1 for a in s2_areas if a < 0.005)
    assert tiny_s2 == 1
    assert sum(s2_aspects) / len(s2_aspects) > 6.0  # wide aspect ratio ~8.34


def test_visual_review_samples_generated():
    """Verify at least 30 rendered JPEG visual review samples exist in docs."""
    assert VISUAL_REVIEW_DIR.exists(), f"Visual review directory missing: {VISUAL_REVIEW_DIR}"
    samples = list(VISUAL_REVIEW_DIR.glob("*.jpg"))
    assert len(samples) >= 30
    
    src1_samples = [s for s in samples if s.name.startswith("src1_epd_")]
    src2_samples = [s for s in samples if s.name.startswith("src2_ebp_")]
    assert len(src1_samples) >= 15
    assert len(src2_samples) >= 15


def test_distribution_csv_schema_and_integrity():
    """Verify the radiator class distribution CSV exists and matches audit data."""
    assert DIST_CSV_PATH.exists(), f"Distribution CSV missing: {DIST_CSV_PATH}"
    with open(DIST_CSV_PATH, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    assert len(reader) == 9  # 4 for S1 (train/valid/test/ALL) + 4 for S2 + 1 COMBINED
    
    # Check Source 1 ALL_SPLITS row
    s1_all = next(r for r in reader if r["source_dataset_zip"] == "Engine Parts Detector.v1i.yolov8.zip" and r["split"] == "ALL_SPLITS")
    assert s1_all["total_images"] == "400"
    assert s1_all["radiator_images"] == "260"
    assert s1_all["radiator_bboxes"] == "269"
    assert s1_all["tiny_boxes_lt_0005"] == "139"
    assert s1_all["cross_split_leakage_pairs"] == "13"
    assert s1_all["exact_duplicates"] == "1"

    # Check Source 2 ALL_SPLITS row
    s2_all = next(r for r in reader if r["source_dataset_zip"] == "engine-bay-parts.v1i.yolov8.zip" and r["split"] == "ALL_SPLITS")
    assert s2_all["total_images"] == "200"
    assert s2_all["radiator_images"] == "118"
    assert s2_all["radiator_bboxes"] == "119"
    assert s2_all["tiny_boxes_lt_0005"] == "1"
    assert s2_all["multi_box_images"] == "1"

    # Check COMBINED row
    comb = next(r for r in reader if r["source_dataset_zip"] == "COMBINED_SOURCES")
    assert comb["total_images"] == "600"
    assert comb["radiator_images"] == "378"
    assert comb["radiator_bboxes"] == "388"
