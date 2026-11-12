"""
Unit and Integration Tests for Stage 2.6: Engine-Bay Dataset Validation and Quality Audit.
Validates the exported Roboflow dataset for Class 0 ('engine_bay'), image-label pairing,
normalized coordinates, polygon-to-bbox compatibility, cardinality, and visual reviews.
"""

import zipfile
from pathlib import Path
import pytest

ZIP_PATH = Path("D:/AI-Vehicle-Diagnosis-System/data/raw/stage2_6_engine_bay/Vehicle Component Detection.v1i.yolov8.zip")
ORIG_IMG_DIR = Path("D:/AI-Vehicle-Diagnosis-System/data/raw/stage2_collection/engine_bay/images")
VISUAL_REVIEW_DIR = Path("D:/AI-Vehicle-Diagnosis-System/docs/stage2_6_engine_bay_visual_review")
DIST_CSV_PATH = Path("D:/AI-Vehicle-Diagnosis-System/docs/stage2_6_engine_bay_class_distribution.csv")


@pytest.fixture(scope="module")
def exported_zip():
    assert ZIP_PATH.exists(), f"Exported archive missing: {ZIP_PATH}"
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        yield zf


def test_zip_archive_and_data_yaml(exported_zip):
    """Verify data.yaml defines engine_bay as Class ID 0."""
    content = exported_zip.read("data.yaml").decode("utf-8")
    assert "nc: 1" in content
    assert "names: ['engine_bay']" in content
    assert "vehicle-component-detection" in content.lower()


def test_exported_image_and_label_counts(exported_zip):
    """Verify exact split counts in the export."""
    images = [n for n in exported_zip.namelist() if n.lower().endswith(('.jpg', '.jpeg', '.png'))]
    labels = [n for n in exported_zip.namelist() if n.lower().endswith('.txt') and not n.endswith('.dataset.txt') and not n.endswith('.roboflow.txt')]
    
    assert len(images) == 192
    assert len(labels) == 192
    
    train_imgs = [n for n in images if n.startswith("train/images/")]
    valid_imgs = [n for n in images if n.startswith("valid/images/")]
    test_imgs = [n for n in images if n.startswith("test/images/")]
    
    assert len(train_imgs) == 135
    assert len(valid_imgs) == 38
    assert len(test_imgs) == 19


def test_image_label_pairing(exported_zip):
    """Verify 100% 1-to-1 pairing between images and labels in export."""
    images = [n for n in exported_zip.namelist() if n.lower().endswith(('.jpg', '.jpeg', '.png'))]
    labels = [n for n in exported_zip.namelist() if n.lower().endswith('.txt') and not n.endswith('.dataset.txt') and not n.endswith('.roboflow.txt')]
    
    img_stems = {Path(p).stem for p in images}
    lbl_stems = {Path(p).stem for p in labels}
    
    assert img_stems == lbl_stems
    assert len(img_stems) == 192


def test_annotation_cardinality_and_empty_file(exported_zip):
    """Verify 191 images have exactly 1 annotation and 1 image is empty (negative sample)."""
    labels = [n for n in exported_zip.namelist() if n.lower().endswith('.txt') and not n.endswith('.dataset.txt') and not n.endswith('.roboflow.txt')]
    
    empty_files = []
    single_annot_files = []
    multi_annot_files = []
    
    for lf in labels:
        lines = exported_zip.read(lf).decode("utf-8").strip().splitlines()
        if len(lines) == 0:
            empty_files.append(lf)
        elif len(lines) == 1:
            single_annot_files.append(lf)
        else:
            multi_annot_files.append(lf)
            
    assert len(empty_files) == 1
    assert "Ford-Ranger-XLT-4x4-3-2-Autoentusiastas-22_jpg" in empty_files[0]
    assert len(single_annot_files) == 191
    assert len(multi_annot_files) == 0


def test_annotation_geometry_and_coordinate_bounds(exported_zip):
    """Verify that all annotations use Class 0 and coordinates are normalized within [0, 1]."""
    labels = [n for n in exported_zip.namelist() if n.lower().endswith('.txt') and not n.endswith('.dataset.txt') and not n.endswith('.roboflow.txt')]
    
    bbox_count = 0
    polygon_count = 0
    
    for lf in labels:
        lines = exported_zip.read(lf).decode("utf-8").strip().splitlines()
        for l in lines:
            toks = l.strip().split()
            assert toks[0] == "0", f"Expected class '0', got {toks[0]} in {lf}"
            coords = [float(x) for x in toks[1:]]
            
            # Check coordinate bounds
            for c in coords:
                assert 0.0 <= c <= 1.0, f"Coordinate out of bounds: {c} in {lf}"
                
            if len(toks) == 5:
                bbox_count += 1
            else:
                assert len(toks) > 5 and len(toks) % 2 == 1, f"Invalid polygon token count in {lf}"
                polygon_count += 1
                
    assert bbox_count == 6
    assert polygon_count == 185


def test_comparison_against_original_193():
    """Verify account of original 193 images vs 192 exported images."""
    assert ORIG_IMG_DIR.exists()
    orig_files = list(ORIG_IMG_DIR.glob("*.jpg")) + list(ORIG_IMG_DIR.glob("*.jpeg")) + list(ORIG_IMG_DIR.glob("*.png"))
    assert len(orig_files) == 193
    
    # Confirm the 1 unexported image is BYD-Dolphin-Mini
    byd_dolphin = [p for p in orig_files if "BYD-Dolphin-Mini" in p.name]
    assert len(byd_dolphin) == 1


def test_visual_review_deliverables():
    """Verify at least 30 visual inspection review samples exist."""
    assert VISUAL_REVIEW_DIR.exists()
    samples = list(VISUAL_REVIEW_DIR.glob("*.jpg"))
    assert len(samples) >= 30, f"Expected at least 30 visual reviews, found {len(samples)}"


def test_distribution_csv_exists():
    """Verify class distribution CSV exists and has expected header and rows."""
    assert DIST_CSV_PATH.exists()
    lines = DIST_CSV_PATH.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 5
    assert "split,total_images" in lines[0]
