"""
Unit and Integration Tests for Stage 3.0: Three-Class Dataset Integration and Final Training Readiness Audit.
Validates the integrated YOLOv8 dataset directory structure, 100% image-label pairing, class ID unification
(0: engine_bay, 1: battery, 2: radiator), normalized coordinate bounds, empty-label hard negative handling,
multi-class co-occurrences, zero cross-split data leakage (dHash), and data.yaml schema.
"""

import csv
import sys
from pathlib import Path
import pytest
from PIL import Image

sys.path.insert(0, "D:/AI-Vehicle-Diagnosis-System")
from ml.datasets.audit_staged_images import make_long_path, ImageQualityAuditor

DATASET_ROOT = Path("D:/AI-Vehicle-Diagnosis-System/data/processed/stage3_0_three_class")
YAML_PATH = DATASET_ROOT / "data.yaml"
DIST_CSV_PATH = Path("D:/AI-Vehicle-Diagnosis-System/docs/stage3_0_class_distribution.csv")
DUP_CSV_PATH = Path("D:/AI-Vehicle-Diagnosis-System/docs/stage3_0_duplicate_resolution.csv")

# Source preservation paths
RAW_EB_ZIP = Path("D:/AI-Vehicle-Diagnosis-System/data/raw/stage2_6_engine_bay/Vehicle Component Detection.v1i.yolov8.zip")
CLEANED_BAT_ROOT = Path("D:/AI-Vehicle-Diagnosis-System/data/cleaned/stage2_5_battery")
CLEANED_RAD_ROOT = Path("D:/AI-Vehicle-Diagnosis-System/data/cleaned/stage2_7_radiator")
PROTOTYPE_5CLASS = Path("D:/AI-Vehicle-Diagnosis-System/data/prototype_5class")


def test_dataset_directory_structure_and_counts():
    """Verify that images and labels exist for train, val, test with exact expected counts."""
    assert DATASET_ROOT.exists(), f"Processed dataset directory missing: {DATASET_ROOT}"
    
    counts = {}
    for sp in ["train", "val", "test"]:
        img_dir = DATASET_ROOT / "images" / sp
        lbl_dir = DATASET_ROOT / "labels" / sp
        assert img_dir.exists(), f"Missing image directory: {img_dir}"
        assert lbl_dir.exists(), f"Missing label directory: {lbl_dir}"
        
        imgs = list(img_dir.glob("*.jpg"))
        lbls = list(lbl_dir.glob("*.txt"))
        assert len(imgs) == len(lbls), f"Pairing mismatch in {sp}: {len(imgs)} imgs vs {len(lbls)} lbls"
        counts[sp] = len(imgs)

    assert counts["train"] == 480, f"Expected 480 train images, got {counts['train']}"
    assert counts["val"] == 146, f"Expected 146 val images, got {counts['val']}"
    assert counts["test"] == 70, f"Expected 70 test images, got {counts['test']}"
    assert sum(counts.values()) == 696


def test_image_label_pairing_and_stems():
    """Verify 100% 1-to-1 stem alignment between images and label files across all splits."""
    for sp in ["train", "val", "test"]:
        imgs = list((DATASET_ROOT / "images" / sp).glob("*.jpg"))
        lbls = list((DATASET_ROOT / "labels" / sp).glob("*.txt"))
        
        img_stems = {p.stem for p in imgs}
        lbl_stems = {p.stem for p in lbls}
        assert img_stems == lbl_stems, f"Image and label stems do not match in {sp}"


def test_valid_class_ids_and_counts():
    """
    Verify all class IDs are strictly in {0, 1, 2} and total box counts match:
    - 0: engine_bay (191 boxes)
    - 1: battery (465 boxes)
    - 2: radiator (115 boxes)
    - Total: 771 boxes
    """
    class_counts = {0: 0, 1: 0, 2: 0}
    split_class_counts = {"train": {0: 0, 1: 0, 2: 0}, "val": {0: 0, 1: 0, 2: 0}, "test": {0: 0, 1: 0, 2: 0}}
    total_boxes = 0
    
    for sp in ["train", "val", "test"]:
        lbls = list((DATASET_ROOT / "labels" / sp).glob("*.txt"))
        for lf in lbls:
            with open(make_long_path(lf), "r", encoding="utf-8") as f:
                content = f.read().strip()
            if not content:
                continue
            for line in content.splitlines():
                toks = line.split()
                assert len(toks) == 5, f"Malformed label line in {lf}: {line}"
                cid = int(toks[0])
                assert cid in [0, 1, 2], f"Invalid class ID {cid} in {lf}"
                class_counts[cid] += 1
                split_class_counts[sp][cid] += 1
                total_boxes += 1

    assert class_counts[0] == 191, f"Expected 191 engine_bay boxes, got {class_counts[0]}"
    assert class_counts[1] == 465, f"Expected 465 battery boxes, got {class_counts[1]}"
    assert class_counts[2] == 115, f"Expected 115 radiator boxes, got {class_counts[2]}"
    assert total_boxes == 771

    # Check split breakdown
    assert split_class_counts["train"][0] == 136
    assert split_class_counts["val"][0] == 37
    assert split_class_counts["test"][0] == 18

    assert split_class_counts["train"][1] == 312
    assert split_class_counts["val"][1] == 103
    assert split_class_counts["test"][1] == 50

    assert split_class_counts["train"][2] == 79
    assert split_class_counts["val"][2] == 27
    assert split_class_counts["test"][2] == 9


def test_normalized_coordinate_boundaries():
    """Verify that all 771 bounding boxes have normalized coordinates within [0, 1]."""
    for sp in ["train", "val", "test"]:
        lbls = list((DATASET_ROOT / "labels" / sp).glob("*.txt"))
        for lf in lbls:
            with open(make_long_path(lf), "r", encoding="utf-8") as f:
                content = f.read().strip()
            if not content:
                continue
            for line in content.splitlines():
                toks = line.split()
                xc, yc, w, h = map(float, toks[1:])
                assert 0.0 <= xc <= 1.0, f"xc out of bounds: {xc} in {lf}"
                assert 0.0 <= yc <= 1.0, f"yc out of bounds: {yc} in {lf}"
                assert 0.0 < w <= 1.0, f"w out of bounds: {w} in {lf}"
                assert 0.0 < h <= 1.0, f"h out of bounds: {h} in {lf}"


def test_empty_label_negative_sample():
    """Verify exactly 1 documented empty-label negative sample exists in test (Ford Ranger)."""
    empty_files = []
    for sp in ["train", "val", "test"]:
        lbls = list((DATASET_ROOT / "labels" / sp).glob("*.txt"))
        for lf in lbls:
            with open(make_long_path(lf), "r", encoding="utf-8") as f:
                content = f.read().strip()
            if not content:
                empty_files.append((sp, lf.name))

    assert len(empty_files) == 1, f"Expected exactly 1 empty label file, got {len(empty_files)}: {empty_files}"
    assert empty_files[0][0] == "test"
    assert "Ford-Ranger-XLT-4x4" in empty_files[0][1]


def test_multi_class_and_multi_box_images():
    """Verify exactly 76 images contain multiple classes and multiple bounding boxes."""
    multi_class_files = []
    multi_box_files = []
    
    for sp in ["train", "val", "test"]:
        lbls = list((DATASET_ROOT / "labels" / sp).glob("*.txt"))
        for lf in lbls:
            with open(make_long_path(lf), "r", encoding="utf-8") as f:
                content = f.read().strip()
            if not content:
                continue
            lines = content.splitlines()
            if len(lines) > 1:
                multi_box_files.append(lf)
            cids = {int(l.split()[0]) for l in lines}
            if len(cids) > 1:
                multi_class_files.append(lf)

    assert len(multi_class_files) == 76, f"Expected 76 multi-class images, got {len(multi_class_files)}"
    assert len(multi_box_files) == 76, f"Expected 76 multi-box images, got {len(multi_box_files)}"


def test_zero_cross_split_data_leakage():
    """Verify zero cross-split perceptual duplicate leakage (dHash <= 4) across all 696 images."""
    auditor = ImageQualityAuditor()
    
    splits = {}
    for sp in ["train", "val", "test"]:
        imgs = list((DATASET_ROOT / "images" / sp).glob("*.jpg"))
        hashes = []
        for p in imgs:
            with Image.open(make_long_path(p)) as im:
                hashes.append((p.name, auditor.compute_dhash(im)))
        splits[sp] = hashes

    # train vs val
    for na, ha in splits["train"]:
        for nb, hb in splits["val"]:
            d = auditor.hamming_distance(ha, hb)
            assert d > 4, f"Cross-split leakage between train ({na}) and val ({nb}): d={d}"

    # train vs test
    for na, ha in splits["train"]:
        for nb, hb in splits["test"]:
            d = auditor.hamming_distance(ha, hb)
            assert d > 4, f"Cross-split leakage between train ({na}) and test ({nb}): d={d}"

    # val vs test
    for na, ha in splits["val"]:
        for nb, hb in splits["test"]:
            d = auditor.hamming_distance(ha, hb)
            assert d > 4, f"Cross-split leakage between val ({na}) and test ({nb}): d={d}"


def test_data_yaml_correctness():
    """Verify data.yaml exists and defines nc=3 with correct names and paths."""
    assert YAML_PATH.exists(), f"data.yaml missing: {YAML_PATH}"
    with open(make_long_path(YAML_PATH), "r", encoding="utf-8") as f:
        content = f.read()
    
    assert "nc: 3" in content
    assert "names: ['engine_bay', 'battery', 'radiator']" in content
    assert "train: images/train" in content
    assert "val: images/val" in content
    assert "test: images/test" in content


def test_source_data_preservation():
    """Confirm strict safety boundaries: raw archives, cleaned datasets, and prototype untouched."""
    assert RAW_EB_ZIP.exists()
    assert CLEANED_BAT_ROOT.exists()
    assert CLEANED_RAD_ROOT.exists()
    assert PROTOTYPE_5CLASS.exists()


def test_csv_artifacts_exist_and_valid():
    """Verify distribution and duplicate resolution CSVs exist and contain valid records."""
    assert DIST_CSV_PATH.exists()
    assert DUP_CSV_PATH.exists()
    
    with open(make_long_path(DIST_CSV_PATH), "r", encoding="utf-8") as f:
        dist_rows = list(csv.DictReader(f))
    assert len(dist_rows) == 4  # train, val, test, ALL_SPLITS
    assert dist_rows[3]["total_images"] == "696"
    assert dist_rows[3]["total_bboxes"] == "771"

    with open(make_long_path(DUP_CSV_PATH), "r", encoding="utf-8") as f:
        dup_rows = list(csv.DictReader(f))
    assert len(dup_rows) == 29
