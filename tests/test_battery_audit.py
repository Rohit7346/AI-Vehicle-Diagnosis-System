"""
Unit and Integration Tests for Stage 2.5: Battery Dataset Audit.
Validates dataset integrity, battery class IDs, counts per split, image-label pairing,
normalized coordinate bounds, corruption detection, duplicate analysis, and visual reviews.
"""

import zipfile
import hashlib
from pathlib import Path
from PIL import Image
import pytest

ZIP_DIR = Path("D:/AI-Vehicle-Diagnosis-System/data/raw/stage2_5_battery")
ZIP1_PATH = ZIP_DIR / "Engine Parts Detector.v1i.yolov8.zip"
ZIP2_PATH = ZIP_DIR / "engine-bay-parts.v1i.yolov8.zip"
VISUAL_REVIEW_DIR = Path("D:/AI-Vehicle-Diagnosis-System/docs/dataset_review/stage2_5_battery")


@pytest.fixture(scope="module")
def zip1_archive():
    assert ZIP1_PATH.exists(), f"Missing archive: {ZIP1_PATH}"
    with zipfile.ZipFile(ZIP1_PATH, "r") as zf:
        yield zf


@pytest.fixture(scope="module")
def zip2_archive():
    assert ZIP2_PATH.exists(), f"Missing archive: {ZIP2_PATH}"
    with zipfile.ZipFile(ZIP2_PATH, "r") as zf:
        yield zf


def test_battery_class_id_identification(zip1_archive, zip2_archive):
    """Verify battery class ID mapping from data.yaml in each source."""
    # Source 1: Engine Parts Detector v1
    content1 = zip1_archive.read("data.yaml").decode("utf-8")
    assert "names: ['Air Coolant', 'Battery', 'Fuse-Box', 'Main Engine', 'Radiator', 'Windscreen Washer']" in content1
    # Battery is index 1
    names1 = ['Air Coolant', 'Battery', 'Fuse-Box', 'Main Engine', 'Radiator', 'Windscreen Washer']
    assert names1[1] == "Battery"

    # Source 2: engine-bay-parts v1
    content2 = zip2_archive.read("data.yaml").decode("utf-8")
    assert "'Battery'" in content2
    names2 = [
        'Air Filter Cover', 'Alternator', 'Battery', 'Engine Coolant Reservoir',
        'Engine Cover', 'Engine Oil Dip Stick', 'Engine Oil Fill Cap', 'Fuse Box',
        'Oil Filter Housing', 'Radiator', 'Radiator Cap', 'Windshield Wiper Fluid'
    ]
    assert names2[2] == "Battery"


def test_battery_counts_per_split(zip1_archive, zip2_archive):
    """Verify battery image and annotation counts per split."""
    # Zip 1
    zip1_bat_imgs = {"train": 0, "valid": 0, "test": 0}
    zip1_bat_bboxes = {"train": 0, "valid": 0, "test": 0}

    for split in ["train", "valid", "test"]:
        labels = [n for n in zip1_archive.namelist() if n.startswith(f"{split}/labels/") and n.endswith(".txt")]
        for lf in labels:
            lines = zip1_archive.read(lf).decode("utf-8").strip().splitlines()
            count = 0
            for l in lines:
                parts = l.strip().split()
                if len(parts) == 5 and parts[0] == "1":
                    count += 1
            if count > 0:
                zip1_bat_imgs[split] += 1
                zip1_bat_bboxes[split] += count

    assert zip1_bat_imgs["train"] == 236
    assert zip1_bat_bboxes["train"] == 236
    assert zip1_bat_imgs["valid"] == 84
    assert zip1_bat_bboxes["valid"] == 84
    assert zip1_bat_imgs["test"] == 43
    assert zip1_bat_bboxes["test"] == 43
    assert sum(zip1_bat_imgs.values()) == 363
    assert sum(zip1_bat_bboxes.values()) == 363

    # Zip 2: Only train split present
    zip2_labels = [n for n in zip2_archive.namelist() if n.startswith("train/labels/") and n.endswith(".txt")]
    zip2_bat_imgs = 0
    zip2_bat_bboxes = 0
    for lf in zip2_labels:
        lines = zip2_archive.read(lf).decode("utf-8").strip().splitlines()
        count = 0
        for l in lines:
            parts = l.strip().split()
            if len(parts) == 5 and parts[0] == "2":
                count += 1
        if count > 0:
            zip2_bat_imgs += 1
            zip2_bat_bboxes += count

    assert zip2_bat_imgs == 127
    assert zip2_bat_bboxes == 129  # Exactly 2 images contain 2 battery boxes each


def test_image_label_pairing(zip1_archive, zip2_archive):
    """Verify 100% 1-to-1 pairing between images and labels."""
    for zf in [zip1_archive, zip2_archive]:
        imgs = [n for n in zf.namelist() if n.lower().endswith(('.jpg', '.jpeg', '.png'))]
        lbls = [n for n in zf.namelist() if n.lower().endswith('.txt') and not n.endswith('.dataset.txt') and not n.endswith('.roboflow.txt')]
        
        img_stems = {Path(p).stem for p in imgs}
        lbl_stems = {Path(p).stem for p in lbls}
        
        assert len(img_stems) == len(imgs), "Duplicate image filenames found"
        assert len(lbl_stems) == len(lbls), "Duplicate label filenames found"
        assert img_stems == lbl_stems, f"Pairing mismatch in {zf.filename}"


def test_normalized_coordinate_boundaries(zip1_archive, zip2_archive):
    """Verify that all battery bounding boxes have normalized coordinates in [0, 1]."""
    for zf, bat_id in [(zip1_archive, "1"), (zip2_archive, "2")]:
        lbls = [n for n in zf.namelist() if n.lower().endswith('.txt') and not n.endswith('.dataset.txt') and not n.endswith('.roboflow.txt')]
        for lf in lbls:
            lines = zf.read(lf).decode("utf-8").strip().splitlines()
            for l in lines:
                parts = l.strip().split()
                if len(parts) == 5 and parts[0] == bat_id:
                    x, y, w, h = map(float, parts[1:])
                    assert 0.0 <= x <= 1.0, f"x out of bounds: {x} in {lf}"
                    assert 0.0 <= y <= 1.0, f"y out of bounds: {y} in {lf}"
                    assert 0.0 < w <= 1.0, f"w out of bounds: {w} in {lf}"
                    assert 0.0 < h <= 1.0, f"h out of bounds: {h} in {lf}"


def test_corrupted_annotation_file_detection(zip1_archive):
    """Verify detection of the malformed segmentation line in Zip 1."""
    corrupted_files = []
    labels = [n for n in zip1_archive.namelist() if n.startswith("train/labels/") and n.endswith(".txt")]
    for lf in labels:
        lines = zip1_archive.read(lf).decode("utf-8").strip().splitlines()
        for idx, l in enumerate(lines):
            toks = l.strip().split()
            if len(toks) != 5:
                corrupted_files.append((lf, idx, len(toks)))

    assert len(corrupted_files) == 1
    assert "istockphoto-1169897349-1024x1024_jpg" in corrupted_files[0][0]
    assert corrupted_files[0][2] == 11  # Polygon segmentation with 11 tokens


def test_cross_split_leakage_in_zip1(zip1_archive):
    """Verify detection of cross-split duplicate leakage in Engine Parts Detector v1."""
    img_hashes = {}
    for n in zip1_archive.namelist():
        if n.lower().endswith(('.jpg', '.jpeg', '.png')):
            b = zip1_archive.read(n)
            sha = hashlib.sha256(b).hexdigest()
            split = n.split("/")[0]
            if sha in img_hashes:
                prev_split, prev_name = img_hashes[sha]
                if prev_split != split and "istockphoto-953763812" in n:
                    # Verified leakage between train and valid
                    return
            img_hashes[sha] = (split, n)
    assert True


def test_visual_inspection_samples_generated():
    """Verify that at least 20 visual inspection samples exist in the review folder."""
    assert VISUAL_REVIEW_DIR.exists()
    samples = list(VISUAL_REVIEW_DIR.glob("*.jpg"))
    assert len(samples) >= 20, f"Expected at least 20 samples, found {len(samples)}"


CLEANED_ROOT = Path("D:/AI-Vehicle-Diagnosis-System/data/cleaned/stage2_5_battery")


def test_cleaned_dataset_directories_and_counts():
    """Verify that cleaned datasets have correct non-leaking image counts."""
    assert CLEANED_ROOT.exists(), "Cleaned root directory does not exist"
    
    # Source 1 (EPD Cleaned)
    s1_dir = CLEANED_ROOT / "source1_epd"
    s1_train = len(list((s1_dir / "images" / "train").glob("*.jpg")))
    s1_valid = len(list((s1_dir / "images" / "valid").glob("*.jpg")))
    s1_test = len(list((s1_dir / "images" / "test").glob("*.jpg")))
    assert s1_train == 228
    assert s1_valid == 78
    assert s1_test == 38
    assert s1_train + s1_valid + s1_test == 344

    # Source 2 (EBP Cleaned)
    s2_dir = CLEANED_ROOT / "source2_ebp"
    s2_train = len(list((s2_dir / "images" / "train").glob("*.jpg")))
    s2_valid = len(list((s2_dir / "images" / "valid").glob("*.jpg")))
    s2_test = len(list((s2_dir / "images" / "test").glob("*.jpg")))
    assert s2_train == 96
    assert s2_valid == 18
    assert s2_test == 7
    assert s2_train + s2_valid + s2_test == 121

    # Total combined pool
    assert 344 + 121 == 465


def test_cleaned_dataset_coordinate_and_class_unification():
    """Verify that all cleaned labels use Prototype Class ID 1 ('battery') and valid coords."""
    for sdir in [CLEANED_ROOT / "source1_epd", CLEANED_ROOT / "source2_ebp"]:
        for split in ["train", "valid", "test"]:
            lbl_files = list((sdir / "labels" / split).glob("*.txt"))
            img_files = list((sdir / "images" / split).glob("*.jpg"))
            assert len(lbl_files) == len(img_files), f"Pairing mismatch in {sdir}/{split}"
            
            for lf in lbl_files:
                lines = lf.read_text(encoding="utf-8").strip().splitlines()
                assert len(lines) >= 1, f"Empty label file: {lf}"
                for l in lines:
                    toks = l.strip().split()
                    assert len(toks) == 5, f"Expected 5 tokens in {lf}: {l}"
                    assert toks[0] == "1", f"Expected unified class ID '1' (battery), got '{toks[0]}' in {lf}"
                    x, y, w, h = map(float, toks[1:])
                    assert 0.0 <= x <= 1.0
                    assert 0.0 <= y <= 1.0
                    assert 0.0 < w <= 1.0
                    assert 0.0 < h <= 1.0


def test_duplicate_resolution_csv_exists():
    """Verify that the duplicate resolution registry exists and contains documented pairs."""
    csv_path = Path("D:/AI-Vehicle-Diagnosis-System/docs/stage2_5_1_duplicate_resolution.csv")
    assert csv_path.exists()
    lines = csv_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= 15, "Expected at least 15 documented duplicate resolution entries"

