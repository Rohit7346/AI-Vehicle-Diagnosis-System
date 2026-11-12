"""
Unit and Integration Tests for Stage 2.7.1: Radiator Dataset Cleaning and Semantic Remediation.
Validates cleaned radiator directory structure, image-label pairing, class ID unification (Prototype ID 3),
coordinate bounds, exclusion registries, duplicate resolution, reproducible group-aware splitting,
zero cross-split leakage, and visual review deliverables.
"""

import csv
from pathlib import Path
import pytest
from PIL import Image

CLEANED_ROOT = Path("D:/AI-Vehicle-Diagnosis-System/data/cleaned/stage2_7_radiator")
S1_DIR = CLEANED_ROOT / "source1_epd"
S2_DIR = CLEANED_ROOT / "source2_ebp"
AUDIT_DIR = CLEANED_ROOT / "audit"
DUP_CSV_PATH = Path("D:/AI-Vehicle-Diagnosis-System/docs/stage2_7_1_duplicate_resolution.csv")
VISUAL_REVIEW_DIR = Path("D:/AI-Vehicle-Diagnosis-System/docs/stage2_7_1_radiator_visual_review")


def test_cleaned_directory_structure_and_counts():
    """Verify cleaned Source 2 dataset structure, split counts, and total images."""
    assert S2_DIR.exists(), f"Cleaned Source 2 directory missing: {S2_DIR}"
    
    train_imgs = list((S2_DIR / "images" / "train").glob("*.jpg"))
    valid_imgs = list((S2_DIR / "images" / "valid").glob("*.jpg"))
    test_imgs = list((S2_DIR / "images" / "test").glob("*.jpg"))
    
    assert len(train_imgs) == 92, f"Expected 92 train images, got {len(train_imgs)}"
    assert len(valid_imgs) == 17, f"Expected 17 valid images, got {len(valid_imgs)}"
    assert len(test_imgs) == 6, f"Expected 6 test images, got {len(test_imgs)}"
    assert len(train_imgs) + len(valid_imgs) + len(test_imgs) == 115


def test_image_label_pairing_and_unified_class_id():
    """Verify 100% 1-to-1 pairing, valid YOLO coordinates, and unified Prototype Class ID 3."""
    for split in ["train", "valid", "test"]:
        imgs = list((S2_DIR / "images" / split).glob("*.jpg"))
        lbls = list((S2_DIR / "labels" / split).glob("*.txt"))
        
        assert len(imgs) == len(lbls), f"Mismatched count in {split}: {len(imgs)} imgs vs {len(lbls)} lbls"
        
        img_stems = {p.stem for p in imgs}
        lbl_stems = {p.stem for p in lbls}
        assert img_stems == lbl_stems, f"Stem mismatch in {split}"
        
        for lf in lbls:
            lines = lf.read_text(encoding="utf-8").strip().splitlines()
            assert len(lines) == 1, f"Expected exactly 1 radiator annotation per image in {lf}"
            toks = lines[0].split()
            assert len(toks) == 5, f"Malformed token count in {lf}"
            assert toks[0] == "3", f"Expected unified class ID '3' (radiator), got '{toks[0]}' in {lf}"
            
            xc, yc, w, h = map(float, toks[1:])
            assert 0.0 <= xc <= 1.0, f"xc out of bounds: {xc} in {lf}"
            assert 0.0 <= yc <= 1.0, f"yc out of bounds: {yc} in {lf}"
            assert 0.0 < w <= 1.0, f"w out of bounds: {w} in {lf}"
            assert 0.0 < h <= 1.0, f"h out of bounds: {h} in {lf}"


def test_cross_split_leakage_absence():
    """Verify zero cross-split near-duplicate leakage (dhash Hamming distance <= 4) in cleaned Source 2."""
    import sys
    sys.path.insert(0, "D:/AI-Vehicle-Diagnosis-System")
    from ml.datasets.audit_staged_images import ImageQualityAuditor
    auditor = ImageQualityAuditor()
    
    splits = {}
    for split in ["train", "valid", "test"]:
        imgs = list((S2_DIR / "images" / split).glob("*.jpg"))
        hashes = []
        for p in imgs:
            with Image.open(p) as im:
                hashes.append((p.name, auditor.compute_dhash(im)))
        splits[split] = hashes
        
    # Check train vs valid
    for na, ha in splits["train"]:
        for nb, hb in splits["valid"]:
            dist = auditor.hamming_distance(ha, hb)
            assert dist > 4, f"Data leakage between train ({na}) and valid ({nb}): dist={dist}"
            
    # Check train vs test
    for na, ha in splits["train"]:
        for nb, hb in splits["test"]:
            dist = auditor.hamming_distance(ha, hb)
            assert dist > 4, f"Data leakage between train ({na}) and test ({nb}): dist={dist}"
            
    # Check valid vs test
    for na, ha in splits["valid"]:
        for nb, hb in splits["test"]:
            dist = auditor.hamming_distance(ha, hb)
            assert dist > 4, f"Data leakage between valid ({na}) and test ({nb}): dist={dist}"


def test_source2_collage_and_duplicate_exclusions():
    """Verify collages and low-res duplicate are excluded from Source 2 and documented in audit."""
    cleaned_all = [p.name for p in S2_DIR.rglob("*.jpg")]
    assert not any("819_jpg" in n for n in cleaned_all), "Horizontal collage 819_jpg found in cleaned data"
    assert not any("602_jpg" in n for n in cleaned_all), "Vertical collage 602_jpg found in cleaned data"
    assert not any("886_jpg" in n for n in cleaned_all), "Internal duplicate 886_jpg found in cleaned data"
    
    s2_csv = AUDIT_DIR / "source2_exclusions.csv"
    assert s2_csv.exists(), f"Source 2 exclusions CSV missing: {s2_csv}"
    with open(s2_csv, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
    assert len(reader) == 3
    filenames = [r["filename"] for r in reader]
    assert any("819_jpg" in n for n in filenames)
    assert any("602_jpg" in n for n in filenames)
    assert any("886_jpg" in n for n in filenames)


def test_source1_semantic_exclusions_and_audit():
    """Verify all 269 Source 1 candidate boxes were excluded under the project definition."""
    s1_csv = AUDIT_DIR / "source1_exclusions.csv"
    assert s1_csv.exists(), f"Source 1 exclusions CSV missing: {s1_csv}"
    with open(s1_csv, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
    assert len(reader) == 269
    
    # Source 1 cleaned labels should contain 0 contaminated annotations
    s1_lbls = list(S1_DIR.rglob("*.txt"))
    assert len(s1_lbls) == 0, f"Expected 0 retained radiator labels in Source 1, found {len(s1_lbls)}"
    assert (S1_DIR / "README.md").exists()


def test_duplicate_resolution_registry():
    """Verify duplicate resolution CSV exists and documents 19 duplicate/leakage pairs."""
    assert DUP_CSV_PATH.exists(), f"Duplicate resolution CSV missing: {DUP_CSV_PATH}"
    with open(DUP_CSV_PATH, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
    assert len(reader) == 19
    
    pair_ids = [r["pair_id"] for r in reader]
    assert "DUP-EXACT-01" in pair_ids
    assert "DUP-NEAR-01" in pair_ids
    assert "DUP-NEAR-02" in pair_ids


def test_visual_review_samples_generated():
    """Verify 30 rendered JPEG visual review samples exist in docs."""
    assert VISUAL_REVIEW_DIR.exists(), f"Visual review directory missing: {VISUAL_REVIEW_DIR}"
    samples = list(VISUAL_REVIEW_DIR.glob("*.jpg"))
    assert len(samples) == 30
    
    retained = [s for s in samples if s.name.startswith("retained_s2_")]
    excluded_s1 = [s for s in samples if s.name.startswith("excluded_s1_")]
    excluded_s2 = [s for s in samples if s.name.startswith("excluded_s2_")]
    
    assert len(retained) == 15
    assert len(excluded_s1) == 12
    assert len(excluded_s2) == 3
