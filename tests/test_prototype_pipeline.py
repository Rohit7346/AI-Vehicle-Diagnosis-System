"""
Automated tests for the 5-class prototype filtering pipeline.
Validates:
1. Class remapping logic (Battery -> 1, Radiator -> 3, Main Engine never -> 0).
2. Bounding box coordinate boundaries and degenerate detection.
3. Skip decisions for images without accepted annotations.
4. Split preservation (train, val, test separation without cross-split leakage).
5. Strict 1:1 image-to-label file pairing in the generated dataset.
6. Prototype dataset YAML specification and class ontology order.
"""

import os
from pathlib import Path
import tempfile
import pytest
import yaml
from PIL import Image

from ml.datasets.filter_remap_prototype import (
    PrototypeDatasetPipeline,
    ORIGINAL_CLASS_NAMES,
    PROTOTYPE_ONTOLOGY,
    PERMITTED_REMAPPING,
    EXCLUDED_SOURCE_CLASSES,
    SUPPORTED_IMAGE_EXTS,
    make_long_path,
)


def test_remapping_logic_battery_and_radiator():
    """Verify Battery maps to 1, Radiator maps to 3, with valid normalized coordinates."""
    # Battery (raw ID 1) -> Target ID 1
    battery_line = "1 0.450000 0.550000 0.200000 0.300000"
    remapped, err, raw_cls = PrototypeDatasetPipeline.remap_annotation_row(battery_line)
    assert err is None
    assert raw_cls == 1
    assert remapped is not None
    assert remapped[0] == 1  # Target ID 1 (battery)
    assert remapped[1:] == (0.45, 0.55, 0.2, 0.3)

    # Radiator (raw ID 4) -> Target ID 3
    radiator_line = "4 0.500000 0.800000 0.600000 0.150000"
    remapped, err, raw_cls = PrototypeDatasetPipeline.remap_annotation_row(radiator_line)
    assert err is None
    assert raw_cls == 4
    assert remapped is not None
    assert remapped[0] == 3  # Target ID 3 (radiator)
    assert remapped[1:] == (0.5, 0.8, 0.6, 0.15)


def test_main_engine_never_mapped_to_engine_bay():
    """
    CRITICAL ARCHITECTURAL SAFEGUARD:
    Main Engine (raw ID 3) must be strictly excluded and NEVER mapped to engine_bay (0).
    """
    main_engine_line = "3 0.500000 0.500000 0.400000 0.400000"
    remapped, err, raw_cls = PrototypeDatasetPipeline.remap_annotation_row(main_engine_line)
    assert err is None
    assert raw_cls == 3
    assert remapped is None, "Main Engine must be excluded and never produce a target annotation"


def test_excluded_classes_filtered_out():
    """Verify Air Coolant (0), Fuse-Box (2), and Windscreen Washer (5) are excluded."""
    for excluded_id in [0, 2, 5]:
        line = f"{excluded_id} 0.5 0.5 0.2 0.2"
        remapped, err, raw_cls = PrototypeDatasetPipeline.remap_annotation_row(line)
        assert err is None
        assert raw_cls == excluded_id
        assert remapped is None


def test_coordinate_validation_boundary_checks():
    """Verify out-of-bounds, negative, and degenerate bounding boxes are detected."""
    # Out of bounds center coordinate
    line_oob = "1 1.500000 0.500000 0.200000 0.200000"
    remapped, err, _ = PrototypeDatasetPipeline.remap_annotation_row(line_oob)
    assert remapped is None
    assert err is not None
    assert "out of [0.0, 1.0]" in err

    # Degenerate dimensions (w <= 0)
    line_degenerate = "1 0.500000 0.500000 0.000000 0.200000"
    remapped, err, _ = PrototypeDatasetPipeline.remap_annotation_row(line_degenerate)
    assert remapped is None
    assert err is not None
    assert "Degenerate box dimensions" in err

    # Corner exceeds 1.0 boundary
    line_corner_exceed = "4 0.900000 0.900000 0.400000 0.400000"
    remapped, err, _ = PrototypeDatasetPipeline.remap_annotation_row(line_corner_exceed)
    assert remapped is None
    assert err is not None
    assert "Box exceeds boundary" in err


def test_synthetic_filtering_pipeline_execution():
    """
    Creates a temporary source dataset and tests that:
    1. Images with only excluded classes are skipped.
    2. Images with empty label files are skipped.
    3. Images with accepted classes are retained with remapped labels.
    4. Splits are strictly preserved without leakage.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        src_dir = Path(tmp_dir) / "source"
        out_dir = Path(tmp_dir) / "output"

        # Create source structure
        for split in ["train", "valid", "test"]:
            (src_dir / split / "images").mkdir(parents=True)
            (src_dir / split / "labels").mkdir(parents=True)

        # Write valid source data.yaml
        src_yaml = {
            "train": "../train/images",
            "val": "../valid/images",
            "test": "../test/images",
            "nc": 6,
            "names": [ORIGINAL_CLASS_NAMES[i] for i in range(6)],
        }
        with open(src_dir / "data.yaml", "w", encoding="utf-8") as f:
            yaml.dump(src_yaml, f)

        # Split 1 (train):
        # Image 1: Has Battery (1) and Fuse-Box (2) -> Retain Battery (1)
        img1 = Image.new("RGB", (100, 100), color=(10, 10, 10))
        img1.save(src_dir / "train" / "images" / "img_battery.jpg")
        with open(src_dir / "train" / "labels" / "img_battery.txt", "w") as f:
            f.write("1 0.3 0.3 0.2 0.2\n2 0.7 0.7 0.1 0.1\n")

        # Image 2: Has only Main Engine (3) and Air Coolant (0) -> SKIP
        img2 = Image.new("RGB", (100, 100), color=(20, 20, 20))
        img2.save(src_dir / "train" / "images" / "img_engine_only.jpg")
        with open(src_dir / "train" / "labels" / "img_engine_only.txt", "w") as f:
            f.write("3 0.5 0.5 0.4 0.4\n0 0.1 0.1 0.1 0.1\n")

        # Image 3: Empty label file -> SKIP
        img3 = Image.new("RGB", (100, 100), color=(30, 30, 30))
        img3.save(src_dir / "train" / "images" / "img_empty.jpg")
        with open(src_dir / "train" / "labels" / "img_empty.txt", "w") as f:
            f.write("")

        # Split 2 (valid):
        # Image 4: Has Radiator (4) -> Retain mapped to Radiator (3)
        img4 = Image.new("RGB", (100, 100), color=(40, 40, 40))
        img4.save(src_dir / "valid" / "images" / "img_radiator.jpg")
        with open(src_dir / "valid" / "labels" / "img_radiator.txt", "w") as f:
            f.write("4 0.5 0.8 0.5 0.2\n")

        pipeline = PrototypeDatasetPipeline(src_dir, out_dir)
        audit = pipeline.execute_filtering()

        # Assertions on Train split
        assert audit["splits"]["train"]["source_images"] == 3
        assert audit["splits"]["train"]["retained_images"] == 1
        assert audit["splits"]["train"]["skipped_images"] == 2
        assert audit["splits"]["train"]["battery_annotations"] == 1
        assert audit["splits"]["train"]["radiator_annotations"] == 0
        assert audit["splits"]["train"]["excluded_annotations"] == 3

        # Assertions on Val split
        assert audit["splits"]["val"]["source_images"] == 1
        assert audit["splits"]["val"]["retained_images"] == 1
        assert audit["splits"]["val"]["skipped_images"] == 0
        assert audit["splits"]["val"]["battery_annotations"] == 0
        assert audit["splits"]["val"]["radiator_annotations"] == 1

        # Check output label content
        lbl_out = out_dir / "labels" / "train" / "img_battery.txt"
        assert lbl_out.exists()
        with open(lbl_out) as lf:
            rows = lf.read().strip().splitlines()
        assert len(rows) == 1
        assert rows[0].startswith("1 ")  # Class 1 (battery)

        # Radiator remapped from 4 to 3
        rad_lbl = out_dir / "labels" / "val" / "img_radiator.txt"
        assert rad_lbl.exists()
        with open(rad_lbl) as lf:
            rows = lf.read().strip().splitlines()
        assert len(rows) == 1
        assert rows[0].startswith("3 ")  # Class 3 (radiator)

        # Check output data.yaml
        out_yaml_path = out_dir / "data.yaml"
        assert out_yaml_path.exists()
        with open(out_yaml_path) as yf:
            cfg = yaml.safe_load(yf)
        assert cfg["nc"] == 5
        assert cfg["names"][0] == "engine_bay"
        assert cfg["names"][1] == "battery"
        assert cfg["names"][2] == "alternator"
        assert cfg["names"][3] == "radiator"
        assert cfg["names"][4] == "exhaust_manifold"


def test_generated_prototype_dataset_pairing_and_ontology():
    """
    Validates the actual generated prototype dataset in data/prototype_5class:
    - Verifies 100% 1:1 image and label pairing.
    - Verifies all label files contain at least 1 annotation.
    - Verifies only class 1 (battery) and class 3 (radiator) exist.
    - Verifies no annotations exist for unpopulated classes (0, 2, 4).
    """
    proto_dir = Path("data/prototype_5class")
    if not proto_dir.exists():
        pytest.skip("data/prototype_5class not yet generated on disk")

    yaml_path = proto_dir / "data.yaml"
    assert yaml_path.exists()
    with open(yaml_path, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    assert data_cfg["nc"] == 5
    assert data_cfg["names"] == PROTOTYPE_ONTOLOGY

    total_images = 0
    total_labels = 0
    class_counts = {i: 0 for i in range(5)}

    for split in ["train", "val", "test"]:
        img_dir = proto_dir / "images" / split
        lbl_dir = proto_dir / "labels" / split

        assert img_dir.exists(), f"Missing split {img_dir}"
        assert lbl_dir.exists(), f"Missing split {lbl_dir}"

        img_names = [f for f in os.listdir(make_long_path(img_dir)) if Path(f).suffix.lower() in SUPPORTED_IMAGE_EXTS]
        lbl_names = [f for f in os.listdir(make_long_path(lbl_dir)) if Path(f).suffix.lower() == ".txt"]

        imgs_stems = {Path(f).stem: f for f in img_names}
        lbls_stems = {Path(f).stem: f for f in lbl_names}

        # Strict 1:1 pairing
        assert imgs_stems.keys() == lbls_stems.keys(), f"Image/Label mismatch in {split} split!"
        total_images += len(imgs_stems)
        total_labels += len(lbls_stems)

        # Check labels content
        for stem, fname in lbls_stems.items():
            lbl_file = make_long_path(lbl_dir / fname)
            with open(lbl_file, "r", encoding="utf-8") as f:
                lines = [l.strip() for l in f if l.strip()]
            assert len(lines) > 0, f"Empty label file created: {lbl_file}"
            for line in lines:
                parts = line.split()
                cls_id = int(parts[0])
                class_counts[cls_id] += 1
                assert cls_id in (1, 3), f"Unexpected class ID {cls_id} found in prototype label {lbl_file}"

    assert total_images == 1148
    assert total_labels == 1148
    assert class_counts[1] == 1104  # All Battery annotations preserved
    assert class_counts[3] == 680   # All Radiator annotations preserved
    assert class_counts[0] == 0     # engine_bay not invented
    assert class_counts[2] == 0     # alternator not invented
    assert class_counts[4] == 0     # exhaust_manifold not invented
