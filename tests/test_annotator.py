"""
Unit tests for YOLO annotation verification and dataset integrity checks.
"""

from pathlib import Path
import tempfile
import pytest
from PIL import Image

from ml.datasets.annotator_utils import AnnotationValidator


def test_annotation_validator_empty_splits():
    with tempfile.TemporaryDirectory() as tmp_dir:
        img_dir = Path(tmp_dir) / "images"
        lbl_dir = Path(tmp_dir) / "labels"
        img_dir.mkdir()
        lbl_dir.mkdir()

        audit = AnnotationValidator.validate_split(img_dir, lbl_dir, expected_classes=10)
        assert audit["valid"] is False
        assert audit["total_images"] == 0


def test_annotation_validator_valid_sample():
    with tempfile.TemporaryDirectory() as tmp_dir:
        img_dir = Path(tmp_dir) / "images"
        lbl_dir = Path(tmp_dir) / "labels"
        img_dir.mkdir()
        lbl_dir.mkdir()

        # Create dummy image and matching label
        img = Image.new("RGB", (640, 640), color=(50, 50, 50))
        img.save(img_dir / "brake_disc_01.jpg")

        # Class 4 (brake_rotor), normalized bbox [0.5, 0.5, 0.4, 0.4]
        with open(lbl_dir / "brake_disc_01.txt", "w") as f:
            f.write("4 0.5 0.5 0.4 0.4\n")

        audit = AnnotationValidator.validate_split(img_dir, lbl_dir, expected_classes=10)
        assert audit["valid"] is True
        assert audit["total_images"] == 1
        assert audit["total_labels"] == 1
        assert audit["total_annotations"] == 1
        assert audit["class_counts_by_id"][4] == 1


def test_annotation_validator_out_of_bounds_and_invalid_class():
    with tempfile.TemporaryDirectory() as tmp_dir:
        img_dir = Path(tmp_dir) / "images"
        lbl_dir = Path(tmp_dir) / "labels"
        img_dir.mkdir()
        lbl_dir.mkdir()

        img = Image.new("RGB", (640, 640), color=(50, 50, 50))
        img.save(img_dir / "invalid_01.jpg")

        # Line 1: Class 15 (exceeds [0, 9])
        # Line 2: Coordinate 1.5 (exceeds [0.0, 1.0])
        with open(lbl_dir / "invalid_01.txt", "w") as f:
            f.write("15 0.5 0.5 0.2 0.2\n")
            f.write("0 1.5 0.5 0.2 0.2\n")

        audit = AnnotationValidator.validate_split(img_dir, lbl_dir, expected_classes=10)
        assert audit["valid"] is False
        assert len(audit["out_of_bounds_errors"]) >= 2


def test_detect_data_leakage():
    with tempfile.TemporaryDirectory() as tmp_dir:
        base_dir = Path(tmp_dir)
        train_dir = base_dir / "images" / "train"
        val_dir = base_dir / "images" / "val"
        train_dir.mkdir(parents=True)
        val_dir.mkdir(parents=True)

        # Create identical content in train and val to simulate leakage
        img1 = Image.new("RGB", (640, 640), color=(120, 130, 140))
        img1.save(train_dir / "engine_01.jpg")
        img1.save(val_dir / "engine_leak_copy.jpg")

        leak_report = AnnotationValidator.detect_data_leakage(base_dir)
        assert leak_report["has_leakage"] is True
        assert len(leak_report["content_duplicates"]) > 0
