"""
Unit tests for YOLO annotation verification.
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
