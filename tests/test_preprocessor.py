"""
Unit tests for ImagePreprocessor.
"""

import numpy as np
import pytest
from PIL import Image

from ml.datasets.preprocessor import ImagePreprocessor


def test_preprocessor_image_resizing_and_letterbox():
    preprocessor = ImagePreprocessor(target_size=(640, 640), apply_clahe=False)
    # Create test 1920x1080 synthetic image
    test_image = Image.new("RGB", (1920, 1080), color=(100, 150, 200))

    processed_arr, metadata = preprocessor.process(test_image)

    assert processed_arr.shape == (640, 640, 3)
    assert metadata["original_width"] == 1920
    assert metadata["original_height"] == 1080
    assert metadata["target_size"] == (640, 640)
    assert metadata["scale"] == pytest.approx(640 / 1920, rel=1e-3)


def test_preprocessor_invalid_source():
    preprocessor = ImagePreprocessor()
    with pytest.raises(FileNotFoundError):
        preprocessor.process("non_existent_vehicle_part.jpg")
