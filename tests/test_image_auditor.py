"""
Unit tests for ImageQualityAuditor.
Tests corruption detection, dimension checks, exact SHA-256 duplicates,
perceptual near-duplicates via dHash, and blur/exposure validation.
"""

from pathlib import Path
import tempfile
import numpy as np
from PIL import Image

from ml.datasets.audit_staged_images import ImageQualityAuditor


def test_corrupted_image_detection():
    auditor = ImageQualityAuditor()
    with tempfile.TemporaryDirectory() as td:
        corrupted_path = Path(td) / "corrupted.jpg"
        with open(corrupted_path, "wb") as f:
            f.write(b"NOT_A_VALID_JPEG_STREAM_CORRUPTED_BYTES")

        res = auditor.inspect_single_image(corrupted_path)
        assert res["is_corrupted"] is True
        assert res["status"] == "REJECTED"
        assert any("Corrupted" in r for r in res["rejection_reasons"])


def test_undersized_image_detection():
    auditor = ImageQualityAuditor(min_width=400, min_height=400)
    with tempfile.TemporaryDirectory() as td:
        small_path = Path(td) / "small.jpg"
        img = Image.new("RGB", (250, 250), color=(100, 100, 100))
        img.save(small_path)

        res = auditor.inspect_single_image(small_path)
        assert res["status"] == "REJECTED"
        assert any("Undersized" in r for r in res["rejection_reasons"])


def test_exact_and_near_duplicate_detection():
    auditor = ImageQualityAuditor(near_duplicate_hamming_dist=4)
    with tempfile.TemporaryDirectory() as td:
        # Create base textured image
        arr = np.random.RandomState(42).randint(50, 200, (500, 500, 3), dtype=np.uint8)
        img_a = Image.fromarray(arr)

        path_a = Path(td) / "image_a.jpg"
        path_b_exact = Path(td) / "image_b_exact.jpg"
        path_c_near = Path(td) / "image_c_near.jpg"

        img_a.save(path_a, quality=95)
        # Exact copy
        img_a.save(path_b_exact, quality=95)

        # Near copy: slight pixel noise
        arr_noisy = arr.copy()
        arr_noisy[0:10, 0:10] = 120
        Image.fromarray(arr_noisy).save(path_c_near, quality=90)

        audit = auditor.audit_collection(Path(td))
        assert audit["total_images"] == 3
        # Should detect exact duplicate
        assert len(audit["exact_duplicate_groups"]) >= 1
        # Should detect near-duplicate pairs
        assert len(audit["near_duplicate_pairs"]) >= 1


def test_valid_image_acceptance():
    auditor = ImageQualityAuditor(min_width=400, min_height=400, blur_threshold=50.0)
    with tempfile.TemporaryDirectory() as td:
        # Sharp high-frequency checkerboard texture
        x = np.arange(600)
        y = np.arange(600)
        xx, yy = np.meshgrid(x, y)
        pattern = ((xx // 20 + yy // 20) % 2 * 200 + 40).astype(np.uint8)
        rgb_arr = np.stack([pattern, pattern, pattern], axis=-1)

        valid_path = Path(td) / "sharp_engine.jpg"
        Image.fromarray(rgb_arr).save(valid_path, quality=95)

        res = auditor.inspect_single_image(valid_path)
        assert res["is_corrupted"] is False
        assert res["width"] == 600
        assert res["height"] == 600
        assert res["blur_variance"] > 100.0
        assert res["status"] == "ACCEPTED"
        assert len(res["rejection_reasons"]) == 0
