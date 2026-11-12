"""
Integration and safety policy tests for the YOLOv8 Component Localization API.
Verifies endpoint functionality, response schemas, error handling,
per-class policies (enabled vs. experimental), and hardware fallbacks.
"""

import io
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.app.core.config import settings
from backend.app.main import app
from backend.app.services.component_service import component_service
from ml.models.base_model import ModelWeightsNotFoundError
from ml.models.component_detector import ComponentDetector

client = TestClient(app)

TEST_IMAGE_PATH = Path("data/processed/stage3_0_three_class/images/test/04312000010000001448_17_jpg.rf.655aaf1f7fd63c6d611ada7ae074db29.jpg")


def _generate_synthetic_image(width=640, height=480, format="JPEG") -> bytes:
    """Generates a valid in-memory test image."""
    img = Image.new("RGB", (width, height), color=(70, 90, 120))
    buf = io.BytesIO()
    img.save(buf, format=format)
    return buf.getvalue()


def test_detect_status_endpoint():
    """Verifies that /api/v1/detect/status reports accurate model and policy metadata."""
    response = client.get("/api/v1/detect/status")
    assert response.status_code == 200
    data = response.json()

    assert data["model_name"] == settings.COMPONENT_MODEL_NAME
    assert data["version"] == settings.COMPONENT_MODEL_VERSION
    assert data["weights_exist_on_disk"] is True
    assert data["loaded"] is True
    assert "classes" in data
    assert data["classes"] == {"0": "engine_bay", "1": "battery", "2": "radiator"}
    assert "class_policies" in data
    assert data["class_policies"]["battery"]["status"] == "enabled"
    assert data["class_policies"]["engine_bay"]["status"] == "enabled"
    assert data["class_policies"]["radiator"]["status"] == "experimental"
    assert data["class_policies"]["radiator"]["is_experimental"] is True


def test_detect_components_synthetic_image():
    """Verifies successful detection run with a synthetic JPEG image."""
    img_bytes = _generate_synthetic_image(640, 480)
    response = client.post(
        "/api/v1/detect/components",
        files={"file": ("test.jpg", img_bytes, "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()

    assert data["success"] is True
    assert data["model_name"] == settings.COMPONENT_MODEL_NAME
    assert data["model_version"] == settings.COMPONENT_MODEL_VERSION
    assert "device" in data
    assert data["inference_latency_ms"] >= 0.0
    assert data["image"]["width"] == 640
    assert data["image"]["height"] == 480
    assert "detections" in data
    assert isinstance(data["detections"], list)
    assert "summary" in data
    assert data["summary"]["total_detections"] == len(data["detections"])
    assert "disclaimer" in data
    assert "mechanical faults" in data["disclaimer"].lower()
    assert "safeguards" in data
    assert "radiator" in data["safeguards"]
    assert "engine_bay" in data["safeguards"]


def test_detect_components_real_test_image():
    """Verifies detection on an authentic vehicle component test set image."""
    assert TEST_IMAGE_PATH.exists(), f"Expected test image at {TEST_IMAGE_PATH}"
    with open(TEST_IMAGE_PATH, "rb") as f:
        img_bytes = f.read()

    response = client.post(
        "/api/v1/detect/components",
        files={"file": (TEST_IMAGE_PATH.name, img_bytes, "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()

    assert data["success"] is True
    assert data["image"]["width"] > 0
    assert data["image"]["height"] > 0
    assert data["summary"]["total_detections"] >= 0

    # Verify per-detection structure if detections occurred
    for det in data["detections"]:
        assert "component_id" in det
        assert det["class_id"] in (0, 1, 2)
        assert det["class_name"] in ("engine_bay", "battery", "radiator")
        assert 0.0 <= det["confidence"] <= 1.0

        # Pixel coordinates
        bp = det["box_pixels"]
        assert 0 <= bp["x_min"] <= bp["x_max"] <= data["image"]["width"]
        assert 0 <= bp["y_min"] <= bp["y_max"] <= data["image"]["height"]

        # Normalized coordinates
        bn = det["box_normalized"]
        assert 0.0 <= bn["x_min"] <= bn["x_max"] <= 1.0
        assert 0.0 <= bn["y_min"] <= bn["y_max"] <= 1.0

        # Class policy safeguards
        if det["class_name"] == "radiator":
            assert det["status"] == "experimental"
            assert det["is_experimental"] is True
            assert det["warning"] is not None
        elif det["class_name"] == "engine_bay":
            assert det["status"] == "enabled"
            assert det["is_experimental"] is False
            assert det["warning"] is not None
        elif det["class_name"] == "battery":
            assert det["status"] == "enabled"
            assert det["is_experimental"] is False


def test_detect_components_confidence_override():
    """Verifies that confidence_threshold parameter is accepted and respected."""
    img_bytes = _generate_synthetic_image(640, 640)
    # High threshold
    response_high = client.post(
        "/api/v1/detect/components",
        files={"file": ("test.jpg", img_bytes, "image/jpeg")},
        data={"confidence_threshold": 0.95},
    )
    assert response_high.status_code == 200
    data_high = response_high.json()

    # Low threshold
    response_low = client.post(
        "/api/v1/detect/components",
        files={"file": ("test.jpg", img_bytes, "image/jpeg")},
        data={"confidence_threshold": 0.05},
    )
    assert response_low.status_code == 200
    data_low = response_low.json()

    assert data_low["summary"]["total_detections"] >= data_high["summary"]["total_detections"]


def test_detect_components_invalid_mime_type():
    """Verifies that non-image MIME types are rejected with HTTP 400."""
    fake_txt = b"Hello, this is not an image."
    response = client.post(
        "/api/v1/detect/components",
        files={"file": ("notes.txt", fake_txt, "text/plain")},
    )
    assert response.status_code == 400
    assert "Unsupported file format" in response.json()["detail"]


def test_detect_components_empty_payload():
    """Verifies that a 0-byte upload is rejected with HTTP 400."""
    response = client.post(
        "/api/v1/detect/components",
        files={"file": ("empty.jpg", b"", "image/jpeg")},
    )
    assert response.status_code == 400
    assert "Empty file uploaded" in response.json()["detail"]


def test_detect_components_corrupted_image_bytes():
    """Verifies that corrupt image data is safely caught and returns HTTP 400."""
    corrupt_bytes = b"\xFF\xD8\xFF\xE0" + b"CORRUPTED_BYTES_HERE" * 10
    response = client.post(
        "/api/v1/detect/components",
        files={"file": ("corrupt.jpg", corrupt_bytes, "image/jpeg")},
    )
    assert response.status_code == 400
    assert "Corrupt or unreadable image file" in response.json()["detail"]


def test_detect_components_invalid_confidence_range():
    """Verifies that out-of-range confidence thresholds are rejected with HTTP 400."""
    img_bytes = _generate_synthetic_image(320, 240)
    for invalid_val in [-0.5, 0.0, 1.5, 99.0]:
        response = client.post(
            "/api/v1/detect/components",
            files={"file": ("test.jpg", img_bytes, "image/jpeg")},
            data={"confidence_threshold": invalid_val},
        )
        assert response.status_code == 400
        assert "Confidence threshold must be between 0.01 and 1.0" in response.json()["detail"]


def test_detect_components_oversized_file(monkeypatch):
    """Verifies that images exceeding MAX_UPLOAD_SIZE_MB trigger HTTP 413."""
    monkeypatch.setattr(settings, "MAX_UPLOAD_SIZE_MB", 0)  # Force limit to 0 MB
    img_bytes = _generate_synthetic_image(100, 100)
    response = client.post(
        "/api/v1/detect/components",
        files={"file": ("test.jpg", img_bytes, "image/jpeg")},
    )
    assert response.status_code == 413
    assert "exceeds limit" in response.json()["detail"]


def test_detect_components_missing_weights_handling(monkeypatch):
    """Verifies that missing weights result in clean HTTP 503 instead of a crash or placeholder output."""
    # Create a fresh detector instance with non-existent path
    dummy_detector = ComponentDetector(
        weights_path=Path("ml/weights/non_existent_stage3_2.pt"),
        device="cpu",
    )
    monkeypatch.setattr(component_service, "_detector", dummy_detector)

    img_bytes = _generate_synthetic_image(100, 100)
    response = client.post(
        "/api/v1/detect/components",
        files={"file": ("test.jpg", img_bytes, "image/jpeg")},
    )
    assert response.status_code == 503
    data = response.json()
    assert "detail" in data
    assert "not found" in data["detail"]["error"].lower() or "uninitialized" in data["detail"]["error"].lower()


def test_component_detector_cpu_device_fallback():
    """Verifies that ComponentDetector can initialize and run explicitly on CPU."""
    detector = ComponentDetector(
        weights_path=settings.COMPONENT_MODEL_WEIGHTS,
        device="cpu",
    )
    assert detector.device == "cpu"
    assert detector.is_loaded is True

    img = Image.new("RGB", (320, 320), color=(50, 50, 50))
    result = detector.detect_components(img)
    assert result["device"] == "cpu"
    assert result["inference_time_ms"] > 0
    assert isinstance(result["detections"], list)
