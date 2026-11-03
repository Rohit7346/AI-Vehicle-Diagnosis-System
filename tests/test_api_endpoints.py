"""
Integration tests for FastAPI endpoints.
"""

import io
from fastapi.testclient import TestClient
from PIL import Image

from backend.app.main import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "project" in data
    assert "health" in data


def test_health_endpoint():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "models" in data
    assert "component_detector" in data["models"]
    assert "fault_detector" in data["models"]
    assert data["models"]["component_detector"]["loaded"] is False


def test_diagnose_image_missing_weights_returns_503():
    # Create test in-memory image
    img = Image.new("RGB", (640, 640), color=(100, 100, 100))
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    buffer.seek(0)

    # When model weights are absent, endpoint must return 503 instead of mock data
    response = client.post(
        "/api/v1/diagnose/image",
        files={"file": ("test_part.jpg", buffer, "image/jpeg")},
    )
    assert response.status_code == 503
    data = response.json()
    assert "detail" in data
    assert data["detail"]["error"] == "Model weights uninitialized or not found on disk."
