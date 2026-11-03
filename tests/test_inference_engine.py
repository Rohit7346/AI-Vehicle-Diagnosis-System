"""
Unit tests verifying inference safeguards and error reporting.
"""

from pathlib import Path
import pytest
from PIL import Image

from ml.inference.engine import DiagnosticInferenceEngine
from ml.models.base_model import ModelWeightsNotFoundError


def test_engine_missing_weights_safeguard():
    engine = DiagnosticInferenceEngine(
        component_weights_path="ml/weights/non_existent_comp.pt",
        fault_weights_path="ml/weights/non_existent_fault.pt",
    )
    status = engine.get_status()
    assert status["pipeline_ready"] is False
    assert status["component_detector"]["loaded"] is False
    assert status["fault_detector"]["loaded"] is False

    dummy_image = Image.new("RGB", (640, 640), color=(128, 128, 128))
    # Must raise ModelWeightsNotFoundError and refuse to emit mock data
    with pytest.raises(ModelWeightsNotFoundError):
        engine.diagnose(dummy_image)
