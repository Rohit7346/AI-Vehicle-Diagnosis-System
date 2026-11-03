"""
Service layer encapsulating ML inference orchestration and image handling.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import sys
from PIL import Image

from backend.app.core.config import settings
from ml.inference.engine import DiagnosticInferenceEngine
from ml.inference.schemas import DiagnosticReport
from ml.models.base_model import ModelWeightsNotFoundError


class DiagnosisService:
    """Manages ML model lifecycle and image diagnostic execution."""

    def __init__(self):
        self._engine: Optional[DiagnosticInferenceEngine] = None

    @property
    def engine(self) -> DiagnosticInferenceEngine:
        """Lazy initializer for the diagnostic engine."""
        if self._engine is None:
            self._engine = DiagnosticInferenceEngine(
                component_weights_path=settings.COMPONENT_MODEL_WEIGHTS,
                fault_weights_path=settings.FAULT_MODEL_WEIGHTS,
                device=settings.DEVICE,
                confidence_threshold=settings.CONFIDENCE_THRESHOLD,
                iou_threshold=settings.IOU_THRESHOLD,
            )
        return self._engine

    def get_system_health(self) -> Dict[str, object]:
        """Collects health and model readiness metrics."""
        import platform

        try:
            import torch
            cuda_avail = torch.cuda.is_available()
        except ImportError:
            cuda_avail = False

        # Ensure upload directory exists and is writable
        upload_path = Path(settings.UPLOAD_DIR)
        upload_path.mkdir(parents=True, exist_ok=True)
        writable = upload_path.exists() and upload_path.is_dir()

        model_status = self.engine.get_status()

        return {
            "status": "healthy",
            "version": settings.VERSION,
            "environment": settings.ENVIRONMENT,
            "system": {
                "python_version": platform.python_version(),
                "cuda_available": cuda_avail,
                "device": settings.DEVICE,
                "upload_dir_writable": writable,
            },
            "models": model_status,
        }

    def process_image_diagnosis(
        self,
        image_bytes: bytes,
        confidence_override: Optional[float] = None,
    ) -> DiagnosticReport:
        """
        Executes end-to-end diagnosis on uploaded image bytes.
        Raises ModelWeightsNotFoundError if weights are not ready on disk.
        """
        return self.engine.diagnose(image_bytes, confidence_override=confidence_override)


diagnosis_service = DiagnosisService()
