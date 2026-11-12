"""
Component Service layer encapsulating YOLOv8 vehicle component localization.
Orchestrates model lifecycle, image decoding, inference execution, and schema conversion.
"""

import io
from pathlib import Path
from typing import Dict, List, Optional, Union
from PIL import Image

from backend.app.core.config import settings
from backend.app.schemas.component_detection import (
    BoundingBoxNormalized,
    BoundingBoxPixels,
    ComponentDetectionResponse,
    ComponentDetectionSummary,
    DetectedComponent,
    ImageDimensions,
)
from ml.models.base_model import ModelWeightsNotFoundError
from ml.models.component_detector import CLASS_POLICIES, ComponentDetector


class ComponentService:
    """Manages the lifecycle and inference execution of the YOLOv8 component detector."""

    def __init__(self):
        self._detector: Optional[ComponentDetector] = None

    @property
    def detector(self) -> ComponentDetector:
        """Lazy initializer for the component detector singleton."""
        if self._detector is None:
            self._detector = ComponentDetector(
                weights_path=settings.COMPONENT_MODEL_WEIGHTS,
                confidence_threshold=settings.COMPONENT_CONFIDENCE_THRESHOLD,
                iou_threshold=settings.IOU_THRESHOLD,
                device=settings.DEVICE,
            )
        return self._detector

    def get_status(self) -> Dict[str, object]:
        """Returns readiness and configuration metrics for the component detector."""
        weights_exist = Path(settings.COMPONENT_MODEL_WEIGHTS).exists()
        is_loaded = self.detector.is_loaded if weights_exist else False
        named_policies = {
            k: v for k, v in CLASS_POLICIES.items() if isinstance(k, str)
        }
        return {
            "model_name": settings.COMPONENT_MODEL_NAME,
            "version": settings.COMPONENT_MODEL_VERSION,
            "weights_path": str(settings.COMPONENT_MODEL_WEIGHTS),
            "weights_exist_on_disk": weights_exist,
            "loaded": is_loaded,
            "device": self.detector.device,
            "confidence_threshold": settings.COMPONENT_CONFIDENCE_THRESHOLD,
            "iou_threshold": settings.IOU_THRESHOLD,
            "classes": self.detector.class_names,
            "class_policies": named_policies,
        }

    def detect_components(
        self,
        image_bytes: bytes,
        confidence_override: Optional[float] = None,
    ) -> ComponentDetectionResponse:
        """
        Executes component localization on raw image bytes.

        Args:
            image_bytes: Raw binary bytes of uploaded image.
            confidence_override: Optional per-request confidence threshold.

        Returns:
            Structured ComponentDetectionResponse.

        Raises:
            ValueError: If image bytes cannot be decoded or verified.
            ModelWeightsNotFoundError: If model weights are missing or not initialized.
        """
        # 1. Verify weights availability
        if not self.detector.is_loaded:
            raise ModelWeightsNotFoundError(
                f"Component detector model weights '{settings.COMPONENT_MODEL_WEIGHTS}' not found or failed to load. "
                "Ensure trained weights are placed at the target path."
            )

        # 2. Decode and validate image with PIL
        try:
            image = Image.open(io.BytesIO(image_bytes))
            # Verify image integrity
            image.verify()
            # Reopen because verify() consumes the buffer and damages PIL handle
            image = Image.open(io.BytesIO(image_bytes))
            image_format = image.format or "JPEG"
        except Exception as exc:
            raise ValueError(f"Corrupt or invalid image payload: {str(exc)}") from exc

        # 3. Execute inference via ComponentDetector
        raw_result = self.detector.detect_components(
            image=image,
            confidence_override=confidence_override,
        )

        # 4. Assemble structured response
        detections: List[DetectedComponent] = []
        for idx, det in enumerate(raw_result["detections"]):
            box_px = BoundingBoxPixels(
                x_min=int(det["box_pixels"]["x_min"]),
                y_min=int(det["box_pixels"]["y_min"]),
                x_max=int(det["box_pixels"]["x_max"]),
                y_max=int(det["box_pixels"]["y_max"]),
            )
            box_norm = BoundingBoxNormalized(
                x_min=float(det["box_normalized"]["x_min"]),
                y_min=float(det["box_normalized"]["y_min"]),
                x_max=float(det["box_normalized"]["x_max"]),
                y_max=float(det["box_normalized"]["y_max"]),
            )
            detections.append(
                DetectedComponent(
                    component_id=f"comp_{idx + 1}",
                    class_id=int(det["class_id"]),
                    class_name=str(det["class_name"]),
                    confidence=float(det["confidence"]),
                    box_pixels=box_px,
                    box_normalized=box_norm,
                    status=str(det["status"]),
                    is_experimental=bool(det["is_experimental"]),
                    warning=det.get("warning"),
                )
            )

        summary = ComponentDetectionSummary(
            total_detections=raw_result["total_detections"],
            detected_classes=list(raw_result["class_summary"].keys()),
            class_counts=raw_result["class_summary"],
        )

        image_dims = ImageDimensions(
            width=raw_result["image_size"]["width"],
            height=raw_result["image_size"]["height"],
            format=image_format,
        )

        safeguards = {
            str(cls_name): str(policy["warning"])
            for cls_name, policy in CLASS_POLICIES.items()
            if isinstance(cls_name, str) and policy.get("warning") is not None
        }

        return ComponentDetectionResponse(
            success=True,
            model_name=settings.COMPONENT_MODEL_NAME,
            model_version=settings.COMPONENT_MODEL_VERSION,
            device=raw_result["device"],
            inference_latency_ms=raw_result["inference_time_ms"],
            image=image_dims,
            detections=detections,
            summary=summary,
            disclaimer=raw_result["disclaimer"],
            safeguards=safeguards,
        )


component_service = ComponentService()
