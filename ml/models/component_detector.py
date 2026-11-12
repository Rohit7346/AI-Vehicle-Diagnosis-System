"""
Vehicle Component Detection Module.
Detects automotive subassemblies (engine bay, battery, radiator)
using trained Ultralytics YOLOv8 architecture.
"""

from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image

from .base_model import BaseVehicleDetector, ModelWeightsNotFoundError

# Strict unified 3-class ontology for Stage 3.x
COMPONENT_CLASSES: Dict[int, str] = {
    0: "engine_bay",
    1: "battery",
    2: "radiator",
}

# Per-class policy metadata
CLASS_POLICIES: Dict[Union[int, str], Dict[str, Any]] = {
    0: {
        "class_id": 0,
        "class_name": "engine_bay",
        "status": "enabled",
        "is_experimental": False,
        "safeguard_warning": (
            "Engine bay localization identifies the visible engine compartment boundary. "
            "Known limitation: closed-hood vehicle exteriors may trigger false positives (e.g. Ford Ranger front end). "
            "Visibility verification required before proceeding with diagnostic workflows."
        ),
        "warning": (
            "Engine bay localization identifies the visible engine compartment boundary. "
            "Known limitation: closed-hood vehicle exteriors may trigger false positives (e.g. Ford Ranger front end). "
            "Visibility verification required before proceeding with diagnostic workflows."
        ),
    },
    1: {
        "class_id": 1,
        "class_name": "battery",
        "status": "enabled",
        "is_experimental": False,
        "safeguard_warning": None,
        "warning": None,
    },
    2: {
        "class_id": 2,
        "class_name": "radiator",
        "status": "experimental",
        "is_experimental": True,
        "safeguard_warning": (
            "Provisional detection: Radiator component model is experimental "
            "(limited training sample volume and extreme aspect ratio). "
            "Requires technician manual verification."
        ),
        "warning": (
            "Provisional detection: Radiator component model is experimental "
            "(limited training sample volume and extreme aspect ratio). "
            "Requires technician manual verification."
        ),
    },
}
# Add named class string aliases
CLASS_POLICIES["engine_bay"] = CLASS_POLICIES[0]
CLASS_POLICIES["battery"] = CLASS_POLICIES[1]
CLASS_POLICIES["radiator"] = CLASS_POLICIES[2]


class ComponentDetector(BaseVehicleDetector):
    """
    Detects and localizes vehicle subsystems (engine_bay, battery, radiator).
    Implements singleton model caching, CUDA/CPU device resolution,
    and granular class policy metadata.
    """

    # Class-level model cache to ensure single load across requests
    _cached_model = None
    _cached_weights_path: Optional[str] = None
    _cached_device: Optional[str] = None

    def __init__(
        self,
        weights_path: Optional[Union[str, Path]] = None,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = "auto",
    ):
        resolved_device = self._resolve_device(device)
        super().__init__(
            weights_path=weights_path,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            device=resolved_device,
        )
        self._class_names = COMPONENT_CLASSES.copy()

    @staticmethod
    def _resolve_device(requested_device: str) -> str:
        """Resolves target compute device with automatic CUDA detection and CPU fallback."""
        try:
            import torch
            cuda_available = torch.cuda.is_available()
        except ImportError:
            cuda_available = False

        if requested_device in ("auto", None):
            return "cuda:0" if cuda_available else "cpu"
        elif "cuda" in requested_device.lower():
            return requested_device if cuda_available else "cpu"
        return "cpu"

    def _load_weights(self, weights_path: Path) -> None:
        """Loads YOLO weights checkpoint once with caching and error handling."""
        resolved_path = weights_path.resolve() if weights_path.exists() else weights_path

        # If identical weights and device are already loaded in memory, reuse them
        if (
            ComponentDetector._cached_model is not None
            and ComponentDetector._cached_weights_path == str(resolved_path)
            and ComponentDetector._cached_device == self.device
        ):
            self.model = ComponentDetector._cached_model
            return

        try:
            from ultralytics import YOLO
            loaded = YOLO(str(weights_path))
            # Push model to resolved device
            if hasattr(loaded, "to"):
                try:
                    loaded.to(self.device)
                except Exception:
                    pass

            self.model = loaded
            ComponentDetector._cached_model = loaded
            ComponentDetector._cached_weights_path = str(resolved_path)
            ComponentDetector._cached_device = self.device

            if hasattr(self.model, "names") and self.model.names:
                self._class_names = {int(k): v for k, v in self.model.names.items()}
        except ImportError:
            raise ImportError(
                "Ultralytics is required for ComponentDetector. Install with: pip install ultralytics"
            )
        except Exception as e:
            self.model = None
            raise RuntimeError(f"Failed to load component model checkpoint from {weights_path}: {e}")

    def predict(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
        confidence: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Runs component detection inference.
        Returns standard detection dictionaries compatible with DiagnosticInferenceEngine.
        Raises ModelWeightsNotFoundError if no trained weights are loaded.
        """
        results_bundle = self.detect_components(image=image, confidence=confidence)
        # Return components list for backward compatibility with DiagnosticInferenceEngine
        return results_bundle["detections"]

    def detect_components(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
        confidence: Optional[float] = None,
        confidence_override: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Executes full component detection pipeline.
        Calculates normalized and pixel bounding boxes, class policy metadata,
        inference latency, and image dimensions.
        """
        if not self.is_loaded:
            raise ModelWeightsNotFoundError(
                f"Component detector weights not found or not loaded (Target path: '{self.weights_path}'). "
                f"Please ensure valid YOLO weights exist at ml/weights/component_detector_best.pt."
            )

        # Determine image dimensions
        img_w, img_h = self._get_image_dimensions(image)

        active_conf = confidence if confidence is not None else confidence_override
        conf = active_conf if active_conf is not None else self.confidence_threshold
        t_start = time.perf_counter()
        
        results = self.model.predict(
            source=image,
            conf=conf,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )
        inference_time_ms = round((time.perf_counter() - t_start) * 1000.0, 2)

        detections: List[Dict[str, Any]] = []
        counts = {"engine_bay": 0, "battery": 0, "radiator": 0}
        has_experimental = False

        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue

            # Update dimensions from result orig_shape if available
            if hasattr(r, "orig_shape") and r.orig_shape:
                img_h, img_w = r.orig_shape[:2]

            for box in boxes:
                xyxy = box.xyxy[0].tolist()
                cls_id = int(box.cls[0].item())
                score = float(box.conf[0].item())
                label = self._class_names.get(cls_id, f"unknown_component_{cls_id}")

                x1, y1, x2, y2 = xyxy
                # Clamp pixel coordinates
                x1_px = max(0, min(img_w, int(round(x1))))
                y1_px = max(0, min(img_h, int(round(y1))))
                x2_px = max(0, min(img_w, int(round(x2))))
                y2_px = max(0, min(img_h, int(round(y2))))

                # Calculate normalized coordinates in [0.0, 1.0]
                x1_norm = round(x1_px / img_w, 4) if img_w > 0 else 0.0
                y1_norm = round(y1_px / img_h, 4) if img_h > 0 else 0.0
                x2_norm = round(x2_px / img_w, 4) if img_w > 0 else 0.0
                y2_norm = round(y2_px / img_h, 4) if img_h > 0 else 0.0

                policy = CLASS_POLICIES.get(
                    cls_id,
                    {"status": "enabled", "is_experimental": False, "safeguard_warning": None},
                )
                if policy["is_experimental"]:
                    has_experimental = True

                if label in counts:
                    counts[label] += 1

                detection_item = {
                    "component_id": f"comp_{cls_id}_{len(detections)+1:02d}",
                    "class_id": cls_id,
                    "class_name": label,
                    "label": label,
                    "confidence": round(score, 4),
                    "bbox": [x1_px, y1_px, x2_px, y2_px],
                    "box_pixels": {
                        "x_min": x1_px,
                        "y_min": y1_px,
                        "x_max": x2_px,
                        "y_max": y2_px,
                    },
                    "box_normalized": {
                        "x_min": x1_norm,
                        "y_min": y1_norm,
                        "x_max": x2_norm,
                        "y_max": y2_norm,
                    },
                    "status": policy["status"],
                    "is_experimental": policy["is_experimental"],
                    "warning": policy["safeguard_warning"],
                }
                detections.append(detection_item)

        classes_found = sorted(list({d["label"] for d in detections}))

        return {
            "detections": detections,
            "inference_time_ms": inference_time_ms,
            "device": self.device,
            "image_dimensions": {"width": img_w, "height": img_h},
            "image_size": {"width": img_w, "height": img_h},
            "total_detections": len(detections),
            "class_summary": counts,
            "disclaimer": (
                "Component localization detects physical vehicle components only and does NOT indicate "
                "mechanical faults or operational status. Detections must not be presented as fault probabilities."
            ),
            "summary": {
                "total_detected": len(detections),
                "classes_detected": classes_found,
                "has_experimental_components": has_experimental,
                "engine_bay_count": counts["engine_bay"],
                "battery_count": counts["battery"],
                "radiator_count": counts["radiator"],
            },
        }

    @staticmethod
    def _get_image_dimensions(image: Union[str, Path, np.ndarray, Image.Image]) -> Tuple[int, int]:
        """Helper to extract width and height from varied input types."""
        if isinstance(image, Image.Image):
            return image.width, image.height
        elif isinstance(image, np.ndarray):
            return int(image.shape[1]), int(image.shape[0])
        elif isinstance(image, (str, Path)):
            with Image.open(image) as im:
                return im.width, im.height
        return 640, 640
