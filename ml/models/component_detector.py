"""
Vehicle Component Detection Module.
Detects automotive subassemblies (engine bay, battery, brake rotors, tires, etc.)
using Ultralytics YOLO architecture.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np
from PIL import Image

from .base_model import BaseVehicleDetector, ModelWeightsNotFoundError

COMPONENT_CLASSES = {
    0: "engine_bay",
    1: "battery",
    2: "alternator",
    3: "radiator",
    4: "brake_rotor",
    5: "brake_caliper",
    6: "tire_tread",
    7: "tire_sidewall",
    8: "suspension_strut",
    9: "exhaust_manifold",
}


class ComponentDetector(BaseVehicleDetector):
    """Detects and localizes vehicle subsystems."""

    def __init__(
        self,
        weights_path: Optional[Union[str, Path]] = None,
        confidence_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        device: str = "cpu",
    ):
        super().__init__(
            weights_path=weights_path,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            device=device,
        )
        self._class_names = COMPONENT_CLASSES

    def _load_weights(self, weights_path: Path) -> None:
        """Loads YOLO weights checkpoint."""
        try:
            from ultralytics import YOLO
            self.model = YOLO(str(weights_path))
            if hasattr(self.model, "names") and self.model.names:
                self._class_names = {int(k): v for k, v in self.model.names.items()}
        except ImportError:
            raise ImportError(
                "Ultralytics is required for ComponentDetector. Install with: pip install ultralytics"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load component model checkpoint from {weights_path}: {e}")

    def predict(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
        confidence: Optional[float] = None,
    ) -> List[Dict[str, object]]:
        """
        Runs component detection inference.
        Raises ModelWeightsNotFoundError if no trained weights are present.
        """
        if not self.is_loaded:
            raise ModelWeightsNotFoundError(
                f"Component detector weights not found or not loaded (Target path: '{self.weights_path}'). "
                f"Please train the model via 'python -m ml.training.train_components' or place "
                f"a valid YOLO checkpoint in ml/weights/."
            )

        conf = confidence if confidence is not None else self.confidence_threshold
        results = self.model.predict(
            source=image,
            conf=conf,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )

        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                xyxy = box.xyxy[0].tolist()
                cls_id = int(box.cls[0].item())
                score = float(box.conf[0].item())
                label = self._class_names.get(cls_id, f"unknown_component_{cls_id}")

                detections.append({
                    "component_id": f"comp_{cls_id}",
                    "label": label,
                    "confidence": round(score, 4),
                    "bbox": [round(coord, 2) for coord in xyxy],
                    "class_id": cls_id,
                })

        return detections
