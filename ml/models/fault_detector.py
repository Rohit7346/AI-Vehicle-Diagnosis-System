"""
Visible Vehicle Fault & Defect Detection Module.
Detects physical wear, cracks, fluid leaks, corrosion, and damage
using Ultralytics YOLO architecture.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np
from PIL import Image

from .base_model import BaseVehicleDetector, ModelWeightsNotFoundError

FAULT_CLASSES = {
    0: "fluid_leak_oil",
    1: "fluid_leak_coolant",
    2: "fluid_leak_brake",
    3: "metal_corrosion",
    4: "belt_cracking",
    5: "hose_deterioration",
    6: "terminal_oxidation",
    7: "rotor_grooving",
    8: "pad_wear_critical",
    9: "tire_tread_depth_low",
    10: "tire_sidewall_bulge",
    11: "structural_dent_crack",
}

FAULT_METADATA = {
    "fluid_leak_oil": {
        "severity": "HIGH",
        "action": "Check engine oil level immediately; inspect valve cover gasket and oil pan drain plug.",
    },
    "fluid_leak_coolant": {
        "severity": "HIGH",
        "action": "Pressure-test cooling system; inspect radiator seams, water pump, and thermostat housing.",
    },
    "fluid_leak_brake": {
        "severity": "CRITICAL",
        "action": "DO NOT OPERATE VEHICLE. Bleed and pressure-test hydraulic lines and caliper pistons.",
    },
    "metal_corrosion": {
        "severity": "MODERATE",
        "action": "De-scale oxidized flake; apply rust converter primer and anti-corrosion barrier coating.",
    },
    "belt_cracking": {
        "severity": "HIGH",
        "action": "Replace serpentine / timing belt immediately to prevent catastrophic valvetrain or alternator failure.",
    },
    "hose_deterioration": {
        "severity": "MODERATE",
        "action": "Squeeze test for sponginess or hardening; replace degraded rubber hose and clamp.",
    },
    "terminal_oxidation": {
        "severity": "LOW",
        "action": "Clean battery posts with neutralizing baking soda solution; coat with dielectric grease.",
    },
    "rotor_grooving": {
        "severity": "CRITICAL",
        "action": "Measure rotor thickness with vernier micrometer. If below discard thickness, replace both axle rotors.",
    },
    "pad_wear_critical": {
        "severity": "CRITICAL",
        "action": "Friction lining below safety threshold (<3mm). Replace front/rear brake pad sets immediately.",
    },
    "tire_tread_depth_low": {
        "severity": "HIGH",
        "action": "Tread wear bars flush (<1.6mm). Significant hydroplaning risk. Replace tires.",
    },
    "tire_sidewall_bulge": {
        "severity": "CRITICAL",
        "action": "Internal cord structure delaminated. Extreme risk of high-speed tire blowout. Replace tire immediately.",
    },
    "structural_dent_crack": {
        "severity": "HIGH",
        "action": "Perform ultrasonic / dye-penetrant inspection to evaluate chassis/frame fatigue integrity.",
    },
}


class FaultDetector(BaseVehicleDetector):
    """Detects and classifies physical vehicle defects."""

    def __init__(
        self,
        weights_path: Optional[Union[str, Path]] = None,
        confidence_threshold: float = 0.30,
        iou_threshold: float = 0.40,
        device: str = "cpu",
    ):
        super().__init__(
            weights_path=weights_path,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            device=device,
        )
        self._class_names = FAULT_CLASSES

    def _load_weights(self, weights_path: Path) -> None:
        """Loads YOLO weights checkpoint."""
        try:
            from ultralytics import YOLO
            self.model = YOLO(str(weights_path))
            if hasattr(self.model, "names") and self.model.names:
                self._class_names = {int(k): v for k, v in self.model.names.items()}
        except ImportError:
            raise ImportError(
                "Ultralytics is required for FaultDetector. Install with: pip install ultralytics"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load fault model checkpoint from {weights_path}: {e}")

    def predict(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
        confidence: Optional[float] = None,
    ) -> List[Dict[str, object]]:
        """
        Runs visible fault detection inference.
        Raises ModelWeightsNotFoundError if no trained weights are present.
        """
        if not self.is_loaded:
            raise ModelWeightsNotFoundError(
                f"Fault detector weights not found or not loaded (Target path: '{self.weights_path}'). "
                f"Please train the model via 'python -m ml.training.train_faults' or place "
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
                label = self._class_names.get(cls_id, f"unknown_fault_{cls_id}")
                meta = FAULT_METADATA.get(label, {"severity": "UNKNOWN", "action": "Manual inspection required."})

                detections.append({
                    "fault_id": f"fault_{cls_id}",
                    "label": label,
                    "confidence": round(score, 4),
                    "severity": meta["severity"],
                    "recommendation": meta["action"],
                    "bbox": [round(coord, 2) for coord in xyxy],
                    "class_id": cls_id,
                })

        return detections
