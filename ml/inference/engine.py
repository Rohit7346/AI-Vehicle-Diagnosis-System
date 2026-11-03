"""
Diagnostic Inference Engine.
Orchestrates dual-stage component localization and visible fault identification.
Strictly relies on real model weights and never outputs fake or mocked predictions.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union
import uuid
import numpy as np
from PIL import Image

from ml.datasets.preprocessor import ImagePreprocessor
from ml.models.base_model import ModelWeightsNotFoundError
from ml.models.component_detector import ComponentDetector
from ml.models.fault_detector import FaultDetector
from ml.inference.schemas import (
    DiagnosticReport,
    DiagnosticSummary,
    ComponentDetection,
    FaultDetection,
    ImageMetadata,
    PipelineStatus,
    FaultSeverity,
)


class DiagnosticInferenceEngine:
    """Unified inference engine for vehicle subassembly and fault diagnosis."""

    def __init__(
        self,
        component_weights_path: Optional[Union[str, Path]] = None,
        fault_weights_path: Optional[Union[str, Path]] = None,
        device: str = "cpu",
        confidence_threshold: float = 0.35,
        iou_threshold: float = 0.45,
    ):
        self.component_weights_path = Path(component_weights_path) if component_weights_path else Path("ml/weights/component_detector_best.pt")
        self.fault_weights_path = Path(fault_weights_path) if fault_weights_path else Path("ml/weights/fault_detector_best.pt")
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold

        self.preprocessor = ImagePreprocessor()
        self.component_detector = ComponentDetector(
            weights_path=self.component_weights_path,
            confidence_threshold=self.confidence_threshold,
            iou_threshold=self.iou_threshold,
            device=self.device,
        )
        self.fault_detector = FaultDetector(
            weights_path=self.fault_weights_path,
            confidence_threshold=self.confidence_threshold,
            iou_threshold=self.iou_threshold,
            device=self.device,
        )

    def get_status(self) -> Dict[str, object]:
        """Returns the readiness of both model weights."""
        return {
            "component_detector": {
                "loaded": self.component_detector.is_loaded,
                "weights_path": str(self.component_weights_path),
                "weights_exist_on_disk": self.component_weights_path.exists(),
            },
            "fault_detector": {
                "loaded": self.fault_detector.is_loaded,
                "weights_path": str(self.fault_weights_path),
                "weights_exist_on_disk": self.fault_weights_path.exists(),
            },
            "device": self.device,
            "pipeline_ready": self.component_detector.is_loaded and self.fault_detector.is_loaded,
        }

    def _correlate_fault_to_component(
        self, fault_box: List[float], components: List[Dict[str, object]], min_ioa: float = 0.30
    ) -> Optional[str]:
        """
        Calculates Intersection-over-Area (IoA) to map a visible defect to its host vehicle subassembly.
        """
        fx1, fy1, fx2, fy2 = fault_box
        f_area = max(0.0, fx2 - fx1) * max(0.0, fy2 - fy1)
        if f_area <= 0:
            return None

        best_label = None
        best_ioa = 0.0

        for comp in components:
            cx1, cy1, cx2, cy2 = comp["bbox"]
            ix1 = max(fx1, cx1)
            iy1 = max(fy1, cy1)
            ix2 = min(fx2, cx2)
            iy2 = min(fy2, cy2)

            inter_w = max(0.0, ix2 - ix1)
            inter_h = max(0.0, iy2 - iy1)
            inter_area = inter_w * inter_h
            ioa = inter_area / f_area

            if ioa > best_ioa and ioa >= min_ioa:
                best_ioa = ioa
                best_label = comp["label"]

        return best_label

    def _calculate_health_score(self, faults: List[FaultDetection]) -> float:
        """
        Computes dynamic vehicle health score from detected defects.
        Starts at 100.0, deducts points based on verified defect severity.
        """
        penalty = 0.0
        for f in faults:
            if f.severity == FaultSeverity.CRITICAL:
                penalty += 35.0
            elif f.severity == FaultSeverity.HIGH:
                penalty += 18.0
            elif f.severity == FaultSeverity.MODERATE:
                penalty += 7.0
            elif f.severity == FaultSeverity.LOW:
                penalty += 2.0

        return max(0.0, round(100.0 - penalty, 1))

    def diagnose(
        self,
        image_source: Union[str, Path, bytes, np.ndarray, Image.Image],
        confidence_override: Optional[float] = None,
    ) -> DiagnosticReport:
        """
        Executes genuine end-to-end diagnosis.
        Raises ModelWeightsNotFoundError if weights are missing.
        """
        status = self.get_status()
        if not status["pipeline_ready"]:
            missing = []
            if not self.component_detector.is_loaded:
                missing.append(f"Component Detector ({self.component_weights_path})")
            if not self.fault_detector.is_loaded:
                missing.append(f"Fault Detector ({self.fault_weights_path})")
            raise ModelWeightsNotFoundError(
                f"Diagnostic inference halted: Missing model weights for: {', '.join(missing)}. "
                f"Train models using 'ml.training' or supply valid weights before running inference."
            )

        # 1. Image preprocessing
        raw_image = self.preprocessor.load_image(image_source)
        width, height = raw_image.size
        format_name = getattr(raw_image, "format", "RGB")

        # 2. Stage 1: Component detection
        raw_components = self.component_detector.predict(raw_image, confidence=confidence_override)

        # 3. Stage 2: Fault detection
        raw_faults = self.fault_detector.predict(raw_image, confidence=confidence_override)

        # 4. Multimodal correlation & aggregation
        formatted_components = [
            ComponentDetection(
                component_id=c["component_id"],
                label=c["label"],
                confidence=c["confidence"],
                bbox=c["bbox"],
            )
            for c in raw_components
        ]

        formatted_faults = []
        critical_count = 0
        high_count = 0

        for f in raw_faults:
            associated = self._correlate_fault_to_component(f["bbox"], raw_components)
            severity_enum = FaultSeverity(f["severity"]) if f["severity"] in FaultSeverity.__members__ else FaultSeverity.UNKNOWN
            
            if severity_enum == FaultSeverity.CRITICAL:
                critical_count += 1
            elif severity_enum == FaultSeverity.HIGH:
                high_count += 1

            formatted_faults.append(
                FaultDetection(
                    fault_id=f["fault_id"],
                    label=f["label"],
                    confidence=f["confidence"],
                    severity=severity_enum,
                    recommendation=f["recommendation"],
                    bbox=f["bbox"],
                    associated_component=associated,
                )
            )

        health_score = self._calculate_health_score(formatted_faults)
        summary_text = (
            f"Detected {len(formatted_components)} subsystem components and {len(formatted_faults)} visible defects. "
            f"Health Score: {health_score}/100."
        )

        return DiagnosticReport(
            inference_id=f"diag_{uuid.uuid4().hex[:12]}",
            pipeline_status=PipelineStatus.READY,
            image_metadata=ImageMetadata(
                original_width=width,
                original_height=height,
                format=format_name,
                clahe_applied=self.preprocessor.apply_clahe,
            ),
            components=formatted_components,
            faults=formatted_faults,
            summary=DiagnosticSummary(
                total_components_detected=len(formatted_components),
                total_faults_detected=len(formatted_faults),
                critical_faults_count=critical_count,
                high_faults_count=high_count,
                overall_health_score=health_score,
                status_summary=summary_text,
            ),
        )
