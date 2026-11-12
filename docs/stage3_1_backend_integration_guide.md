# Stage 3.1: Backend Integration Blueprint & Readiness Guide

**Project Title:** AI-Powered Multimodal Vehicle Fault Detection and Diagnosis System  
**Stage:** Stage 3.1 – Backend Integration Preparation  
**Target Module:** FastAPI Diagnostics Engine (`backend/app/services/diagnosis_service.py`, `ml/inference/engine.py`)  
**Trained Model Checkpoint:** [`runs/detect/stage3_1_yolov8n/weights/best.pt`](file:///D:/AI-Vehicle-Diagnosis-System/runs/detect/stage3_1_yolov8n/weights/best.pt)  
**Date:** November 11, 2026  

---

## 1. Overview & Integration Boundaries

Stage 3.1 produces the first functional weights for the `ComponentDetector` subsystem. Per project guidelines, **the existing backend model weights are not replaced in this stage**. This document details the exact integration blueprint for Stage 3.2, ensuring seamless connectivity between the FastAPI REST API, the diagnostic inference engine, and the YOLOv8 Nano model.

---

## 2. Checkpoint Provenance & Target Deployment Location

- **Trained Best Checkpoint:**
  `D:\AI-Vehicle-Diagnosis-System\runs\detect\stage3_1_yolov8n\weights\best.pt`
- **Standard Backend Production Path:**
  `D:\AI-Vehicle-Diagnosis-System\ml\weights\component_detector_best.pt`
- **Configuration Hook:**
  Defined in `backend/app/core/config.py`:
  ```python
  COMPONENT_MODEL_WEIGHTS: Path = Path("ml/weights/component_detector_best.pt")
  ```

---

## 3. Model Loading & Lifecycle Orchestration

In `ml/models/component_detector.py`, `ComponentDetector` manages model loading and inference:

```python
from ultralytics import YOLO

class ComponentDetector(BaseVehicleDetector):
    def _load_weights(self, weights_path: Path) -> None:
        self.model = YOLO(str(weights_path))
        if hasattr(self.model, "names") and self.model.names:
            self._class_names = {int(k): v for k, v in self.model.names.items()}
```

The model automatically synchronizes its internal `_class_names` dictionary with the 3 classes embedded within `best.pt`:
```python
{0: "engine_bay", 1: "battery", 2: "radiator"}
```

---

## 4. Input Preprocessing & Image Specification

The inference pipeline accepts input via FastAPI endpoint `POST /api/v1/diagnose/image`:
- **Format:** Multipart image upload (`image/jpeg`, `image/png`, `image/webp`).
- **Input Dimension:** Native image decoded via PIL / OpenCV, letterbox-resized to `640 x 640` maintaining original aspect ratio.
- **Normalization:** Automatically handled by YOLOv8 ($[0, 255] \rightarrow [0.0, 1.0]$ FP16 tensor).
- **Execution Device:** `cuda:0` when available, seamlessly falling back to `cpu` if CUDA hardware is uninitialized.

---

## 5. Output Prediction Schema & Response Harmonization

The model predictions map directly to Pydantic schemas in `ml/inference/schemas.py`:

```json
{
  "report_id": "f5a2b109-77c8-4e89-b76f-cb9127521e05",
  "status": "success",
  "timestamp": "2026-11-11T16:30:00Z",
  "components_detected": [
    {
      "component_id": "comp_001",
      "class_id": 0,
      "label": "engine_bay",
      "confidence": 0.924,
      "bounding_box": [0.042, 0.125, 0.958, 0.887]
    },
    {
      "component_id": "comp_002",
      "class_id": 1,
      "label": "battery",
      "confidence": 0.881,
      "bounding_box": [0.650, 0.280, 0.840, 0.490]
    },
    {
      "component_id": "comp_003",
      "class_id": 2,
      "label": "radiator",
      "confidence": 0.742,
      "bounding_box": [0.150, 0.810, 0.820, 0.895]
    }
  ],
  "summary": {
    "total_components": 3,
    "confidence_average": 0.849,
    "primary_subsystem": "engine_bay"
  }
}
```

---

## 6. Threshold Configuration Recommendations

| Parameter | Recommended Baseline | Operational Rationale |
|---|:---:|---|
| **Confidence Threshold (`conf`)** | `0.30` | Filters weak edge hallucinations while capturing radiators with subtle grill texture |
| **IoU Threshold (`iou`)** | `0.45` | Standard Non-Maximum Suppression (NMS) threshold to eliminate duplicate bounding boxes |
| **Max Detections (`max_det`)** | `50` | Supports dense subassembly detection scenes |

---

## 7. Migration Checklist for Stage 3.2

1. Verify `runs/detect/stage3_1_yolov8n/weights/best.pt` file size and checksum.
2. Copy `best.pt` to `ml/weights/component_detector_best.pt`.
3. Update `backend/app/core/config.py` default settings.
4. Run integration test suite (`tests/test_inference_engine.py` and `tests/test_api_endpoints.py`) to confirm that `test_diagnose_image_missing_weights_returns_503` transitions to a functional `200 OK` inference test.
