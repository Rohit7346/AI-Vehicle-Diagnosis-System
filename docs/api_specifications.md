# Backend REST API Specifications

The backend service is powered by FastAPI and provides an asynchronous REST interface for inspection ingest, diagnosis retrieval, and model health diagnostics.

---

## Base URL
```
http://localhost:8000/api/v1
```

---

## Endpoints

### 1. System & Model Health Status
- **Route:** `GET /health`
- **Description:** Returns operational readiness, environment flags, CUDA acceleration status, and whether trained weights are loaded for component and fault detectors.
- **Response `200 OK`:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "environment": "development",
  "system": {
    "python_version": "3.13.3",
    "cuda_available": false,
    "device": "cpu"
  },
  "models": {
    "component_detector": {
      "loaded": false,
      "weights_path": "ml/weights/component_detector_best.pt"
    },
    "fault_detector": {
      "loaded": false,
      "weights_path": "ml/weights/fault_detector_best.pt"
    }
  }
}
```

---

### 2. Image Upload & Multimodal Diagnosis
- **Route:** `POST /diagnose/image`
- **Content-Type:** `multipart/form-data`
- **Parameters:**
  - `file`: Image file (`image/jpeg`, `image/png`, `image/webp`). Max: 25MB.
  - `confidence_threshold` *(optional, float, default: 0.35)*: Minimum confidence threshold.
  - `iou_threshold` *(optional, float, default: 0.45)*: Non-maximum suppression overlap threshold.
- **Response `200 OK`:**
```json
{
  "success": true,
  "inference_id": "diag_a8c2f1b4",
  "timestamp": "2026-11-03T18:50:00Z",
  "image_metadata": {
    "filename": "brake_inspection_01.jpg",
    "width": 1920,
    "height": 1080,
    "format": "JPEG"
  },
  "pipeline_status": "READY",
  "components_detected": [
    {
      "label": "brake_rotor",
      "confidence": 0.942,
      "bbox": [120, 240, 890, 960]
    }
  ],
  "faults_detected": [
    {
      "label": "rotor_grooving",
      "confidence": 0.887,
      "severity": "CRITICAL",
      "associated_component": "brake_rotor",
      "bbox": [250, 400, 520, 680],
      "recommendation": "Inspect disc runout with dial indicator. Resurface or replace disc immediately."
    }
  ],
  "overall_health_score": 42.0,
  "executive_summary": "1 critical fault localized on brake rotor."
}
```

- **Error Responses:**
  - `400 Bad Request`: Invalid image format or corrupt file.
  - `413 Payload Too Large`: Upload exceeds maximum allowed file size.
  - `503 Service Unavailable`: Model weights are not initialized or missing from disk (with actionable instructions).
