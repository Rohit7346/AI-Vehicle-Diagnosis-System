# Stage 3.2: Component Detection REST API Documentation

**Service:** AI-Powered Multimodal Vehicle Diagnosis System  
**Base URL:** `http://localhost:8000/api/v1`  
**API Version:** `v1`  
**Model Architecture:** YOLOv8 Nano (`YOLOv8n-VehicleComponents-3Class`)  
**Ontology Classes:** `0: engine_bay`, `1: battery`, `2: radiator`  

---

## 1. Overview

The Component Detection API provides real-time vehicle subsystem localization from digital imagery. It is designed for frontend inspection interfaces, interactive technician tablets, and multi-stage diagnostic pipelines.

### Key Capabilities
- **Real-Time Localization:** Fast inference (~8–15ms on NVIDIA RTX GPUs, ~70–120ms on modern multi-core CPUs).
- **Dual Coordinate Systems:** Emits absolute pixel boundaries for static image manipulation and normalized $[0.0, 1.0]$ floats for responsive canvas overlays.
- **Strict Ethical Separation:** Localized components are **not** classified as defective or functional. A mandatory disclaimer is attached to every payload.
- **Safety Policy Metadata:** Incorporates operational flags (`status: "enabled"` vs. `status: "experimental"`) and warnings directly in each detection object.

---

## 2. API Endpoints

### 2.1 Detect Vehicle Components

Executes YOLOv8 object localization on an uploaded vehicle photograph.

- **Endpoint:** `POST /api/v1/detect/components`
- **Content-Type:** `multipart/form-data`
- **Authentication:** None (Internal / Development)

#### Request Parameters

| Parameter | Type | In | Required | Default | Description |
|---|---|---|---|---|---|
| `file` | `UploadFile` | `formData` | **Yes** | — | High-resolution vehicle image (`image/jpeg`, `image/png`, `image/webp`). Max size: 25 MB. |
| `confidence_threshold` | `float` | `formData` | No | `0.25` | Per-request detection confidence cutoff. Valid range: `[0.01, 1.0]`. |

#### Response: `200 OK`

```json
{
  "success": true,
  "model_name": "YOLOv8n-VehicleComponents-3Class",
  "model_version": "3.1.0",
  "device": "cuda:0",
  "inference_latency_ms": 11.82,
  "image": {
    "width": 1920,
    "height": 1080,
    "format": "JPEG"
  },
  "detections": [
    {
      "component_id": "comp_1",
      "class_id": 0,
      "class_name": "engine_bay",
      "confidence": 0.9412,
      "box_pixels": {
        "x_min": 184,
        "y_min": 95,
        "x_max": 1740,
        "y_max": 990
      },
      "box_normalized": {
        "x_min": 0.0958,
        "y_min": 0.0880,
        "x_max": 0.9062,
        "y_max": 0.9167
      },
      "status": "enabled",
      "is_experimental": false,
      "warning": "Engine bay localization identifies the visible engine compartment boundary. Known limitation: closed-hood vehicle exteriors may trigger false positives (e.g. Ford Ranger front end). Visibility verification required before proceeding with diagnostic workflows."
    },
    {
      "component_id": "comp_2",
      "class_id": 1,
      "class_name": "battery",
      "confidence": 0.8875,
      "box_pixels": {
        "x_min": 1120,
        "y_min": 310,
        "x_max": 1460,
        "y_max": 620
      },
      "box_normalized": {
        "x_min": 0.5833,
        "y_min": 0.2870,
        "x_max": 0.7604,
        "y_max": 0.5741
      },
      "status": "enabled",
      "is_experimental": false,
      "warning": null
    },
    {
      "component_id": "comp_3",
      "class_id": 2,
      "class_name": "radiator",
      "confidence": 0.4210,
      "box_pixels": {
        "x_min": 450,
        "y_min": 680,
        "x_max": 1380,
        "y_max": 920
      },
      "box_normalized": {
        "x_min": 0.2344,
        "y_min": 0.6296,
        "x_max": 0.7188,
        "y_max": 0.8519
      },
      "status": "experimental",
      "is_experimental": true,
      "warning": "Provisional detection: Radiator component model is experimental (limited training sample volume and extreme aspect ratio). Requires technician manual verification."
    }
  ],
  "summary": {
    "total_detections": 3,
    "detected_classes": [
      "battery",
      "engine_bay",
      "radiator"
    ],
    "class_counts": {
      "engine_bay": 1,
      "battery": 1,
      "radiator": 1
    }
  },
  "disclaimer": "Component localization detects physical vehicle components only and does NOT indicate mechanical faults or operational status. Detections must not be presented as fault probabilities.",
  "safeguards": {
    "engine_bay": "Engine bay localization identifies the visible engine compartment boundary. Known limitation: closed-hood vehicle exteriors may trigger false positives (e.g. Ford Ranger front end). Visibility verification required before proceeding with diagnostic workflows.",
    "radiator": "Provisional detection: Radiator component model is experimental (limited training sample volume and extreme aspect ratio). Requires technician manual verification."
  }
}
```

#### Error Responses

##### `400 Bad Request` — Unsupported MIME Type
```json
{
  "detail": "Unsupported file format 'text/plain'. Supported: JPEG, PNG, WEBP."
}
```

##### `400 Bad Request` — Corrupt Image Payload
```json
{
  "detail": "Corrupt or unreadable image file: cannot identify image file <_io.BytesIO object at 0x...>"
}
```

##### `400 Bad Request` — Invalid Confidence Value
```json
{
  "detail": "Confidence threshold must be between 0.01 and 1.0."
}
```

##### `413 Content Too Large` — File Size Exceeded
```json
{
  "detail": "Uploaded image (32.4MB) exceeds limit of 25MB."
}
```

##### `503 Service Unavailable` — Model Weights Missing
```json
{
  "detail": {
    "error": "Component detector weights uninitialized or not found on disk.",
    "message": "Component detector model weights 'ml/weights/component_detector_best.pt' not found or failed to load.",
    "action_required": "Ensure trained weights exist at 'ml/weights/component_detector_best.pt'."
  }
}
```

---

### 2.2 Component Detector Status

Retrieves model readiness, active device, class ontologies, and operational policies.

- **Endpoint:** `GET /api/v1/detect/status`
- **Content-Type:** `application/json`

#### Response: `200 OK`

```json
{
  "model_name": "YOLOv8n-VehicleComponents-3Class",
  "version": "3.1.0",
  "weights_path": "ml/weights/component_detector_best.pt",
  "weights_exist_on_disk": true,
  "loaded": true,
  "device": "cuda:0",
  "confidence_threshold": 0.25,
  "iou_threshold": 0.45,
  "classes": {
    "0": "engine_bay",
    "1": "battery",
    "2": "radiator"
  },
  "class_policies": {
    "engine_bay": {
      "class_id": 0,
      "class_name": "engine_bay",
      "status": "enabled",
      "is_experimental": false,
      "safeguard_warning": "Engine bay localization identifies the visible engine compartment boundary. Known limitation: closed-hood vehicle exteriors may trigger false positives (e.g. Ford Ranger front end). Visibility verification required before proceeding with diagnostic workflows.",
      "warning": "Engine bay localization identifies the visible engine compartment boundary. Known limitation: closed-hood vehicle exteriors may trigger false positives (e.g. Ford Ranger front end). Visibility verification required before proceeding with diagnostic workflows."
    },
    "battery": {
      "class_id": 1,
      "class_name": "battery",
      "status": "enabled",
      "is_experimental": false,
      "safeguard_warning": null,
      "warning": null
    },
    "radiator": {
      "class_id": 2,
      "class_name": "radiator",
      "status": "experimental",
      "is_experimental": true,
      "safeguard_warning": "Provisional detection: Radiator component model is experimental (limited training sample volume and extreme aspect ratio). Requires technician manual verification.",
      "warning": "Provisional detection: Radiator component model is experimental (limited training sample volume and extreme aspect ratio). Requires technician manual verification."
    }
  }
}
```

---

## 3. Client Integration Examples

### 3.1 Python (`requests`)

```python
import requests

url = "http://localhost:8000/api/v1/detect/components"
image_path = "path/to/engine_inspection.jpg"

with open(image_path, "rb") as f:
    files = {"file": ("inspection.jpg", f, "image/jpeg")}
    data = {"confidence_threshold": 0.30}
    response = requests.post(url, files=files, data=data)

if response.status_code == 200:
    result = response.json()
    print(f"Detected {result['summary']['total_detections']} components in {result['inference_latency_ms']} ms")
    for det in result["detections"]:
        print(f"- [{det['class_name'].upper()}] conf={det['confidence']:.2f} experimental={det['is_experimental']}")
        if det["warning"]:
            print(f"  Warning: {det['warning']}")
else:
    print(f"Error {response.status_code}: {response.text}")
```

### 3.2 JavaScript / TypeScript (Fetch API)

```typescript
async function detectVehicleComponents(imageFile: File, confThreshold: number = 0.25) {
  const formData = new FormData();
  formData.append("file", imageFile);
  formData.append("confidence_threshold", confThreshold.toString());

  const response = await fetch("http://localhost:8000/api/v1/detect/components", {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const errorBody = await response.json();
    throw new Error(`Detection failed (${response.status}): ${JSON.stringify(errorBody)}`);
  }

  const data = await response.json();
  return data;
}
```

### 3.3 cURL

```bash
curl -X POST "http://localhost:8000/api/v1/detect/components" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@test_engine.jpg;type=image/jpeg" \
     -F "confidence_threshold=0.25"
```
