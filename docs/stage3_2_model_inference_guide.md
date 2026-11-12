# Stage 3.2: Model Inference & Integration Guide

**System:** AI-Powered Multimodal Vehicle Fault Detection and Diagnosis System  
**Component:** YOLOv8 Nano Three-Class Subsystem Localization Engine  
**Target Path:** `ml/weights/component_detector_best.pt`  
**API Surface:** `ml.models.component_detector.ComponentDetector`, `POST /api/v1/detect/components`  

---

## 1. Overview & Architecture

The component localization subsystem serves as the perceptual front-end of the vehicle diagnosis pipeline. It analyzes high-resolution vehicle imagery and isolates the spatial boundaries of key automotive components:

1. **`0: engine_bay`** — Full visible perimeter of the open vehicle engine compartment.
2. **`1: battery`** — Automotive 12V lead-acid or AGM starter battery.
3. **`2: radiator`** — Vehicle front cooling radiator core assembly.

This module operates independently of fault classification. It does **not** evaluate whether a component is damaged, corroded, or leaking; it solely outputs spatial coordinates, model confidence, and operational policy flags.

---

## 2. Model Checkpoint Specifications

| Metric / Attribute | Specification |
|---|---|
| **Checkpoint Path** | `ml/weights/component_detector_best.pt` |
| **Model Architecture** | Ultralytics YOLOv8n (Nano) |
| **Parameter Count** | 3,006,233 (3.01 Million) |
| **Model Size** | 6.5 MB (6,500,771 bytes) |
| **SHA-256 Digest** | `2211ADB8A4ECAB4BC91D7511C475B5EC5E3BBB3CE858762E78DD80E739346AAE` |
| **Default Input Resolution** | $640 \times 640$ pixels |
| **Class Count** | 3 (`engine_bay`, `battery`, `radiator`) |
| **Hardware Compatibility** | NVIDIA CUDA GPUs (compute capability 5.0+) and multi-core x86_64 CPUs |

---

## 3. Python Model API Usage

The detector can be utilized directly in Python scripts or Jupyter notebooks using `ml.models.component_detector.ComponentDetector`.

### 3.1 Initializing the Detector

```python
from pathlib import Path
from PIL import Image
from ml.models.component_detector import ComponentDetector

# Auto-resolves CUDA if available, falls back to CPU
detector = ComponentDetector(
    weights_path=Path("ml/weights/component_detector_best.pt"),
    confidence_threshold=0.25,
    iou_threshold=0.45,
    device="auto",  # Or explicitly "cuda:0" or "cpu"
)

print(f"Device: {detector.device}")
print(f"Classes: {detector.class_names}")
```

### 3.2 In-Memory Singleton Caching

The `ComponentDetector` maintains class-level caching (`_cached_model`, `_cached_weights_path`, `_cached_device`). When multiple instances or HTTP request handlers query the detector, model weights are loaded once into memory, preventing repeated I/O bottlenecks and VRAM thrashing.

### 3.3 Running Inference with `detect_components()`

```python
# Load image with PIL or OpenCV
img = Image.open("tests/fixtures/sample_engine.jpg")

# Run detection
result = detector.detect_components(
    image=img,
    confidence_override=0.30,  # Optional per-call override
)

print(f"Inference Latency: {result['inference_time_ms']:.2f} ms")
print(f"Total Detections: {result['total_detections']}")

for det in result["detections"]:
    print(f"\nComponent: {det['class_name']} (ID: {det['class_id']})")
    print(f"  Confidence: {det['confidence']:.4f}")
    print(f"  Pixel Box: {det['box_pixels']}")
    print(f"  Normalized Box: {det['box_normalized']}")
    print(f"  Status: {det['status']} (Experimental: {det['is_experimental']})")
    if det["warning"]:
        print(f"  Warning: {det['warning']}")
```

---

## 4. Coordinate Representation Standard

The detector outputs bounding boxes in two complementary coordinate systems:

### 4.1 Pixel Bounding Box (`box_pixels`)
Bounded integer coordinates $[x_{\min}, y_{\min}, x_{\max}, y_{\max}]$ mapped to the original image dimensions.
- Use case: Image cropping, OpenCV bounding box rendering, backend image slicing.
- Coordinates satisfy:
  $$0 \le x_{\min} \le x_{\max} \le \text{width}$$
  $$0 \le y_{\min} \le y_{\max} \le \text{height}$$

### 4.2 Normalized Bounding Box (`box_normalized`)
Normalized floating-point coordinates $[x_{\min}, y_{\min}, x_{\max}, y_{\max}]$ scaled to $[0.0, 1.0]$.
- Use case: Responsive HTML5 Canvas, SVG overlays, and React components where images resize dynamically.
- Formula:
  $$x_{\text{norm}} = \frac{x_{\text{px}}}{\text{width}}, \quad y_{\text{norm}} = \frac{y_{\text{px}}}{\text{height}}$$

---

## 5. Operational Policies & Safeguards

To prevent operational hazards and misinterpretation, all inference outputs adhere to the following policies:

### 5.1 Clear Separation from Fault Diagnosis
Component detection must never be presented as mechanical fault assessment.
- **Rule:** A high confidence score (e.g. `confidence: 0.95` on `battery`) denotes the high visual probability of an automotive battery being present, **not** that the battery is in good condition or faulty.
- **Enforcement:** Every API response bundles an unalterable `disclaimer` string.

### 5.2 Radiator Class Experimental Policy
- **Rule:** The `radiator` class has limited exposed-core training representations and achieves $0.27\text{ mAP50}$ on validation. It must always be flagged with `status: "experimental"` and `is_experimental: true`.
- **Enforcement:** Downstream UIs must display an amber warning badge requesting human technician validation.

### 5.3 Engine Bay Closed-Hood False Positive Caution
- **Rule:** Frontal closed-hood exterior views of pickup trucks can occasionally trigger false-positive engine bay detections.
- **Enforcement:** A warning note is attached to engine bay detections instructing the operator to ensure the hood is open before executing component-level diagnostics.

---

## 6. Error Handling & Troubleshooting

| Error Condition | HTTP Code | Root Cause | Solution |
|---|---|---|---|
| `ModelWeightsNotFoundError` | `503 Service Unavailable` | Model checkpoint absent at `ml/weights/component_detector_best.pt`. | Ensure checkpoint is copied from `runs/detect/stage3_1_yolov8n/weights/best.pt`. |
| `Unsupported file format` | `400 Bad Request` | Uploaded file MIME type is not JPEG, PNG, or WEBP. | Convert source imagery to JPEG or PNG before uploading. |
| `Corrupt or unreadable image file` | `400 Bad Request` | Byte payload failed PIL header verification. | Check client file transfer stream for truncation or corruption. |
| `Confidence threshold out of bounds` | `400 Bad Request` | `confidence_threshold` is outside $[0.01, 1.0]$. | Pass values such as `0.25`, `0.35`, or `0.50`. |
| `Payload exceeds limit` | `413 Content Too Large` | Image file exceeds 25 MB. | Compress or downscale image before upload. |
| `CUDA out of memory` | `500 Server Error` | GPU VRAM exhausted by other processes. | Set `DEVICE="cpu"` in `.env` to engage automatic CPU fallback. |
