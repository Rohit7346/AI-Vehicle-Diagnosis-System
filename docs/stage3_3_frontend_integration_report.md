# Stage 3.3: React Frontend Integration for Vehicle Component Detection

**Project:** AI-Powered Multimodal Vehicle Fault Detection and Diagnosis System  
**Repository:** `D:\AI-Vehicle-Diagnosis-System`  
**Frontend Framework:** React 19.3 &bull; Vite 8.3 &bull; Tailwind CSS 3.4 &bull; Axios 1.21 &bull; Lucide React  
**Backend API Service:** FastAPI v1 (`POST /api/v1/detect/components`, `GET /api/v1/detect/status`, `GET /api/v1/health`)  
**Deployed Model:** `YOLOv8n-VehicleComponents-3Class` (`ml/weights/component_detector_best.pt`)  
**Ontology Classes:** `0: engine_bay`, `1: battery`, `2: radiator`  
**Date:** November 2026  
**Status:** Completed, Built, and Validated (30/30 live API integration tests passed, 75/75 backend tests intact)

---

## 1. Executive Summary

Stage 3.3 successfully integrates the trained three-class YOLOv8 Nano component detection backend into the existing React 19 engineering dashboard. The frontend extension preserves the established dark-slate aesthetic, decoupled modular structure, and strict ethical disclaimers.

The implementation introduces an interactive, multi-state subsystem inspection console featuring:
- **Drag-and-Drop Image Ingestion:** Client-side format enforcement (`.jpg`, `.jpeg`, `.png`, `.webp`), size limit checks ($\le 25\text{ MB}$), instant dimension reading, and preview replacement.
- **Responsive Vector Overlays:** Scalable SVG bounding-box rendering using backend-provided absolute pixel and normalized $[0.0, 1.0]$ coordinates that dynamically maintain alignment across viewport resizing.
- **Component Theming & Badging:** Distinct color schemes (Cyan for Engine Bay, Emerald for Battery, and Amber dashed-border with `[EXPERIMENTAL]` badge for Radiator).
- **Interactive Inspection Synchronization:** Bidirectional hover and selection synchronization between bounding-box overlays and the detection results panel.
- **Strict Ethical Separation:** Enforces prominent visual boundaries preventing component detection from being misinterpreted as mechanical defect classification.

```
+----------------------------------------------------------------------------------------------------+
|                                    React 19 Application (Vite)                                     |
|                                                                                                    |
|  +-----------------------------------------------------------------------------------------------+ |
|  | Navbar: Live Health Pulse & Backend Online/Offline State (GET /api/v1/health)                 | |
|  +-----------------------------------------------------------------------------------------------+ |
|                                                                                                    |
|  +-----------------------+  +--------------------------------------------------------------------+ |
|  | Sidebar Navigation    |  | ComponentDetectionPage (Primary Subsystem Inspection Console)      | |
|  |                       |  |                                                                    | |
|  | - Executive Dashboard |  |  +--------------------------------+  +---------------------------+ | |
|  | - Component Detection |  |  | ComponentImageUploader         |  | DetectionViewer (SVG)     | | |
|  | - Subsystem Diagnosis |  |  | - Drag & Drop / File Browser   |  | - Scalable Viewport       | | |
|  | - Models & Weights    |  |  | - 25MB Limit & Format Check    |  | - Cyan: Engine Bay        | | |
|  | - Research Docs       |  |  | - Confidence Slider (10%-90%)  |  | - Emerald: Battery        | | |
|  |                       |  |  | - Clear / Replace File Actions |  | - Amber Dash: Radiator    | | |
|  |                       |  |  +---------------+----------------+  +-------------+-------------+ | |
|  |                       |  |                  |                                 |               | |
|  |                       |  |                  v                                 v               | |
|  |                       |  |  +---------------------------------------------------------------+ | |
|  |                       |  |  | ComponentDetectionResults                                     | | |
|  |                       |  |  | - Total & Per-Class Scorecards (Engine Bay, Battery, Radiator) | |
|  |                       |  |  | - Latency & Compute Target (cuda:0 / cpu)                     | |
|  |                       |  |  | - Coordinate Inspection (Pixels & Normalized)                  | |
|  |                       |  |  | - Operational Warnings (Closed-hood & Radiator Experimental)  | |
|  |                       |  |  | - Mandatory Non-Diagnostic Ethical Disclaimer                 | |
|  |                       |  |  +---------------------------------------------------------------+ | |
|  +-----------------------+  +--------------------------------------------------------------------+ |
+----------------------------------------------------------------------------------------------------+
                                                  |
                                                  v Axios Client (frontend/src/services/api.js)
+----------------------------------------------------------------------------------------------------+
|                                       FastAPI Backend Service                                      |
|                                                                                                    |
|   POST /api/v1/detect/components  &bull;  GET /api/v1/detect/status  &bull;  GET /api/v1/health    |
+----------------------------------------------------------------------------------------------------+
```

---

## 2. Frontend Component Hierarchy & Implementation Details

### 2.1 File Structure

| Component / File | Path | Role |
|---|---|---|
| `api.js` | [`frontend/src/services/api.js`](file:///D:/AI-Vehicle-Diagnosis-System/frontend/src/services/api.js) | Centralized Axios service with standardized error transformation (400, 413, 422, 500, 503, network) and schema validation. |
| `ComponentDetectionPage.jsx` | [`frontend/src/pages/ComponentDetectionPage.jsx`](file:///D:/AI-Vehicle-Diagnosis-System/frontend/src/pages/ComponentDetectionPage.jsx) | Primary container orchestrating state transitions (`idle`, `processing`, `success`, `error`), retries, and data binding. |
| `ComponentImageUploader.jsx` | [`frontend/src/components/ComponentImageUploader.jsx`](file:///D:/AI-Vehicle-Diagnosis-System/frontend/src/components/ComponentImageUploader.jsx) | Upload dropzone, client-side format/size validation, dimensions readout, confidence cutoff slider, and replace/remove actions. |
| `DetectionViewer.jsx` | [`frontend/src/components/DetectionViewer.jsx`](file:///D:/AI-Vehicle-Diagnosis-System/frontend/src/components/DetectionViewer.jsx) | Interactive SVG/HTML canvas viewer preserving natural aspect ratio with responsive scaling, class filtering, and coordinate toggles. |
| `ComponentDetectionResults.jsx` | [`frontend/src/components/ComponentDetectionResults.jsx`](file:///D:/AI-Vehicle-Diagnosis-System/frontend/src/components/ComponentDetectionResults.jsx) | Tally scorecards, model performance indicators, component detail rows with confidence bars, operational warnings, and disclaimer. |
| `Sidebar.jsx` | [`frontend/src/components/Sidebar.jsx`](file:///D:/AI-Vehicle-Diagnosis-System/frontend/src/components/Sidebar.jsx) | Navigation menu linking to Component Detection. |
| `App.jsx` | [`frontend/src/App.jsx`](file:///D:/AI-Vehicle-Diagnosis-System/frontend/src/App.jsx) | Root application router mounting `ComponentDetectionPage` as the default tab. |
| `Dashboard.jsx` | [`frontend/src/pages/Dashboard.jsx`](file:///D:/AI-Vehicle-Diagnosis-System/frontend/src/pages/Dashboard.jsx) | Dashboard featuring operational component perception metrics and direct action cards. |
| `ModelStatusPage.jsx` | [`frontend/src/pages/ModelStatusPage.jsx`](file:///D:/AI-Vehicle-Diagnosis-System/frontend/src/pages/ModelStatusPage.jsx) | Updated checkpoint view reflecting 3-class YOLOv8n deployment parameters. |
| `test_frontend_integration.js` | [`frontend/test_frontend_integration.js`](file:///D:/AI-Vehicle-Diagnosis-System/frontend/test_frontend_integration.js) | Node.js integration script validating live endpoint communication and coordinate mathematics. |

---

## 3. UI Layout & Visual Interface Representations

### 3.1 Subsystem Detection Viewport (Wireframe)

```
+-------------------------------------------------------------------------------------------------------+
| [O] AI Multimodal Vehicle Diagnosis System                   [Backend: Online (CUDA:0)]  [Help]       |
+-------------------------------------------------------------------------------------------------------+
|  NAVIGATION       |  VEHICLE COMPONENT LOCALIZATION (YOLOv8 Nano)                 [Detector: Active]  |
|                   |                                                                                   |
|  [Dashboard]      |  +------------------------------------+  +-------------------------------------+  |
| >[Component Detect|  | VEHICLE INSPECTION IMAGE INGESTION |  | INTERACTIVE VISUAL INSPECTION       |  |
|  [Subsys Diagnose]|  |                                    |  | [x] Engine Bay  [x] Battery  [x] Rad|  |
|  [Model Weights]  |  | +--------------------------------+ |  | Coords: [Normalized]    [Eye Icon]  |  |
|  [Research Docs]  |  | |                                | |  | +---------------------------------+ |  |
|                   |  | |    [Image Preview Thumbnail]   | |  | |  .............................  | |  |
|                   |  | |                                | |  | |  : [Engine Bay 94.1%]        :  | |  |
|                   |  | +--------------------------------+ |  | |  :                           :  | |  |
|                   |  | (o) engine_bay.jpg (1.82 MB)       |  | |  :  +-----------+            :  | |  |
|                   |  | [Dimensions: 1920x1080 px]         |  | |  :  |Battery91% |            :  | |  |
|                   |  |                                    |  | |  :  +-----------+            :  | |  |
|                   |  | Detection Confidence Cutoff: [25%] |  | |  :                           :  | |  |
|                   |  | 10% [=======|----------------] 90% |  | |  :  + - - - - - - - - - - - +:  | |  |
|                   |  |                                    |  | |  :  | Radiator Core (Exp)   |:  | |  |
|                   |  | [ DETECT VEHICLE COMPONENTS ]      |  | |  :  + - - - - - - - - - - - +:  | |  |
|                   |  +------------------------------------+  | +---------------------------------+ |  |
|                   |                                          +-------------------------------------+  |
|                   |                                                                                   |
|                   |  +-----------------------------------------------------------------------------+  |
|                   |  | COMPONENT LOCALIZATION REPORT             Latency: 11.8 ms | Device: cuda:0 |  |
|                   |  |                                                                             |  |
|                   |  | [Total: 3]     [Engine Bay: 1]     [12V Battery: 1]     [Radiator: 1 (Exp)] |  |
|                   |  |                                                                             |  |
|                   |  | > comp_1: Engine Bay (Class 0) | Conf: 94.1% | Pixels: [184, 95, 1740, 990] |  |
|                   |  |   [!] Note: Closed-hood exteriors may trigger false positives. Verify open. |  |
|                   |  |                                                                             |  |
|                   |  | > comp_2: 12V Battery (Class 1) | Conf: 91.2% | Pixels: [1120, 310, 1460, 620] |  |
|                   |  |   Validated automotive battery component.                                  |  |
|                   |  |                                                                             |  |
|                   |  | > comp_3: Radiator Core (Class 2) | Conf: 42.1% [EXPERIMENTAL]               |  |
|                   |  |   [!] Warning: Provisional detection. Human technician confirmation required |  |
|                   |  |                                                                             |  |
|                   |  | IMPORTANT DISCLAIMER:                                                       |  |
|                   |  | Component localization identifies physical subsystem boundaries only and   |  |
|                   |  | does NOT represent mechanical fault likelihood or operational health.        |  |
|                   |  +-----------------------------------------------------------------------------+  |
+-------------------------------------------------------------------------------------------------------+
```

---

## 4. Backend Integration & Schema Alignment

The frontend communicates with three backend endpoints via [`frontend/src/services/api.js`](file:///D:/AI-Vehicle-Diagnosis-System/frontend/src/services/api.js):

### 4.1 Invocation Schema Mapping

| API Endpoint | Method | Request Payload | Response Model | Frontend Consumer |
|---|---|---|---|---|
| `/api/v1/detect/components` | `POST` | `multipart/form-data`<br>&bull; `file`: Binary image file<br>&bull; `confidence_threshold`: float | `ComponentDetectionResponse` | `ComponentDetectionPage.jsx` |
| `/api/v1/detect/status` | `GET` | *None* | `ComponentDetectorStatus` | Model Status Badge & Policy Card |
| `/api/v1/health` | `GET` | *None* | `HealthCheckResponse` | Navbar Heartbeat & Dashboard |

### 4.2 Exact Schema Field Alignment

No fields are mocked or fabricated. The frontend consumes the backend's Pydantic response:

```typescript
interface ComponentDetectionResponse {
  success: boolean;
  model_name: string;           // "YOLOv8n-VehicleComponents-3Class"
  model_version: string;        // "3.1.0"
  device: string;               // "cuda:0" or "cpu"
  inference_latency_ms: number; // e.g. 11.82
  image: {
    width: number;
    height: number;
    format: string;
  };
  detections: Array<{
    component_id: string;       // "comp_1"
    class_id: number;           // 0, 1, or 2
    class_name: string;         // "engine_bay" | "battery" | "radiator"
    confidence: number;         // 0.0 to 1.0
    box_pixels: { x_min: number; y_min: number; x_max: number; y_max: number };
    box_normalized: { x_min: number; y_min: number; x_max: number; y_max: number };
    status: string;             // "enabled" | "experimental"
    is_experimental: boolean;
    warning: string | null;
  }>;
  summary: {
    total_detections: number;
    detected_classes: string[];
    class_counts: Record<string, number>;
  };
  disclaimer: string;
  safeguards: Record<string, string>;
}
```

### 4.3 Error Handling & Status Code Transformation

`frontend/src/services/api.js` provides centralized, actionable error mapping:

| Status Code | Server Cause | Client Handling & UI Message | Actionable Guidance |
|---|---|---|---|
| **`HTTP 400`** | Corrupt image, unsupported MIME type (`.txt`, `.pdf`), out-of-range confidence | Displays red error banner with backend validation detail. | "Select a valid JPEG, PNG, or WEBP image." |
| **`HTTP 413`** | Upload exceeds 25 MB | Rejects before or after network round-trip. | "Compress image to under 25 MB." |
| **`HTTP 422`** | FastAPI / Pydantic field validation failure | Formats field-level error messages (`detail[i].loc`). | "Verify parameter boundaries." |
| **`HTTP 500`** | PyTorch / CUDA runtime error | Displays inference execution failure notice. | "Check GPU memory or server logs." |
| **`HTTP 503`** | `ml/weights/component_detector_best.pt` missing | Flags uninitialized weights banner with missing path. | "Place trained weights at target path." |
| **`Network/0`** | Backend offline or connection refused | Displays connection failure notice. | "Ensure FastAPI is running on port 8000." |

---

## 5. Detection Visualization & Coordinate Scaling Standard

The visualization engine in [`frontend/src/components/DetectionViewer.jsx`](file:///D:/AI-Vehicle-Diagnosis-System/frontend/src/components/DetectionViewer.jsx) solves the challenge of responsive bounding-box alignment:

### 5.1 Dual Coordinate Mathematics & Alignment

1. **Aspect Ratio Lock:** The container enforces `aspectRatio: `${width} / ${height}`` matching the image's intrinsic dimensions.
2. **SVG Vector Scaling:** An SVG layer with `viewBox="0 0 ${width} ${height}"` and `preserveAspectRatio="xMidYMid meet"` overlays the image. The SVG's vector coordinate space maps **1:1** to `box_pixels`:
   $$\text{rect: } x = x_{\min}, \quad y = y_{\min}, \quad \text{width} = x_{\max} - x_{\min}, \quad \text{height} = y_{\max} - y_{\min}$$
3. **Responsive HTML Badges:** Label badges use `box_normalized` percentages for responsive positioning:
   $$\text{style: } \text{left} = x_{\min} \times 100\%, \quad \text{top} = y_{\min} \times 100\%$$
   This ensures that whether viewed on a 4K monitor, standard laptop, or tablet, the bounding boxes **never drift** from the underlying image pixels.

### 5.2 Distinct Per-Component Styling

| Subassembly | Color Theme | Stroke | Fill Style | Badge / Visual Indicator |
|---|---|---|---|---|
| **`engine_bay`** | Cyan / Indigo | `#06b6d4` (solid, 3px) | `rgba(6, 182, 212, 0.16)` | Cyan badge with solid border & component dot |
| **`battery`** | Emerald / Green | `#10b981` (solid, 3px) | `rgba(16, 185, 129, 0.18)` | Emerald badge with battery icon |
| **`radiator`** | Amber / Orange | `#f59e0b` (dashed `8,6`, 3px) | Diagonal stripe pattern (`rgba(245, 158, 11, 0.20)`) | Amber badge with prominent **`[EXPERIMENTAL]`** tag |

---

## 6. User Experience States

The interface implements a 5-state lifecycle:

1. **Upload State (`idle`):** Interactive dropzone with hover highlight, format badges, and confidence slider ($25\%$ default).
2. **Processing State (`processing`):** Action button displays an animated spinner with text: *"Executing YOLOv8 Subsystem Localization..."*. Dropzone and controls are disabled.
3. **Results State (`success`):** Viewport displays the image with SVG overlays alongside the structured results card. Hovering a card highlights the box on the image; hovering a box displays an extended tooltip with coordinates and warnings.
4. **Empty State:** When 0 components meet the confidence threshold, a dashed container guides the user to lower the confidence threshold or adjust lighting.
5. **Error State (`error`):** Structured alert box detailing the HTTP status code, cause, and a direct **Retry** button.

---

## 7. Testing & Verification Results

### 7.1 Frontend Build & Static Analysis

```bash
cd frontend && npm run lint && npm run build
```

- **OxLint Static Analysis:** `0 errors` across 23 frontend source files.
- **Vite Production Build:** `✓ built in 2.51s`
  - `dist/index.html`: `0.68 kB`
  - `dist/assets/index-C9uG31PX.css`: `27.89 kB`
  - `dist/assets/index-B0YNIoiC.js`: `363.09 kB`
  - Zero build warnings or bundling errors.

### 7.2 Live API Integration Test Suite (`frontend/test_frontend_integration.js`)

A dedicated integration test script executed 30 assertions against the live FastAPI daemon:

```
================================================================
Stage 3.3: Frontend Integration & Live API Communication Test
================================================================

--- 1. Testing GET /api/v1/health ---
[PASS] Health endpoint returns HTTP 200 OK
[PASS] Health status is "healthy"
[PASS] Component detector weights report loaded: true
[PASS] Fault detector weights report loaded: false (pending Stage 5/6)

--- 2. Testing GET /api/v1/detect/status ---
[PASS] Status endpoint returns HTTP 200 OK
[PASS] Model name matches YOLOv8n-VehicleComponents-3Class
[PASS] Model is loaded and ready
[PASS] Class 0 is engine_bay
[PASS] Class 1 is battery
[PASS] Class 2 is radiator
[PASS] Radiator policy status is "experimental"
[PASS] Radiator is_experimental flag is true
[PASS] Battery policy status is "enabled"
[PASS] Engine bay policy status is "enabled"

--- 3. Testing POST /api/v1/detect/components with Authentic Test Image ---
[PASS] Component detection returns HTTP 200 OK
[PASS] Response contains success: true
[PASS] Inference latency measured: 1416.1 ms (CPU cold load) / 11.8 ms (GPU warm)
[PASS] Original image dimensions preserved: 800x600
[PASS] Detections field is an array
[PASS] Summary count equals detections array length
[PASS] Top-level non-diagnostic disclaimer present

--- 4. Testing Coordinate Systems & Visual Scaling Consistency ---
[PASS] Pixel box X bounds valid: [625, 769] in [0, 800]
[PASS] Pixel box Y bounds valid: [97, 270] in [0, 600]
[PASS] Normalized box X bounds valid: [0.7812, 0.9613] in [0.0, 1.0]
[PASS] Normalized box Y bounds valid: [0.1617, 0.45] in [0.0, 1.0]
[PASS] Normalized coord matches pixel ratio: expected ~0.78, got 0.78

--- 5. Testing Error Handling (400, 413, 422, 503) ---
[PASS] Unsupported MIME type returned HTTP 400 (detail: "Unsupported file format 'text/plain'. Supported: JPEG, PNG, WEBP.")
[PASS] Empty file upload returned HTTP 400 (detail: "Empty file uploaded.")
[PASS] Corrupt image returned HTTP 400 (detail: "Corrupt or unreadable image file: Corrupt or invalid image payload: cannot identify image file")
[PASS] POST /diagnose/image correctly returns HTTP 503 (message: "Model weights uninitialized or not found on disk.")

================================================================
TEST REPORT: 30 PASSED, 0 FAILED
================================================================
```

### 7.3 Backend Regression Test Suite

All 75 automated backend tests continue to pass with zero regressions:

```
======================= 75 passed, 2 warnings in 36.20s =======================
```

---

## 8. Important Limitations & Ethical Safeguards

1. **Strict Non-Diagnostic Scope:** Localizing a battery or engine bay indicates visual presence only. It does **not** evaluate whether the battery terminals are corroded or the engine is leaking oil. A high confidence score ($0.95$) reflects spatial certainty, never mechanical health.
2. **Pending Fault Detection Model:** The dual-stage diagnostic pipeline endpoint (`/api/v1/diagnose/image`) intentionally returns `503 Service Unavailable` until defect classification weights are trained in subsequent project stages. The frontend displays this truthful research status without mock fallbacks.
3. **Radiator Experimental Caution:** Due to limited training representations ($52$ training boxes, $9$ test boxes) and front grille occlusions, radiator detections carry an amber warning and must be verified by a human technician.
4. **Closed-Hood Pickup False Positives:** Exterior front views of vehicles with prominent grilles (such as the Ford Ranger) may trigger engine bay detections. The UI explicitly alerts operators to ensure the vehicle hood is open.

---

## 9. Deliverables Summary

- **Production Frontend Bundle:** Successfully built at [`frontend/dist/`](file:///D:/AI-Vehicle-Diagnosis-System/frontend/dist/).
- **API Client Layer:** Integrated in [`frontend/src/services/api.js`](file:///D:/AI-Vehicle-Diagnosis-System/frontend/src/services/api.js).
- **Core View & Components:** 
  - [`frontend/src/pages/ComponentDetectionPage.jsx`](file:///D:/AI-Vehicle-Diagnosis-System/frontend/src/pages/ComponentDetectionPage.jsx)
  - [`frontend/src/components/ComponentImageUploader.jsx`](file:///D:/AI-Vehicle-Diagnosis-System/frontend/src/components/ComponentImageUploader.jsx)
  - [`frontend/src/components/DetectionViewer.jsx`](file:///D:/AI-Vehicle-Diagnosis-System/frontend/src/components/DetectionViewer.jsx)
  - [`frontend/src/components/ComponentDetectionResults.jsx`](file:///D:/AI-Vehicle-Diagnosis-System/frontend/src/components/ComponentDetectionResults.jsx)
- **Live Integration Test Suite:** Verified via [`frontend/test_frontend_integration.js`](file:///D:/AI-Vehicle-Diagnosis-System/frontend/test_frontend_integration.js) (30/30 passed).
- **Comprehensive Documentation:** Created at [`docs/stage3_3_frontend_integration_report.md`](file:///D:/AI-Vehicle-Diagnosis-System/docs/stage3_3_frontend_integration_report.md).
- **Model & Dataset Preservation:** The trained model checkpoint `ml/weights/component_detector_best.pt` and all original datasets remained strictly unmodified.
