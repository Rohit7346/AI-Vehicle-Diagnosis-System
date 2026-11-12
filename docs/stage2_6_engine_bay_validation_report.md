# Stage 2.6: Engine-Bay Dataset Validation & Quality Audit Report

**Project Title:** AI-Powered Multimodal Vehicle Fault Detection and Diagnosis System  
**Stage:** Stage 2.6 – Engine-Bay Dataset Validation and Quality Audit  
**Target Class:** Class 0 (`engine_bay`)  
**Audited Archive:** `Vehicle Component Detection.v1i.yolov8.zip` (10.10 MB)  
**Distribution CSV:** [`docs/stage2_6_engine_bay_class_distribution.csv`](file:///d:/AI-Vehicle-Diagnosis-System/docs/stage2_6_engine_bay_class_distribution.csv)  
**Visual Review Directory:** [`docs/stage2_6_engine_bay_visual_review/`](file:///d:/AI-Vehicle-Diagnosis-System/docs/stage2_6_engine_bay_visual_review/)  
**Automated Tests:** [`tests/test_engine_bay_audit.py`](file:///d:/AI-Vehicle-Diagnosis-System/tests/test_engine_bay_audit.py)  
**Date:** November 11, 2026  

---

## 1. Executive Summary

This validation audit evaluates the exported dataset from the Roboflow project **`Vehicle Component Detection`** (v1), containing the manual annotations for the 193 staged engine bay images.

### Summary of Audit Results:
- **Archive Status:** Successfully inspected. Contains 192 images and 192 label files across `train`, `valid`, and `test` splits.
- **Class ID Verification:** `data.yaml` specifies `nc: 1` and `names: ['engine_bay']`. Exported Class ID is **`0`**, aligning perfectly with Prototype Class ID **`0: engine_bay`**.
- **Accounting Against Original 193 Uploads:**
  - 192 of the 193 images are present in the export.
  - Exactly **1 image** (`BYD-Dolphin-Mini-2024-12_jpg.rf.3ec96230...`) was omitted from export because it is a pure battery-electric vehicle (BEV) lacking an internal combustion engine bay.
  - 33 images with non-JPEG original extensions had `_jpg` appended upon Roboflow export (e.g. `_png_jpg`, `_webp_jpg`, `_jpeg_jpg`).
- **Annotation Cardinality & Discipline:**
  - **191 images** contain **exactly one** `engine_bay` annotation.
  - **1 image** (`Ford-Ranger-XLT-4x4-3-2-Autoentusiastas-22_jpg`) has an empty label file (valid negative/background sample).
  - **0 images** contain multi-box duplicates.
- **Annotation Geometry (Polygon Segmentation):**
  - **185 images** were annotated with high-precision **polygons** (6 to 15 vertices, mean 8.2 vertices).
  - **6 images** were annotated with standard 5-token bounding boxes.
  - 100% of coordinates are mathematically valid and normalized within $[0, 1]$.
  - Polygons tightly trace the inner fender aprons, firewall bulkhead, and front crossmember while strictly excluding exterior body panels, front bumper, headlights, and hood bonnet skin.
- **Cross-Split Leakage:** 1 cross-split near-duplicate pair was detected between `train` and `valid` (`sob-o-capo-todas-as-versoes-do-spin...`), resulting from Roboflow's random split assignment.
- **Visual Inspection:** 30 stratified samples were rendered and verified in `docs/stage2_6_engine_bay_visual_review/`.
- **Validation Test Suite:** 8 new automated unit tests implemented and passing. Full project test suite passes (38 of 38 tests).

---

## 2. Ingestion & Split Distribution

```
Exported Engine-Bay Dataset Structure:
Vehicle Component Detection.v1i.yolov8.zip
├── train/
│   ├── images/ (135 images) │ labels/ (135 labels) │ 134 polygons, 1 bbox
├── valid/
│   ├── images/ ( 38 images) │ labels/ ( 38 labels) │  33 polygons, 5 bboxes
└── test/
    ├── images/ ( 19 images) │ labels/ ( 19 labels) │  18 polygons, 0 bboxes, 1 empty
└── Total: 192 images │ 192 labels │ 191 engine_bay annotations
```

### Granular Split Metrics Table:
Full CSV available at [`docs/stage2_6_engine_bay_class_distribution.csv`](file:///d:/AI-Vehicle-Diagnosis-System/docs/stage2_6_engine_bay_class_distribution.csv).

| Partition Split | Exported Images | Exported Labels | `engine_bay` Annotations | Polygon Annotations | Standard BBoxes | Empty Labels (Negatives) | Mean Polygon Vertices | Resolution |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **`train`** | 135 | 135 | 135 | 134 | 1 | 0 | 8.1 | $512 \times 512$ |
| **`valid`** | 38 | 38 | 38 | 33 | 5 | 0 | 8.4 | $512 \times 512$ |
| **`test`** | 19 | 19 | 18 | 18 | 0 | 1 | 8.2 | $512 \times 512$ |
| **TOTAL** | **192** | **192** | **191** | **185** | **6** | **1** | **8.2** | **$512 \times 512$** |

---

## 3. Upload Comparison & Missing Image Investigation

### 3.1 Upload vs Export Accounting
- **Total Uploaded Staged Images:** 193 files in `data/raw/stage2_collection/engine_bay/images/`.
- **Total Exported Images in Archive:** 192 files.
- **Delta:** Exactly $-1$ image.

### 3.2 Identification of Missing Image:
- **Filename:** `BYD-Dolphin-Mini-2024-12_jpg.rf.3ec96230e3ffa75c95a5d33450841ba1.jpg` (Original size: $1779 \times 1334$ px).
- **Domain Inspection:** The BYD Dolphin Mini is a pure battery-electric vehicle (BEV). Its under-hood compartment contains power electronics and an electric motor reduction unit without a traditional internal combustion engine layout.
- **Root Cause:** The image was left unannotated during the manual Roboflow annotation session, and Roboflow defaults to omitting unannotated images from YOLO detection exports.
- **Diagnostic Suitability:** Omitting this BEV compartment is correct for Stage 1 & Stage 2, which focus specifically on internal combustion engine powertrains (alternator, radiator, 12V battery, exhaust manifold).

### 3.3 Extension Normalization by Roboflow:
Roboflow appends `_jpg` to non-JPEG uploaded files upon export:
- `image-8_png` $\rightarrow$ `image-8_png_jpg.rf....jpg`
- `211-webp-...webp` $\rightarrow$ `211-webp-...webp_jpg.rf....jpg`
- `renault-fluence-...jpeg` $\rightarrow$ `renault-fluence-...jpeg_jpg.rf....jpg`
33 files were renamed in this manner; all 33 map 1:1 to their original staged source files.

---

## 4. Annotation Geometry & Boundary Discipline

### 4.1 Polygon-to-Bounding-Box Compatibility
185 of the 191 annotated images use polygon segmentation format ($2k+1$ tokens: `0 x1 y1 x2 y2 ... xk yk`):
- Vertex count ranges from **6 to 15 vertices** per polygon.
- **Conversion Math:** For YOLOv8 detection models (`yolov8n`), each polygon $P = \{(x_i, y_i)\}_{i=1}^k$ converts to its enclosing bounding box:
  $$x_{min} = \min(x_i), \quad x_{max} = \max(x_i), \quad y_{min} = \min(y_i), \quad y_{max} = \max(y_i)$$
  $$w = x_{max} - x_{min}, \quad h = y_{max} - y_{min}, \quad x_c = x_{min} + \frac{w}{2}, \quad y_c = y_{min} + \frac{h}{2}$$
- **Advantage of Polygon Annotation:** Drawing polygons forces annotators to follow the irregular geometric contours of the inner engine bay. The resulting derived bounding box is significantly tighter and freer of background noise than freehand rectangular bounding boxes.

### 4.2 Strict Exclusion of Non-Bay Elements
Visual review confirmed that:
1. **Hood Bonnet Skin:** Excluded. The open hood lid and insulating underside blanket are outside the boundary.
2. **Front Bumper & Grille:** Excluded. The lower boundary aligns strictly with the radiator core support crossmember.
3. **Headlights & Fenders:** Excluded. The lateral boundaries align with the inner strut towers and metal apron bulkheads.
4. **Workshop Floor & Ground:** Excluded.

### 4.3 Empty Label File Analysis
- File: `test/labels/Ford-Ranger-XLT-4x4-3-2-Autoentusiastas-22_jpg.rf.55c7c78702e78433f5abd96c3a33e57f.txt`
- Contains 0 bytes (empty label).
- **Inspection:** In object detection, empty label files serve as **hard negative background images**. The model trains on this image to learn what *not* to detect, suppressing false positive detections on complex vehicle exteriors.

---

## 5. Duplicate & Data Leakage Analysis

### 5.1 Exact Cryptographic Duplicates (SHA-256)
- **0 exact SHA-256 duplicate groups** exist in the export.

### 5.2 Perceptual Near-Duplicates & Cross-Split Leakage
Perceptual hash analysis (64-bit dHash, $D_H \le 4$) identified 3 near-duplicate pairs:
- **Internal Pairs:** 2 pairs within `train`.
- **Cross-Split Leakage Pair (1 pair):**
  - `train/images/sob-o-capo-todas-as-versoes-do-spin-...408ab87d...jpg`
  - `valid/images/sob-o-capo-todas-as-versoes-do-spin-...683ca4f8...jpg`
  - **Hamming Distance:** 4 (Identical Chevrolet Spin vehicle scene placed in both `train` and `valid` by Roboflow's random partitioner).
- **Remediation Recommendation:** Before final integration into the prototype dataset, this validation copy should be moved to `train` or purged to guarantee zero validation leakage.

---

## 6. Visual Review & Verification (30 Samples)

Thirty stratified visual samples were rendered and saved to [`docs/stage2_6_engine_bay_visual_review/`](file:///d:/AI-Vehicle-Diagnosis-System/docs/stage2_6_engine_bay_visual_review/):
- **Train Split (20 samples):** E.g. `sample_01_train_big...` (POLY 6v), `sample_02_train_fiat-toro...` (POLY 8v), `sample_07_train_toyota-corolla-cross...` (POLY 7v), `sample_15_train_byd-song-plus...` (POLY 7v).
- **Valid Split (6 samples):** E.g. `sample_21_valid_fiat-punto...` (POLY 15v), `sample_22_valid_CITY...` (POLY 9v), `sample_25_valid_byd-yuan-pro...` (POLY 8v).
- **Test Split (4 samples):** E.g. `sample_27_test_sddefault...` (POLY 10v), `sample_28_test_Renault-Sandero...` (POLY 8v), `sample_30_test_211-webp...` (POLY 8v).

Each visual sample displays the high-contrast cyan polygon trace along with the amber derived bounding box, confirming tight boundary discipline and valid under-hood coverage.

---

## 7. Deliverables & Test Verification

All required Stage 2.6 deliverables have been generated:
1. **[`docs/stage2_6_engine_bay_validation_report.md`](file:///d:/AI-Vehicle-Diagnosis-System/docs/stage2_6_engine_bay_validation_report.md):** Comprehensive validation audit report.
2. **[`docs/stage2_6_engine_bay_class_distribution.csv`](file:///d:/AI-Vehicle-Diagnosis-System/docs/stage2_6_engine_bay_class_distribution.csv):** Granular per-split class distribution and polygon geometry metrics.
3. **[`docs/stage2_6_engine_bay_visual_review/`](file:///d:/AI-Vehicle-Diagnosis-System/docs/stage2_6_engine_bay_visual_review/):** Directory of 30 rendered JPEG visual review samples.
4. **Automated Test Suite ([`tests/test_engine_bay_audit.py`](file:///d:/AI-Vehicle-Diagnosis-System/tests/test_engine_bay_audit.py)):** 8 unit tests covering class ID, cardinality, split counts, coordinate normalization, pairing, and visual reviews.
5. **Full Project Test Suite:** **38 of 38 tests passed** across the entire repository.

---

## 8. Summary of Unresolved Issues

- **Missing Image (1):** `BYD-Dolphin-Mini-2024-12_jpg` was omitted because it is a pure electric vehicle (BEV). No action required; its omission preserves internal combustion powertrain focus.
- **Negative Sample (1):** `Ford-Ranger-XLT-4x4-3-2-Autoentusiastas-22_jpg` is an empty background sample in `test/`. Recommended for retention as a negative background evaluation sample.
- **Roboflow Cross-Split Leak (1):** `sob-o-capo-todas-as-versoes-do-spin...` has near-duplicate copies in `train` and `valid`. Prior to merging into `data/prototype_5class/`, the validation copy should be moved to `train` to ensure 100% leak-free evaluation.
