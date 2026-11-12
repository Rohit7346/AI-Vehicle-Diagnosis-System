# Stage 2.5: Battery Dataset Audit & Multi-Source Verification

**Project Title:** AI-Powered Multimodal Vehicle Fault Detection and Diagnosis System  
**Stage:** Stage 2.5 – Battery Dataset Audit  
**Component Focus:** Class 1 (`battery`)  
**Auditor Engine:** `tests/test_battery_audit.py` & `ml/datasets/audit_staged_images.py`  
**Distribution CSV:** [`docs/stage2_5_battery_class_distribution.csv`](file:///d:/AI-Vehicle-Diagnosis-System/docs/stage2_5_battery_class_distribution.csv)  
**Visual Review Directory:** [`docs/dataset_review/stage2_5_battery/`](file:///d:/AI-Vehicle-Diagnosis-System/docs/dataset_review/stage2_5_battery/)  
**Date:** November 11, 2026  

---

## 1. Executive Summary

This audit evaluates battery-related data from two public candidate YOLOv8 dataset archives downloaded from Roboflow:
1. **`Engine Parts Detector.v1i.yolov8.zip`** (64.30 MB, 804 entries)
2. **`engine-bay-parts.v1i.yolov8.zip`** (53.91 MB, 402 entries)

A third archive (`Vehicle Component Detection.v1i.yolov8.zip`, 9.64 MB) was inspected and identified as an exported single-class (`engine_bay`) dataset; per specifications, it was excluded from this battery-focused audit.

### Core Audit Findings:
- **Battery Class ID Identification:**
  - Source 1 (`Engine Parts Detector v1`): Battery is **Class ID 1** (`Battery`).
  - Source 2 (`engine-bay-parts v1`): Battery is **Class ID 2** (`Battery`).
- **Total Battery Candidates Available:** **490 unique images** containing **492 bounding boxes** across both sources.
- **Image-to-Label Pairing:** **100% paired** (0 missing label files, 0 orphan labels) across both archives.
- **Normalized Coordinates:** 100% of bounding boxes satisfy $x, y \in [0, 1]$ and $w, h \in (0, 1]$.
- **Automotive Domain Relevance:** 100% of inspected battery images depict authentic under-hood automotive 12V lead-acid / AGM starter batteries. Zero non-automotive consumer batteries (e.g. AA/AAA, smartphone, power tool) were found.
- **Defects & Anomalies Detected:**
  - **Annotation Corruption:** 1 file in Source 1 (`train/labels/istockphoto-1169897349-1024x1024_jpg.rf.0a3a3ff93316acb8b0c63664769318da.txt`) contains an 11-token segmentation polygon line instead of the standard 5-token YOLO bbox format.
  - **Cross-Split Duplicate Leakage:** Source 1 contains 1 exact SHA-256 duplicate pair leaking across `train` and `valid` (`istockphoto-953763812`), plus 19 cross-split near-duplicate pairs ($D_H \le 4$).
  - **Split Imbalance in Source 2:** Source 2 contains only a `train` split (200 images total, 127 battery images); `valid` and `test` splits are completely empty ($0$ images).
  - **Multi-Panel Collage Images:** 2 images in Source 2 (`819_jpg` and `602_jpg`) are multi-image stitched collages containing 2 batteries each.

---

## 2. Dataset Provenance & Class Identification

| Source Archive | Roboflow Workspace | Project Slug | Version | License | Target Class Name | Source Class ID | Prototype Class ID |
|---|---|---|:---:|:---:|---|:---:|:---:|
| `Engine Parts Detector.v1i.yolov8.zip` | `rohits-workspace-4tw8c` | `engine-parts-detector-wqzdk` | v1 | **CC BY 4.0** | `Battery` | **1** | **1** |
| `engine-bay-parts.v1i.yolov8.zip` | `rohits-workspace-4tw8c` | `engine-bay-parts-agjnc` | v1 | **MIT** | `Battery` | **2** | **1** |

### Class Ontologies in Sources:
- **Source 1 (6 Classes):** `0: Air Coolant`, `1: Battery`, `2: Fuse-Box`, `3: Main Engine`, `4: Radiator`, `5: Windscreen Washer`.
- **Source 2 (12 Classes):** `0: Air Filter Cover`, `1: Alternator`, `2: Battery`, `3: Engine Coolant Reservoir`, `4: Engine Cover`, `5: Engine Oil Dip Stick`, `6: Engine Oil Fill Cap`, `7: Fuse Box`, `8: Oil Filter Housing`, `9: Radiator`, `10: Radiator Cap`, `11: Windshield Wiper Fluid`.

---

## 3. Battery Distribution by Split

```
Candidate Battery Data Distribution:
├── Source 1: Engine Parts Detector v1 (CC BY 4.0)
│   ├── train:   265 total images  │  236 battery images  │  236 battery bboxes
│   ├── valid:    86 total images  │   84 battery images  │   84 battery bboxes
│   ├── test:     49 total images  │   43 battery images  │   43 battery bboxes
│   └── Total:   400 total images  │  363 battery images  │  363 battery bboxes
└── Source 2: engine-bay-parts v1 (MIT)
    ├── train:   200 total images  │  127 battery images  │  129 battery bboxes (2 dual)
    ├── valid:     0 total images  │    0 battery images  │    0 battery bboxes
    ├── test:      0 total images  │    0 battery images  │    0 battery bboxes
    └── Total:   200 total images  │  127 battery images  │  129 battery bboxes
```

### Statistical Class Distribution Table:
Full CSV record available in [`docs/stage2_5_battery_class_distribution.csv`](file:///d:/AI-Vehicle-Diagnosis-System/docs/stage2_5_battery_class_distribution.csv).

| Source | Split | Total Images | Total Labels | Battery Images | Battery BBoxes | Battery Prevalence | Background Images (No Battery) |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Engine Parts Detector v1** | `train` | 265 | 265 | 236 | 236 | 89.06% | 29 |
| | `valid` | 86 | 86 | 84 | 84 | 97.67% | 2 |
| | `test` | 49 | 49 | 43 | 43 | 87.76% | 6 |
| | **Subtotal** | **400** | **400** | **363** | **363** | **90.75%** | **37** |
| **engine-bay-parts v1** | `train` | 200 | 200 | 127 | 129 | 63.50% | 73 |
| | `valid` | 0 | 0 | 0 | 0 | N/A | 0 |
| | `test` | 0 | 0 | 0 | 0 | N/A | 0 |
| | **Subtotal** | **200** | **200** | **127** | **129** | **63.50%** | **73** |
| **COMBINED** | **ALL** | **600** | **600** | **490** | **492** | **81.67%** | **110** |

---

## 4. Integrity, Pairing & Annotation Coordinate Audit

### 4.1 Image-Label Pairing Verification
Every image in both archives was checked against its corresponding `.txt` label:
- **Missing Label Files (Orphan Images):** **0**
- **Missing Image Files (Orphan Labels):** **0**
- **Pairing Rate:** **100.0%**

### 4.2 YOLO Normalized Coordinate Bounds
Every bounding box coordinate $[x_c, y_c, w, h]$ was tested against strict mathematical boundaries:
- $0.0 \le x_c \le 1.0$: **100% compliant** (0 violations)
- $0.0 \le y_c \le 1.0$: **100% compliant** (0 violations)
- $0.0 < w \le 1.0$: **100% compliant** (0 violations)
- $0.0 < h \le 1.0$: **100% compliant** (0 violations)
- No inverted bounding boxes ($w \le 0$ or $h \le 0$).

### 4.3 Corrupted & Empty Annotation Files
- **Empty Label Files:** **0** (All images have at least one annotation; negative samples have no zero-byte `.txt` files).
- **Corrupted Label Files:** **1 defect identified**:
  - File: `train/labels/istockphoto-1169897349-1024x1024_jpg.rf.0a3a3ff93316acb8b0c63664769318da.txt` (Source 1)
  - Line 0: `3 0.4895 0.3837 0.7297 0.6829` (Valid Class 3 Main Engine bbox)
  - Line 1: `0 0.7454 0.6735 0.9603 0.5930 0.9570 0.8053 0.7520 1 0.6283 1` (**11 tokens** — malformed polygon segmentation coordinates mixed into a detection annotation file).

---

## 5. Duplicate & Data Leakage Analysis

### 5.1 Exact Cryptographic Duplicates (SHA-256)
- **Source 1 (Engine Parts Detector v1):** 2 duplicate groups:
  1. `train` internal duplicate: `68ac5874d3800199...` (two augmented variants with identical bitstreams).
  2. **Cross-Split Leakage:** Exact SHA-256 match between `train` and `valid`:
     - `train/images/istockphoto-953763812-612x612_jpg.rf.bae7e5a1df9b7ab3cc77cf00d50f4886.jpg`
     - `valid/images/istockphoto-953763812-612x612_jpg.rf.1e1cc7a3e90e88bd579d1185e76fe7a1.jpg`
     *(Matches the exact duplicate identified and resolved in Stage 2.1).*
- **Source 2 (engine-bay-parts v1):** **0 exact duplicate groups**.
- **Between Source 1 and Source 2:** **0 exact duplicate groups** (completely disjoint image pools).

### 5.2 Perceptual Near-Duplicates (64-bit dHash, $D_H \le 4$)
- **Source 1:** 32 near-duplicate pairs, including **19 cross-split leakage pairs** between `train`, `valid`, and `test`. These represent augmented or burst-camera versions of the same vehicle scene.
- **Source 2:** 2 near-duplicate pairs within `train`.
- **Between Source 1 and Source 2:** **0 near-duplicate pairs** ($D_H > 4$ across all cross-source comparisons).

---

## 6. Visual Annotation Inspection (20 Battery Samples)

Twenty representative samples were extracted and rendered with bounding boxes into [`docs/dataset_review/stage2_5_battery/`](file:///d:/AI-Vehicle-Diagnosis-System/docs/dataset_review/stage2_5_battery/):

### Rendered Inspection Roster:
1. `zip1_epd_sample_01_800px-Honda_K24A_Engine_01...jpg` (800x536, 1 box) — Clean transverse I4 engine bay, tight battery box.
2. `zip1_epd_sample_02_preview-928x522...jpg` (928x619, 1 box) — Clear passenger-side battery with clamp terminals.
3. `zip1_epd_sample_03_civic_05_engine_large...jpg` (700x525, 1 box) — Honda Civic under-hood, tight battery perimeter.
4. `zip1_epd_sample_04_35287_29...jpg` (1280x853, 1 box) — High-res DSLR engine bay shot.
5. `zip1_epd_sample_05_1a92c257942f0d0cdf22c114b0d781f3e8089bbd...jpg` (1280x794, 1 box) — Clean battery casing with visible red positive terminal cover.
6. `zip1_epd_sample_06_CR-V-MAIN11...jpg` (1000x600, 1 box) — SUV engine bay, accurate battery localization.
7. `zip1_epd_sample_07_2007-honda-civic-si-in-gray-engine...jpg` (1300x956, 1 box) — High-detail view.
8. `zip1_epd_sample_08_image16_jpeg...jpg` (335x151, 1 box) — **Low-resolution thumbnail (undersized)**, tight box.
9. `zip1_epd_sample_09_2018_Honda_CR_V_110...jpg` (1200x600, 1 box) — Modern vehicle engine bay.
10. `zip1_epd_sample_10_gallery_used-car-carlist-honda-accord...jpg` (860x645, 1 box) — Standard dealership inspection photo.
11. `zip2_ebp_sample_01_469...jpg` (1080x810, 1 box) — Source 2 sample, well-centered battery casing.
12. `zip2_ebp_sample_02_259...jpg` (1280x720, 1 box) — Clear battery with hold-down bracket.
13. `zip2_ebp_sample_03_955...jpg` (2048x2048, 1 box) — 2K square resolution, excellent detail.
14. `zip2_ebp_sample_04_1493...jpg` (686x386, 1 box) — Slightly undersized height, valid battery box.
15. `zip2_ebp_sample_05_232...jpg` (1024x642, 1 box) — Authentic engine bay battery.
16. `zip2_ebp_sample_06_1194...jpg` (1536x1024, 1 box) — European hatchback layout.
17. `zip2_ebp_sample_07_638...jpg` (686x386, 1 box) — Valid battery label.
18. `zip2_ebp_sample_08_770...jpg` (1024x768, 1 box) — High contrast battery casing.
19. `zip2_ebp_sample_09_769...jpg` (1600x1157, 1 box) — Sharp macro view.
20. `zip2_ebp_sample_10_1128...jpg` (1300x956, 1 box) — Clear hold-down and cable routing.

### Qualitative Findings:
- **Semantic Authenticity:** 100% of samples show real automotive lead-acid or AGM starter batteries installed in vehicle engine bays.
- **Bounding Box Tightness:** High quality overall. Boxes enclose the rectangular body, top cover, terminal posts, and hold-down bracket without excessive background padding.
- **Occlusion Handling:** Partially occluded batteries (e.g. covered by intake duct or battery thermal insulation blanket) are appropriately labeled with boundary boxes around the visible unit.
- **Collage Detection in Source 2:** Samples `819_jpg` and `602_jpg` were flagged as stitched multi-image collages containing 2 batteries each.
