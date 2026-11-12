# Stage 2.5: Battery Dataset Quality Engineering & Risk Evaluation Report

**Project Title:** AI-Powered Multimodal Vehicle Fault Detection and Diagnosis System  
**Stage:** Stage 2.5 – Battery Dataset Audit  
**Component Focus:** Class 1 (`battery`)  
**Audited Sources:** `Engine Parts Detector.v1i.yolov8.zip` & `engine-bay-parts.v1i.yolov8.zip`  
**Distribution CSV:** [`docs/stage2_5_battery_class_distribution.csv`](file:///d:/AI-Vehicle-Diagnosis-System/docs/stage2_5_battery_class_distribution.csv)  
**Detailed Audit Log:** [`docs/stage2_5_battery_dataset_audit.md`](file:///d:/AI-Vehicle-Diagnosis-System/docs/stage2_5_battery_dataset_audit.md)  
**Date:** November 11, 2026  

---

## 1. Quality Engineering Scorecards

Both dataset archives were evaluated across seven core computer vision quality dimensions:

| Quality Dimension | Evaluation Standard | Source 1: Engine Parts Detector v1 | Source 2: engine-bay-parts v1 |
|---|---|:---:|:---:|
| **1. Domain Authenticity** | Real automotive under-hood 12V batteries | **100% PASS** (No consumer cells) | **100% PASS** (No consumer cells) |
| **2. Image-Label Pairing** | 1:1 image-to-label file match | **100% PASS** (400/400 paired) | **100% PASS** (200/200 paired) |
| **3. Coordinate Validity** | Normalized YOLO bbox format $[0, 1]$ | **99.75% PASS** (1 polygon error) | **100% PASS** (0 errors) |
| **4. Resolution Quality** | Minimum $400 \times 400$ px resolution | **90.0% PASS** (40 undersized) | **91.5% PASS** (17 undersized) |
| **5. Split Integrity** | Clean train/val/test partitions | **78.0% WARNING** (19 leaking pairs) | **50.0% DEFECT** (100% train only) |
| **6. Duplicate Discipline** | Zero identical or augmented leakage | **92.0% WARNING** (2 SHA-256 groups) | **99.0% PASS** (0 SHA-256 dups) |
| **7. Composition Consistency** | Single vehicle scene per photograph | **100% PASS** (Single scenes) | **98.0% WARNING** (2 stitched collages) |
| **OVERALL QUALITY SCORE** | Weighted composite score (0–100) | **88 / 100 (GOOD)** | **82 / 100 (ACCEPTABLE)** |

---

## 2. Risk Evaluation & Critical Anomalies

### 2.1 Anomaly 1: Cross-Split Duplicate Leakage (Source 1)
- **Defect Description:** An exact SHA-256 duplicate image exists simultaneously in `train` and `valid`:
  - `train/images/istockphoto-953763812-612x612_jpg.rf.bae7e5a1df9b7ab3cc77cf00d50f4886.jpg`
  - `valid/images/istockphoto-953763812-612x612_jpg.rf.1e1cc7a3e90e88bd579d1185e76fe7a1.jpg`
- **Near-Duplicate Leakage:** 19 additional pairs across `train`, `valid`, and `test` have dHash Hamming distance $\le 4$, representing augmented or burst-camera versions of identical vehicle bays.
- **Risk to ML System:** Evaluating a trained model on validation images that share identical visual scenes with training images produces artificially inflated mAP scores ($>0.95$) that collapse during real-world workshop deployment.

### 2.2 Anomaly 2: Complete Split Absence (Source 2)
- **Defect Description:** Source 2 contains 200 images, but **100% are placed in the `train/` directory**. There are 0 images in `valid/` and 0 images in `test/`.
- **Risk to ML System:** Naively passing Source 2 directly to an Ultralytics YOLO training run will trigger training validation errors due to an empty validation set, or require automatic re-partitioning.

### 2.3 Anomaly 3: Corrupted Annotation Line (Source 1)
- **Defect Description:** In `train/labels/istockphoto-1169897349-1024x1024_jpg.rf.0a3a3ff93316acb8b0c63664769318da.txt`:
  - Line 1 contains: `0 0.7454 0.6735 0.9603 0.5930 0.9570 0.8053 0.7520 1 0.6283 1` (11 space-separated numbers).
- **Root Cause:** A polygon segmentation mask was erroneously exported into a bounding-box detection dataset.
- **Risk to ML System:** Ultralytics YOLOv8 will raise a parsing warning or error if strict 5-element format is enforced during dataset caching.

### 2.4 Anomaly 4: Stitched Multi-Panel Collages (Source 2)
- **Defect Description:** Images `819_jpg.rf.ad642749...` and `602_jpg.rf.8362a893...` contain two engine bays stitched side-by-side or vertically into a single image, resulting in two duplicate labels for every component.
- **Risk to ML System:** The model learns that an "engine bay" contains two side-by-side batteries and two radiators, distorting anchor box priors and non-maximum suppression (NMS).

---

## 3. Dimensional & Bounding Box Size Distribution

### 3.1 Undersized Resolution Analysis
Our project enforces a minimum resolution of $400 \times 400$ pixels to ensure fine edge gradients are preserved:
- **Source 1:** 40 of 363 battery images ($11.0\%$) are $< 400$ px (e.g. $573 \times 321$, $250 \times 201$, $335 \times 151$).
- **Source 2:** 17 of 127 battery images ($13.4\%$) are $< 400$ px (e.g. $480 \times 360$, $686 \times 386$, $400 \times 267$).
- **Recommendation:** Rescale or filter undersized images prior to inclusion in prototype training splits.

### 3.2 Bounding Box Relative Area Distribution ($w \times h$)
- **Normal Range ($0.02 \le \text{Area} \le 0.25$):** $88.4\%$ of all battery boxes fall into this range, representing standard in-situ battery proportions within an open engine bay.
- **Close-Up Range ($\text{Area} > 0.40$):** 33 images in Source 1 depict tight macro shots where the battery occupies $45\%$ to $76\%$ of the frame.
- **Distant / Tiny Range ($\text{Area} < 0.005$):** 1 image in Source 2 (`754_jpg`, area = $0.0022$) depicts a distant panoramic engine bay where the battery is an unresolvable sliver.

---

## 4. Academic License & Attribution Registry

Both datasets provide open, academically and commercially permissive licensing compatible with final-year research:

| Source Dataset | Upstream Workspace / Author | Upstream URL | Declared License | Attribution Compliance Requirement |
|---|---|---|:---:|---|
| **Engine Parts Detector v1** | `rohits-workspace-4tw8c` (forked from upstream community) | [Roboflow Project Link](https://app.roboflow.com/rohits-workspace-4tw8c/engine-parts-detector-wqzdk/1) | **CC BY 4.0** | Attribute original dataset creators and Roboflow in project thesis |
| **engine-bay-parts v1** | `rohits-workspace-4tw8c` (forked from upstream community) | [Roboflow Project Link](https://app.roboflow.com/rohits-workspace-4tw8c/engine-bay-parts-agjnc/1) | **MIT** | Include standard MIT license copyright notice in repository documentation |

---

## 5. Actionable Roadmap for Stage 2 Integration

Before any battery data from these archives is integrated into the prototype dataset:

1. **Polygon Sanitization:** Discard or convert the corrupted polygon line in `istockphoto-1169897349` to a standard 4-coordinate bounding box.
2. **Duplicate Purge:** Remove the duplicate `istockphoto-953763812` validation instance and resolve the 19 near-duplicate leakage pairs.
3. **Partition Reconstruction for Source 2:** Resplit Source 2's 127 battery images using a stratified 80% train / 15% val / 5% test partition.
4. **Collage Pruning:** Exclude stitched multi-image collages (`819_jpg` and `602_jpg`) from training to avoid confusing the detector.
5. **Class ID Remapping:**
   - Source 1 Class 1 (`Battery`) $\rightarrow$ Prototype Class 1 (`battery`)
   - Source 2 Class 2 (`Battery`) $\rightarrow$ Prototype Class 1 (`battery`)
6. **Strict Isolation:** Keep `data/prototype_5class/` untouched until formal integration approval is granted.
