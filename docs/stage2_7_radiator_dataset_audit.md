# Stage 2.7: Radiator Dataset Audit & Multi-Source Quality Analysis

**Project Title:** AI-Powered Multimodal Vehicle Fault Detection and Diagnosis System  
**Stage:** Stage 2.7 – Radiator Dataset Audit  
**Component Focus:** Class 3 (`radiator`)  
**Auditor Engine:** `tests/test_radiator_audit.py` & `scratch/deep_audit_stage2_7.py`  
**Distribution CSV:** [`docs/stage2_7_radiator_class_distribution.csv`](file:///D:/AI-Vehicle-Diagnosis-System/docs/stage2_7_radiator_class_distribution.csv)  
**Quality Engineering Report:** [`docs/stage2_7_radiator_quality_report.md`](file:///D:/AI-Vehicle-Diagnosis-System/docs/stage2_7_radiator_quality_report.md)  
**Visual Review Directory:** [`docs/stage2_7_radiator_visual_review/`](file:///D:/AI-Vehicle-Diagnosis-System/docs/stage2_7_radiator_visual_review/)  
**Date:** November 11, 2026  

---

## 1. Executive Summary

This audit performs an exhaustive, rigorous data-quality and provenance evaluation of radiator annotations across two candidate public YOLOv8 dataset archives downloaded from Roboflow and staged in `data/raw/stage2_7_radiator/`:
1. **`Engine Parts Detector.v1i.yolov8.zip`** (Source 1, Roboflow project: `engine-parts-detector-wqzdk`, v1)
2. **`engine-bay-parts.v1i.yolov8.zip`** (Source 2, Roboflow project: `engine-bay-parts-agjnc`, v1)

Per strict safety constraints:
- The candidate raw ZIP archives were audited in place without alteration.
- The 5-class prototype dataset (`data/prototype_5class/`) and the cleaned battery dataset (`data/cleaned/stage2_5_battery/`) were preserved untouched.
- No dataset merging or model training was performed.

### Key Audit Metrics at a Glance:

| Metric | Source 1 (`Engine Parts Detector v1`) | Source 2 (`engine-bay-parts v1`) | Combined Total |
|---|:---:|:---:|:---:|
| **Source Radiator Class ID** | **`4`** | **`9`** | Maps to Prototype Class **`3`** |
| **Total Images in Archive** | 400 | 200 | 600 |
| **Total Label Files** | 400 | 200 | 600 |
| **Image-to-Label Pairing** | 100% (400/400) | 100% (200/200) | 100% (600/600) |
| **Radiator Images** | **260** (65.0%) | **118** (59.0%) | **378** (63.0%) |
| **Radiator Bounding Boxes** | **269** | **119** | **388** |
| **Multi-Box Radiator Images** | 9 images (18 boxes) | 1 image (2 boxes) | 10 images (20 boxes) |
| **Mean Bounding Box Area ($w \times h$)** | **0.00715** (0.72% of image) | **0.04459** (4.46% of image) | **0.01863** |
| **Mean Aspect Ratio ($w / h$)** | **0.98** (near-square 1:1) | **8.34** (wide horizontal span) | **3.24** |
| **Tiny Boxes (Area $< 0.005$)** | **139** (51.7% of boxes) | **1** (0.8% of boxes) | **140** (36.1%) |
| **Exact SHA-256 Duplicates** | 1 group (2 images in train) | 0 | 1 group |
| **Perceptual Near-Duplicates ($D_H \le 4$)** | 21 pairs | 5 pairs | 26 pairs |
| **Cross-Split Leakage Pairs** | **13 pairs** (Train $\leftrightarrow$ Valid $\leftrightarrow$ Test) | 0 (Train only) | 13 pairs |
| **Undersized Images ($< 400 \times 400$)** | 19 images | 10 images | 29 images |
| **Dataset License** | CC BY 4.0 | MIT | Open Open-Source Compatible |

---

## 2. Class Ontology & Target Mapping

### 2.1 Source 1: `Engine Parts Detector v1`
From `data.yaml`:
```yaml
nc: 6
names: ['Air Coolant', 'Battery', 'Fuse-Box', 'Main Engine', 'Radiator', 'Windscreen Washer']
```
- **Radiator Class ID:** **`4`**
- **Prototype Mapping:** Maps directly to Prototype Class ID **`3`** (`radiator`).
- **Semantic Finding:** Source 1 lacks dedicated classes for `radiator_cap` or `coolant_reservoir`. Consequently, human annotators applied Class 4 to pressure relief caps, coolant filler necks, and small upper tanks rather than the full radiator core.

### 2.2 Source 2: `engine-bay-parts v1`
From `data.yaml`:
```yaml
nc: 12
names: [
  'Air Filter Cover', 'Alternator', 'Battery', 'Engine Coolant Reservoir',
  'Engine Cover', 'Engine Oil Dip Stick', 'Engine Oil Fill Cap', 'Fuse Box',
  'Oil Filter Housing', 'Radiator', 'Radiator Cap', 'Windshield Wiper Fluid'
]
```
- **Radiator Class ID:** **`9`**
- **Prototype Mapping:** Maps directly to Prototype Class ID **`3`** (`radiator`).
- **Semantic Finding:** Source 2 annotators explicitly isolated **`Radiator Cap` (Class 10)** and **`Engine Coolant Reservoir` (Class 3)**. Class 9 specifically and exclusively encompasses the wide horizontal radiator core, top tank, and front header assembly.

---

## 3. Split Distribution & Image-Label Pairing

### 3.1 Split Breakdown
- **Source 1 (`Engine Parts Detector v1`):**
  - `train`: 265 total images, 177 radiator images, 184 radiator bounding boxes (7 multi-box images).
  - `valid`: 86 total images, 54 radiator images, 56 radiator bounding boxes (2 multi-box images).
  - `test`: 49 total images, 29 radiator images, 29 radiator bounding boxes (0 multi-box images).
- **Source 2 (`engine-bay-parts v1`):**
  - `train`: 200 total images, 118 radiator images, 119 radiator bounding boxes (1 multi-box image: `819_jpg` collage).
  - `valid`: 0 images (Empty split).
  - `test`: 0 images (Empty split).

### 3.2 Label Pairing & Coordinate Integrity
- **Pairing Rate:** Across all 600 images in both archives, exactly 600 corresponding `.txt` label files exist (100% 1-to-1 match). Zero orphan images and zero orphan label files were detected.
- **Coordinate Boundaries:** All 388 radiator bounding boxes strictly adhere to normalized YOLO format ($0.0 \le x_c, y_c \le 1.0$ and $0.0 < w, h \le 1.0$). Zero negative coordinates or $>1.0$ coordinate overflow conditions were detected.
- **Corrupted Label Alert:** In Source 1, `train/labels/0e7643ff-FD2-Civic-Type-R-sedan49_jpg.rf.7b017b3ca78ea3be73119133f5f61e60.txt` contains an 11-token segmentation polygon line. (This polygon belongs to other components; the radiator bounding box in this file remains extractable).

---

## 4. Radiator Geometry & Semantic Annotation Discrepancy (Task 12)

A primary objective of Stage 2.7 is to determine whether candidate bounding boxes delineate the **complete radiator assembly** or merely subcomponents (such as the radiator cap, upper neck, or overflow tank).

### 4.1 Quantitative Geometric Analysis

| Parameter | Source 1 (`Engine Parts Detector v1`) | Source 2 (`engine-bay-parts v1`) |
|---|:---:|:---:|
| **Sample Size ($N$)** | 269 boxes | 119 boxes |
| **Minimum Box Area** | $0.00055$ ($0.05\%$ of frame) | $0.00470$ ($0.47\%$ of frame) |
| **Maximum Box Area** | $0.32241$ ($32.24\%$ of frame) | $0.24451$ ($24.45\%$ of frame) |
| **Mean Box Area** | **$0.00715$ ($0.72\%$ of frame)** | **$0.04459$ ($4.46\%$ of frame)** |
| **Boxes with Area $< 0.005$** | **139 boxes (51.67%)** | **1 box (0.84%)** |
| **Boxes with Area $0.005 - 0.05$** | 129 boxes (47.96%) | 83 boxes (69.75%) |
| **Boxes with Area $> 0.05$** | 1 box (0.37%) | 35 boxes (29.41%) |
| **Mean Aspect Ratio ($w/h$)** | **0.98** (Square, Cap-like) | **8.34** (Elongated, Header-like) |

### 4.2 Semantic Breakdown
1. **Source 1 Labels the Radiator Cap:** In over half of Source 1 instances, the bounding box measures less than 0.5% of the total image area, with a width-to-height aspect ratio of ~1.0. Visual review confirms that these boxes tightly enclose only the metal/plastic pressure relief cap on the radiator neck.
2. **Source 2 Labels the Complete Core Assembly:** In Source 2, the mean aspect ratio is 8.34:1, representing wide horizontal boxes stretching across the front radiator support / slam panel of the engine bay. These boxes properly identify the upper radiator tank, cooling fins, and mounting structure.
3. **Multi-Box Anomalies:** 
   - Source 1 contains 9 images with 2 radiator boxes. In several cases (e.g. `train/images/2022-Honda-Civic-eHEV-RS-Hybrid...`), annotators boxed both the main radiator cap and the auxiliary inverter cooling cap as two separate "radiators".
   - Source 2 contains exactly 1 multi-box image (`train/images/819_jpg...`), which is a two-panel stitched comparison photo showing two independent engine bays.

---

## 5. Duplicate Detection & Cross-Split Leakage

### 5.1 Exact SHA-256 Duplicates
- **Source 1:** Contains 1 exact duplicate group in `train`:
  - `train/images/68ac5874d38001994836704cb5356fb6ec1cfe94_jpg.rf.8c873b85e14482411754af1c0f361b03.jpg`
  - `train/images/68ac5874d38001994836704cb5356fb6ec1cfe94_jpg.rf.8efcb56c0a378b42f07094a56a465848.jpg`
  (Hash: `7e2af39cbae4813038a8e3fca4013401fa9be05bb3db44a37703efb7d2fbf04d`)
- **Source 2:** Zero exact duplicates detected (0 SHA-256 collisions).
- **Cross-Source:** Zero exact duplicates between Source 1 and Source 2.

### 5.2 Perceptual Near-Duplicates & Data Leakage ($D_H \le 4$)
- **Source 1 Data Leakage:** Source 1 contains **21 near-duplicate pairs**, of which **13 pairs cross dataset split boundaries**:
  - `test/images/800px-Honda_K24A_Engine_01_jpg...` ($D_H=0$) $\longleftrightarrow$ `train/images/1280px-Honda_K24A_Engine_01_jpeg...`
  - `test/images/1a92c257942f0d0cdf22c114b0d781f3e8089bbd_jpg...` ($D_H=0$) $\longleftrightarrow$ `train/images/Honda_City_V_Ext-26_jpg...`
  - `test/images/2020_honda_civic_sedan_46_jpg...` ($D_H=4$) $\longleftrightarrow$ `train/images/HD-wallpaper-2020-honda-civic-sedan-touring-engine-car_jpg...`
  - `train/images/best-honda-engines-07_jpg...` ($D_H=0$) $\longleftrightarrow$ `valid/images/1c01390de2ba4b1b84c35aa450356635_jpeg...`
  - Additional 9 pairs linking `train` and `valid`.
- **Source 2 Near-Duplicates:** Contains 5 near-duplicate pairs, all residing within `train` (zero cross-split leakage because validation and test are empty).
- **Cross-Source Near-Duplicates:** Zero near-duplicate pairs detected between Source 1 and Source 2.

---

## 6. Image Resolution & Composition Quality

### 6.1 Undersized Images ($< 400 \times 400$)
- **Source 1:** 19 images below $400 \times 400$ px (lowest: $300 \times 225$ px).
- **Source 2:** 10 images below $400 \times 400$ px (lowest: $300 \times 225$ px).
- **Usability Verdict:** The radiator core or cap remains identifiable in all 29 undersized images when upscaled with bilinear interpolation during YOLO pre-processing, but noise and compression artifacts require visual filtering in Stage 2.7.1.

### 6.2 Multi-Panel Collages
- In Source 2, `train/images/819_jpg.rf.ad6427498d2d7e6daaac820841956f91.jpg` is a horizontal two-panel collage containing two separate engine compartments and two radiator annotations. This image must be quarantined or split in the cleaning stage.

---

## 7. Visual Inspection Deliverables

As required by Task 11, **30 rendered JPEG visual review samples** have been generated and archived in:
[`docs/stage2_7_radiator_visual_review/`](file:///D:/AI-Vehicle-Diagnosis-System/docs/stage2_7_radiator_visual_review/)

- **Source 1 Samples (15 images):** `src1_epd_sample_01` through `src1_epd_sample_15`.
- **Source 2 Samples (15 images):** `src2_ebp_sample_01` through `src2_ebp_sample_15`.
- Each review image renders:
  - Sky Blue 3px bounding box for `radiator`.
  - Informative bounding box badge indicating pixel dimensions ($W \times H$), aspect ratio ($AR = w/h$), and relative area ($w \times h$).
  - Light gray 1px wireframes for non-radiator auxiliary annotations (batteries, engines, fuse boxes).

---

## 8. Provenance & Dataset Licensing

1. **Source 1: `Engine Parts Detector v1`**
   - Roboflow Universe Project: `engine-parts-detector-wqzdk/1`
   - License: **Creative Commons Attribution 4.0 International (CC BY 4.0)**
   - Commercial / Academic Use: Permitted with attribution.
2. **Source 2: `engine-bay-parts v1`**
   - Roboflow Universe Project: `engine-bay-parts-agjnc/1`
   - License: **MIT License**
   - Commercial / Academic Use: Permitted with attribution.
