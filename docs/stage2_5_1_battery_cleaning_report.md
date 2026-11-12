# Stage 2.5.1: Battery Dataset Cleaning & Remediation Report

**Project Title:** AI-Powered Multimodal Vehicle Fault Detection and Diagnosis System  
**Stage:** Stage 2.5.1 – Battery Dataset Cleaning and Remediation  
**Component Focus:** Class 1 (`battery`)  
**Remediated Storage Location:** [`data/cleaned/stage2_5_battery/`](file:///d:/AI-Vehicle-Diagnosis-System/data/cleaned/stage2_5_battery/)  
**Duplicate Resolution Registry:** [`docs/stage2_5_1_duplicate_resolution.csv`](file:///d:/AI-Vehicle-Diagnosis-System/docs/stage2_5_1_duplicate_resolution.csv)  
**Partitioning Report:** [`docs/stage2_5_1_split_report.md`](file:///d:/AI-Vehicle-Diagnosis-System/docs/stage2_5_1_split_report.md)  
**Date:** November 11, 2026  

---

## 1. Executive Summary & Remediation Overview

Following the Stage 2.5 audit, seven critical data-cleaning and quality remediation tasks were executed across the two candidate battery sources:
1. **Source 1:** `Engine Parts Detector v1` (Roboflow project: `engine-parts-detector-wqzdk`, CC BY 4.0)
2. **Source 2:** `engine-bay-parts v1` (Roboflow project: `engine-bay-parts-agjnc`, MIT)

All operations were performed strictly out-of-place: the original ZIP archives remain unmodified, `data/prototype_5class/` remains untouched, and all cleaned data is isolated in `data/cleaned/stage2_5_battery/`.

### Key Outcomes:
- **Corrupted Annotation Line:** Thoroughly investigated. Converted mathematical bounding box was calculated, but because the file contains only `Air Coolant` and `Main Engine` (zero battery labels), it was excluded from the battery dataset.
- **Exact Cryptographic Duplicates Purged:** 2 duplicate instances removed (including the cross-split train-validation leak of `istockphoto-953763812`).
- **Cross-Split Data Leakage Purged:** All 19 cross-split near-duplicate pairs in Source 1 were systematically resolved, retaining higher-resolution primary instances and eliminating all train/val/test data leakage.
- **Low-Resolution Evaluation (57 images $<400\times400$):** 51 images retained (sharp landscape formats with clear, identifiable batteries $\ge 32\times32$ px); 6 images excluded (4 extreme thumbnails $<300$ px, 2 tiny unresolvable boxes $<32$ px).
- **Multi-Panel Collages Excluded:** 2 stitched comparison collages (`819_jpg` and `602_jpg`) were pruned to prevent spatial prior distortion.
- **Source 2 Group-Aware Splitting:** 121 cleaned images partitioned into an 80/15/5 split (96 train, 18 valid, 7 test) using perceptual cluster grouping to guarantee zero leakage.
- **Cross-Source Independence Confirmed:** Zero exact duplicates and zero near-duplicates between Source 1 and Source 2.
- **Final Cleaned Pool:** **465 high-quality, verified, non-leaking battery images** ready for integration.

---

## 2. Task-by-Task Remediation Results

### Task 1: Investigation of Corrupted Polygon Annotation
- **Affected File:** `train/labels/istockphoto-1169897349-1024x1024_jpg.rf.0a3a3ff93316acb8b0c63664769318da.txt`
- **Root Cause:** Line 1 contains an 11-token string representing a 5-point segmentation polygon (`0 x1 y1 x2 y2 x3 y3 x4 y4 x5 y5`), inadvertently exported into a bounding-box detection split.
- **Mathematical Bounding Box Conversion:**
  - Original vertices: $(0.7454, 0.6735)$, $(0.9603, 0.5930)$, $(0.9570, 0.8053)$, $(0.7520, 1.0)$, $(0.6283, 1.0)$.
  - Extrema: $x_{min}=0.628255, x_{max}=0.960286, y_{min}=0.592972, y_{max}=1.0$.
  - Converted YOLO Bounding Box: `0 0.794271 0.796486 0.332031 0.407028`.
- **Battery Impact:** The image contains only Class 3 (`Main Engine`) and Class 0 (`Air Coolant`). It contains **zero battery labels**. Therefore, the file was naturally excluded from the battery dataset, ensuring 100% clean detection annotations.

### Task 2: Exact Duplicates Resolution
All exact SHA-256 collisions were cataloged in [`docs/stage2_5_1_duplicate_resolution.csv`](file:///d:/AI-Vehicle-Diagnosis-System/docs/stage2_5_1_duplicate_resolution.csv):
1. **Cross-Split Leak (`DUP-EXACT-01`):** `istockphoto-953763812-612x612_jpg` existed in both `train` and `valid`.
   - *Resolution:* Preserved the training copy (`train/...bae7e5a1df9b7ab3cc77cf00d50f4886.jpg`); permanently purged the validation copy (`valid/...1e1cc7a3e90e88bd579d1185e76fe7a1.jpg`).
2. **Internal Train Duplicate (`DUP-EXACT-02`):** `68ac5874d38001994836704cb5356fb6ec1cfe94_jpg` had two identical bitstream copies in `train`.
   - *Resolution:* Retained primary copy (`...8c873b85e14482411754af1c0f361b03.jpg`); purged redundant copy (`...8efcb56c0a378b42f07094a56a465848.jpg`).

### Task 3: Near-Duplicates & Cross-Split Leakage Purging
Cross-split near duplicates ($D_H \le 4$) were evaluated pair-by-pair:
- **19 Cross-Split Pairs in Source 1:** Caused by the same physical vehicle scenes photographed at slightly different angles or resized (e.g. Wikipedia 800px vs 1280px Honda K24A engine).
- **Resolution Strategy:** Retained the higher-resolution primary instance; purged the secondary leaking instance. For pairs of equal resolution, the training copy was prioritized.
- **Source 2 Internal Near-Duplicate:** `886_jpg` ($400 \times 267$) was an exact downsampled duplicate of `838_jpg` ($995 \times 663$). The lower-resolution copy was purged.
- **Verification:** An exhaustive post-cleaning scan verified **0 remaining cross-split near-duplicate pairs** in both cleaned sources.

### Task 4: Low-Resolution Images Review (57 Images $< 400 \times 400$)
Rather than blindly discarding all 57 sub-$400\text{px}$ images, each was inspected for optical clarity, blur variance, and target bounding box pixel dimensions:
- **Retained (51 images):** High-aspect-ratio landscape photographs (e.g. $686 \times 386$, $573 \times 321$, $500 \times 375$, $480 \times 360$) with exceptional sharpness ($\sigma^2 > 1000$) where the battery object spans substantial pixels ($> 50 \times 40$ px, up to $403 \times 91$ px).
- **Excluded (6 images):**
  1. `test/...image25_jpeg` ($250 \times 201$ px) — Extreme low-res thumbnail.
  2. `valid/...images74_jpg` ($259 \times 194$ px) — Extreme low-res thumbnail.
  3. `valid/...images229_jpg` ($260 \times 194$ px) — Extreme low-res thumbnail.
  4. `train/...images203_jpg` ($275 \times 183$ px) — Extreme low-res thumbnail.
  5. `train/...843_jpg` ($250 \times 186$ px, battery box $25 \times 25$ px) — Tiny unresolvable object.
  6. `train/...1088_jpg` ($400 \times 300$ px, battery box width 29 px) — Narrow sliver box below 32 px threshold.

### Task 5: Multi-Panel Collage Inspection
- **Investigated Files:** `819_jpg.rf.ad642749...` and `602_jpg.rf.8362a893...` in Source 2.
- **Finding:** `819_jpg` contains two horizontal side-by-side engine bays; `602_jpg` contains two vertically stacked engine bays. Each image carries two distinct battery labels.
- **Diagnostic Decision:** Both collages were **permanently excluded**. Stitched comparison graphics violate real-world single-compartment spatial distributions and would teach the detector to expect multiple batteries in one frame.

### Task 6: Group-Aware Partitioning for Source 2
- Source 2's clean pool of 121 images was clustered into 120 perceptual scene groups.
- Partitioned via group-aware deterministic allocation (`seed=42`) into:
  - **Train:** 96 images (79.3%)
  - **Validation:** 18 images (14.9%)
  - **Test:** 7 images (5.8%)
- See [`docs/stage2_5_1_split_report.md`](file:///d:/AI-Vehicle-Diagnosis-System/docs/stage2_5_1_split_report.md) for full partition documentation.

### Task 7: Cross-Source Duplicate Audit
Exhaustive pairwise cryptographic and perceptual comparison between Source 1 and Source 2:
- **Exact SHA-256 Collisions:** **0**
- **Perceptual Near-Duplicates ($D_H \le 4$):** **0**
- **Conclusion:** Source 1 and Source 2 represent completely independent photographic datasets.

---

## 3. Final Cleaned Battery Dataset Inventory

All cleaned images and labels are stored under [`data/cleaned/stage2_5_battery/`](file:///d:/AI-Vehicle-Diagnosis-System/data/cleaned/stage2_5_battery/) with normalized YOLO coordinates and unified Class ID 1 (`battery`):

| Source Dataset | Cleaned Directory | Train | Valid | Test | Total Images | Battery BBoxes | License |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Source 1 (EPD Cleaned)** | `data/cleaned/stage2_5_battery/source1_epd/` | 228 | 78 | 38 | **344** | 344 | CC BY 4.0 |
| **Source 2 (EBP Cleaned)** | `data/cleaned/stage2_5_battery/source2_ebp/` | 96 | 18 | 7 | **121** | 121 | MIT |
| **CONSOLIDATED TOTAL** | `data/cleaned/stage2_5_battery/` | **324** (69.7%) | **96** (20.6%) | **45** (9.7%) | **465** | **465** | Open / Permissive |

---

## 4. Quality Control & Audit Checklist

- [x] Malformed polygon line investigated and excluded.
- [x] All exact duplicate pairs purged (0 duplicates remaining).
- [x] All 19 cross-split near-duplicate leakage pairs resolved (0 leakage remaining).
- [x] All 57 sub-$400\text{px}$ images audited; 51 high-clarity retained, 6 invalid excluded.
- [x] Artificial multi-panel collages pruned.
- [x] Source 2 successfully split into leak-free 80/15/5 partitions.
- [x] Zero overlap confirmed between Source 1 and Source 2.
- [x] 100% of annotations unified to Prototype Class ID 1 (`battery`).
- [x] `data/prototype_5class/` untouched and preserved.
- [x] Complete automated test suite passing.
