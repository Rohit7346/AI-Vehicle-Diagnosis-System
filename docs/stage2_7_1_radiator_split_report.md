# Stage 2.7.1: Dataset Partitioning & Leakage Prevention Report

**Project Title:** AI-Powered Multimodal Vehicle Fault Detection and Diagnosis System  
**Stage:** Stage 2.7.1 – Radiator Dataset Cleaning and Semantic Remediation  
**Component Focus:** Class 3 (`radiator`)  
**Subject:** Train / Validation / Test Group-Aware Partitioning & Data Leakage Elimination  
**Cleaned Dataset Root:** [`data/cleaned/stage2_7_radiator/`](file:///D:/AI-Vehicle-Diagnosis-System/data/cleaned/stage2_7_radiator/)  
**Date:** November 11, 2026  

---

## 1. Partitioning Strategy & Architectural Objectives

In production computer vision pipelines, splitting images randomly without clustering allows photos from the same photographic burst or vehicle photoshoot to leak across training, validation, and test sets. This inflates evaluation metrics and masks real-world generalization errors.

Prior to Stage 2.7.1, the candidate radiator datasets exhibited two major architectural partitioning defects:
1. **Source 1 (`EPD v1`):** Contained **13 cross-split perceptual near-duplicate leakage pairs** ($D_H \le 4$) bridging `train`, `valid`, and `test`. (These were resolved by the complete exclusion of Source 1's cap-only annotations).
2. **Source 2 (`EBP v1`):** Had **100% of its data in `train/`** (200 total images, 118 candidate radiator images), with completely empty validation and test partitions ($0$ images).

Stage 2.7.1 established a rigorous, reproducible group-aware 80/15/5 partitioning pipeline for Source 2.

---

## 2. Source 2 Group-Aware Partitioning Methodology

### 2.1 Cluster Formation
- **Initial Candidate Pool:** 118 radiator images in Source 2.
- **Pre-Split Exclusions ($N = 3$):**
  - `819_jpg.rf.ad6427498d2d7e6daaac820841956f91.jpg` (Horizontal stitched collage)
  - `602_jpg.rf.8362a8937d799db7104e324ef055ef5d.jpg` (Vertical stitched collage)
  - `886_jpg.rf.5de572abb2d39ec63a1571962ab1eaf1.jpg` (Low-resolution duplicate of `838_jpg`)
- **Cleaned Population:** **115 unique, verified radiator images**.
- **Perceptual Clustering ($D_H \le 4$):** Pairwise Hamming distances between all 115 difference hashes (dHash) were computed.
  - **111 total clusters** were formed.
  - **4 multi-image clusters (8 images):**
    1. Cluster 1: `248_jpg` ($2048\times2048$) $\longleftrightarrow$ `266_jpg` ($600\times600$), $D_H = 0$
    2. Cluster 2: `504_jpg` ($1080\times1920$) $\longleftrightarrow$ `536_jpg` ($640\times1137$), $D_H = 0$
    3. Cluster 3: `710_jpg` ($3662\times2122$) $\longleftrightarrow$ `747_jpg` ($1080\times625$), $D_H = 0$
    4. Cluster 4: `836_jpg` ($1008\times671$) $\longleftrightarrow$ `862_jpg` ($1008\times671$), $D_H = 2$
  - **107 single-image clusters (107 images).**

---

### 2.2 Split Assignment Algorithm (Seed 42)
To guarantee zero leakage:
1. **Multi-image clusters (8 images) are locked into the `train` partition**, ensuring related images never cross into validation or test.
2. The remaining 107 single-image clusters are shuffled deterministically with `random.seed(42)` and assigned to meet the target ratios:
   - Target Train: $115 \times 0.80 = 92$ images
   - Target Valid: $115 \times 0.15 = 17$ images
   - Target Test:  $115 \times 0.05 = 6$ images

---

## 3. Final Split Accounting & Verification

### 3.1 Partition Summary:

| Partition | Images | Percentage | Radiator BBoxes | Mean BBox Area | Mean Aspect Ratio ($w/h$) | Resolution Min/Max |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **`train`** | **92** | **80.0%** | **92** | 0.0451 | 8.28 | $400\times267$ to $3662\times2122$ |
| **`valid`** | **17** | **14.8%** | **17** | 0.0428 | 8.45 | $400\times267$ to $2048\times1536$ |
| **`test`** | **6** | **5.2%** | **6** | 0.0435 | 8.31 | $600\times450$ to $1920\times1080$ |
| **TOTAL** | **115** | **100.0%** | **115** | **0.0446** | **8.34** | — |

---

### 3.2 Cross-Split Leakage Audit

A comprehensive post-split perceptual hash comparison was executed across all cross-split pairs:
- **`train` vs `valid` pairs evaluated:** $92 \times 17 = 1,564$ pairs
  - Minimum observed Hamming distance: **$D_H = 12$** (Threshold is $\le 4$)
  - **Cross-split leakage: 0 pairs (PASS)**
- **`train` vs `test` pairs evaluated:** $92 \times 6 = 552$ pairs
  - Minimum observed Hamming distance: **$D_H = 14$**
  - **Cross-split leakage: 0 pairs (PASS)**
- **`valid` vs `test` pairs evaluated:** $17 \times 6 = 102$ pairs
  - Minimum observed Hamming distance: **$D_H = 16$**
  - **Cross-split leakage: 0 pairs (PASS)**

---

## 4. Source 1 Accounting Summary

- **Raw Candidate Images:** 260 images (177 train, 54 valid, 29 test)
- **Candidate Radiator Bounding Boxes:** 269 boxes
- **Retained Images:** **0 images**
- **Retained Bounding Boxes:** **0 boxes**
- **Reason:** 100% of candidate boxes represent isolated pressure caps, filler necks, coolant overflow bottles, or gross engine bay errors. Zero boxes annotate the complete radiator assembly. All 269 boxes are documented in [`data/cleaned/stage2_7_radiator/audit/source1_exclusions.csv`](file:///D:/AI-Vehicle-Diagnosis-System/data/cleaned/stage2_7_radiator/audit/source1_exclusions.csv).

---

## 5. Cleaned Dataset Inventory

All cleaned files reside in:
- Images: `data/cleaned/stage2_7_radiator/source2_ebp/images/{train,valid,test}/`
- Labels: `data/cleaned/stage2_7_radiator/source2_ebp/labels/{train,valid,test}/`
- Every label file contains exactly one annotation formatted as:
  ```
  3 <x_center> <y_center> <width> <height>
  ```
  where class ID `3` corresponds to `radiator` in the prototype 5-class ontology.
