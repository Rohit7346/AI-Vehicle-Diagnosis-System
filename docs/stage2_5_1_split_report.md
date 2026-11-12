# Stage 2.5.1: Dataset Partitioning & Leakage Prevention Report

**Project Title:** AI-Powered Multimodal Vehicle Fault Detection and Diagnosis System  
**Stage:** Stage 2.5.1 – Battery Dataset Cleaning and Remediation  
**Subject:** Train / Validation / Test Splitting & Data Leakage Prevention  
**Cleaned Dataset Path:** [`data/cleaned/stage2_5_battery/`](file:///d:/AI-Vehicle-Diagnosis-System/data/cleaned/stage2_5_battery/)  
**Date:** November 11, 2026  

---

## 1. Partitioning Strategy & Architectural Objectives

A fundamental requirement of real-world computer vision engineering is the prevention of data leakage between training, validation, and testing partitions. Prior to cleaning, the candidate archives exhibited two severe architectural defects:
1. **Source 1 (EPD v1):** Carried 19 cross-split near-duplicate pairs ($D_H \le 4$) and 1 exact SHA-256 duplicate between `train` and `valid`.
2. **Source 2 (EBP v1):** Had 100% of its data in `train/` with zero validation or test sets.

Stage 2.5.1 resolved both issues by enforcing strict group-aware partitioning discipline.

---

## 2. Source 1 (Engine Parts Detector v1) Split Remediation

### Pre-Cleaning vs Post-Cleaning Split Accounting:
| Split | Raw Battery Images | Removed (Leaking / Duplicates / Extreme Thumbnails) | Cleaned Retained Battery Images | Retained Percentage |
|---|:---:|:---:|:---:|:---:|
| **`train`** | 236 | -8 (1 internal exact, 7 leaking near-duplicates) | **228** | 66.3% |
| **`valid`** | 84 | -6 (1 exact cross-split leak, 4 leaking near-duplicates, 1 thumbnail) | **78** | 22.7% |
| **`test`** | 43 | -5 (5 leaking near-duplicates) | **38** | 11.0% |
| **TOTAL** | **363** | **-19** | **344** | **100.0%** |

### Post-Cleaning Leakage Audit:
- **Exact SHA-256 Collisions across splits:** **0**
- **Perceptual Near-Duplicates ($D_H \le 4$) across splits:** **0**
- **Image-to-Label Pairing:** **100.0%** (228 train, 78 valid, 38 test paired)

---

## 3. Source 2 (engine-bay-parts v1) Group-Aware Split Creation

### 3.1 Initial Pool & Exclusions
- Raw images in archive: 200 (all in `train/`)
- Raw battery images: 127
- **Pruned Unsuitable Images (6 files):**
  - Stitched comparison collages (`819_jpg` and `602_jpg`): 2 files.
  - Lower-resolution duplicate (`886_jpg`, $400 \times 267$ copy of `838_jpg` $995 \times 663$): 1 file.
  - Tiny-box images with box dimension $<32$ px (`843_jpg`, `1088_jpg`, `1143_jpg`): 3 files.
- **Clean Pool for Partitioning:** **121 images**

### 3.2 Group-Aware Clustering Algorithm
To prevent splitting related images of the same vehicle across subsets, images were clustered using perceptual hash similarity:
$$\text{Cluster}(I_a, I_b) \iff D_H(\text{dHash}(I_a), \text{dHash}(I_b)) \le 4$$

The 121 images formed **120 independent scene clusters** (one cluster contained two angle variations, which were locked together into the same split).

### 3.3 Deterministic Partition Allocation
Using a fixed pseudorandom seed (`seed=42`), clusters were assigned sequentially to target ratios of 80% Train, 15% Validation, and 5% Test:

| Split | Target Percentage | Target Count | Actual Assigned Images | Actual Percentage | Battery BBoxes |
|---|:---:|:---:|:---:|:---:|:---:|
| **`train`** | 80.0% | 96 | **96** | **79.3%** | 96 |
| **`valid`** | 15.0% | 18 | **18** | **14.9%** | 18 |
| **`test`** | 5.0% | 7 | **7** | **5.8%** | 7 |
| **TOTAL** | **100.0%** | **121** | **121** | **100.0%** | **121** |

### Post-Splitting Leakage Audit:
- **Exact SHA-256 Collisions across splits:** **0**
- **Perceptual Near-Duplicates ($D_H \le 4$) across splits:** **0**
- **Image-to-Label Pairing:** **100.0%** (96 train, 18 valid, 7 test paired)

---

## 4. Consolidated Cleaned Battery Dataset Overview

When evaluated as a unified cleaned battery collection across both sources in `data/cleaned/stage2_5_battery/`:

```
Consolidated Cleaned Battery Dataset:
├── Source 1 (EPD Cleaned):  344 images │ 344 bboxes │ CC BY 4.0
│   ├── train: 228 images
│   ├── valid:  78 images
│   └── test:   38 images
├── Source 2 (EBP Cleaned):  121 images │ 121 bboxes │ MIT
│   ├── train:  96 images
│   ├── valid:  18 images
│   └── test:    7 images
└── COMBINED TOTAL:          465 images │ 465 bboxes │ Zero Leakage
    ├── train: 324 images (69.7%)
    ├── valid:  96 images (20.6%)
    └── test:   45 images ( 9.7%)
```

---

## 5. Verification Checklist for Prototype Integration

Before merging this cleaned pool into `data/prototype_5class/`:
- [x] All 465 images verified readable with valid pixel dimensions.
- [x] All 465 label files carry standard 5-element YOLO coordinates for Class ID 1 (`battery`).
- [x] Zero cross-split leakage exists in either source dataset.
- [x] Zero duplicate images exist between Source 1 and Source 2.
- [x] All original ZIP archives remain untouched.
- [x] Existing prototype dataset (`data/prototype_5class/`) remains completely isolated.
