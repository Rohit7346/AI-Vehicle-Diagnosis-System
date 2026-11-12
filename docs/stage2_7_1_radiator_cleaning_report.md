# Stage 2.7.1: Radiator Dataset Cleaning & Semantic Remediation Report

**Project Title:** AI-Powered Multimodal Vehicle Fault Detection and Diagnosis System  
**Stage:** Stage 2.7.1 – Radiator Dataset Cleaning and Semantic Remediation  
**Component Focus:** Class 3 (`radiator`)  
**Cleaned Dataset Root:** [`data/cleaned/stage2_7_radiator/`](file:///D:/AI-Vehicle-Diagnosis-System/data/cleaned/stage2_7_radiator/)  
**Duplicate Resolution Registry:** [`docs/stage2_7_1_duplicate_resolution.csv`](file:///D:/AI-Vehicle-Diagnosis-System/docs/stage2_7_1_duplicate_resolution.csv)  
**Partitioning & Leakage Report:** [`docs/stage2_7_1_radiator_split_report.md`](file:///D:/AI-Vehicle-Diagnosis-System/docs/stage2_7_1_radiator_split_report.md)  
**Visual Review Samples:** [`docs/stage2_7_1_radiator_visual_review/`](file:///D:/AI-Vehicle-Diagnosis-System/docs/stage2_7_1_radiator_visual_review/)  
**Test Suite:** [`tests/test_radiator_cleaning.py`](file:///D:/AI-Vehicle-Diagnosis-System/tests/test_radiator_cleaning.py)  
**Date:** November 11, 2026  

---

## 1. Executive Summary & Remediation Outcomes

Following the Stage 2.7 audit, this remediation stage resolved critical semantic divergence, data leakage, and structural anomalies across the two candidate radiator sources:
1. **Source 1:** `Engine Parts Detector v1` (Roboflow project: `engine-parts-detector-wqzdk`, CC BY 4.0)
2. **Source 2:** `engine-bay-parts v1` (Roboflow project: `engine-bay-parts-agjnc`, MIT)

All operations were executed strictly out-of-place: the original ZIP archives remain untouched, `data/prototype_5class/` remains untouched, and the cleaned battery dataset (`data/cleaned/stage2_5_battery/`) remains preserved.

### Core Remediation Outcomes:
- **Enforcement of Project Specification:**
  > *"The radiator class must represent the actual radiator cooling core and identifiable radiator assembly, not an isolated pressure cap, coolant reservoir, filler neck or unrelated component."*
- **Source 1 Semantic Filtering (100% Excluded):** All 269 candidate radiator boxes in Source 1 were audited individually. Zero boxes captured the radiator cooling core; 263 boxes were isolated pressure caps or filler necks ($AR \approx 0.98$), 5 were coolant overflow bottles, and 1 covered the engine compartment. All 269 boxes were excluded from the training dataset to prevent model corruption, with full per-box audit records saved to `data/cleaned/stage2_7_radiator/audit/source1_exclusions.csv`.
- **Source 2 Quality Validation (115 Images Retained):** All 119 boxes in Source 2 were audited. 115 images feature validated, high-quality wide radiator cooling cores and header assemblies (mean area $0.045$, mean aspect ratio $8.34$).
- **Collage & Redundancy Removal:**
  - 1 horizontal two-panel collage (`819_jpg`, 2 boxes) excluded.
  - 1 vertical two-panel collage (`602_jpg`, 1 box) excluded.
  - 1 low-resolution internal duplicate (`886_jpg`, $400\times267$, copy of `838_jpg`) purged.
- **Reproducible Group-Aware Splitting (Seed 42):** Source 2's 115 retained images were partitioned into an 80/15/5 distribution:
  - **`train`**: **92 images** (80.0%)
  - **`valid`**: **17 images** (14.8%)
  - **`test`**: **6 images** (5.2%)
  - All 4 internal near-duplicate clusters were kept intact within `train`.
  - **Zero cross-split leakage:** $D_H > 4$ across all split boundaries.
- **Unified Ontology:** 100% of retained annotations were remapped to Prototype Class ID **`3`** (`radiator`).

---

## 2. Source 1 Semantic Audit & Filtering Details

Source 1 presented a critical computer vision challenge: because its raw ontology defined only `Radiator` without a corresponding `radiator_cap` class, human annotators universally applied Class 4 to circular pressure caps.

### 2.1 Per-Box Taxonomy of Source 1 Exclusions ($N = 269$)

| Category | Box Count | Pct | Mean AR | Mean Area | Rationale |
|---|:---:|:---:|:---:|:---:|---|
| **Tiny Cap Only** | 135 | 50.2% | 1.09 | 0.0028 | Square pressure cap ($< 0.005$ area) |
| **Tiny Elongated Cap Mount** | 4 | 1.5% | 2.15 | 0.0031 | Cap flange tab ($< 0.005$ area) |
| **Cap Flange & Neck** | 124 | 46.1% | 0.96 | 0.0084 | Cap, neck, and mounting ear ($0.005 - 0.02$) |
| **Coolant Overflow Bottle** | 4 | 1.5% | 0.45 | 0.0185 | Transparent expansion reservoir |
| **Mislabeled Component** | 1 | 0.4% | 1.21 | 0.0393 | Auxiliary fuse/bracket region (`57_a99d...`) |
| **Engine Bay Placement Error** | 1 | 0.4% | 0.98 | 0.3224 | Central engine block (`1_c8KbsG2p...`) |
| **TOTAL EXCLUDED** | **269** | **100%** | **0.98** | **0.0072** | **Full documentation in `audit/source1_exclusions.csv`** |

> [!IMPORTANT]
> **Data Quality Safeguard:**
> Permitting cap-only bounding boxes into the `radiator` class would cause anchor-free and anchor-based detectors to fail when encountering the actual wide rectangular radiator core. Complete exclusion of Source 1 cap annotations protects downstream model convergence.

---

## 3. Source 2 Validation & Collage Remediation

Source 2 provided clean, high-fidelity annotations of the radiator core assembly across the front slam panel.

### 3.1 Exclusions from Source 2 ($N = 3$)
1. **`819_jpg.rf.ad6427498d2d7e6daaac820841956f91.jpg`:** Horizontal two-panel collage containing two separate engine compartments and two radiator boxes. Excluded due to split-scene artifacts.
2. **`602_jpg.rf.8362a8937d799db7104e324ef055ef5d.jpg`:** Vertical two-panel collage containing two separate vehicle bays. The radiator annotation spans across the central horizontal seam. Excluded.
3. **`886_jpg.rf.5de572abb2d39ec63a1571962ab1eaf1.jpg`:** Low-resolution thumbnail ($400\times267$) bitstream duplicate of high-resolution master `838_jpg` ($995\times663$). Purged to eliminate redundancy.

### 3.2 Retained Source 2 Population
- **Total Valid Images Retained:** **115 images**
- **Total Radiator Bounding Boxes:** **115 boxes** (1:1 cardinality per image)
- **Mean Aspect Ratio ($w/h$):** **8.34**
- **Mean Area ($w \times h$):** **0.04459** (4.46% of image frame)
- **Class Remapping:** Raw Class `9` $\rightarrow$ Prototype Class ID **`3`**

---

## 4. Duplicate & Leakage Resolution Registry

The registry [`docs/stage2_7_1_duplicate_resolution.csv`](file:///D:/AI-Vehicle-Diagnosis-System/docs/stage2_7_1_duplicate_resolution.csv) tracks 19 duplicate/leakage pairs:

| Pair ID | Dataset Source | Type | Hamming Dist | Action Taken | Rationale |
|---|---|:---:|:---:|:---:|---|
| **DUP-EXACT-01** | Source 1 | SHA-256 | 0 | Purge copy B; both excluded | Exact bitstream duplicate in `train` (`68ac5874d...`) |
| **DUP-NEAR-01** | Source 2 | dHash | 0 | Purge `886_jpg`, retain `838_jpg` | Retain sharp $995\times663$ master; remove $400\times267$ copy |
| **DUP-NEAR-02** | Source 2 | dHash | 0 | Group both in `train` | Pair `248_jpg` ($2048\times2048$) & `266_jpg` ($600\times600$) |
| **DUP-NEAR-03** | Source 2 | dHash | 0 | Group both in `train` | Pair `504_jpg` & `536_jpg` (Different crops) |
| **DUP-NEAR-04** | Source 2 | dHash | 0 | Group both in `train` | Pair `710_jpg` ($3662\times2122$) & `747_jpg` ($1080\times625$) |
| **DUP-NEAR-05** | Source 2 | dHash | 2 | Group both in `train` | Pair `836_jpg` & `862_jpg` (Burst photos) |
| **DUP-NEAR-06 to 18** | Source 1 | dHash | $0 - 4$ | Both instances excluded | 13 cross-split leaking pairs in Source 1 eliminated |

---

## 5. Cleaned Dataset Architecture & Verification

### 5.1 Storage Layout
```
data/cleaned/stage2_7_radiator/
├── audit/
│   ├── source1_exclusions.csv      (269 records: cap/bottle exclusions)
│   └── source2_exclusions.csv      (3 records: collages & duplicate)
├── source1_epd/
│   ├── README.md                   (Audit documentation of 100% exclusion)
│   ├── images/ {train, valid, test}
│   └── labels/ {train, valid, test}
└── source2_ebp/
    ├── images/
    │   ├── train/                  (92 images, 80.0%)
    │   ├── valid/                  (17 images, 14.8%)
    │   └── test/                   (6 images, 5.2%)
    └── labels/
        ├── train/                  (92 labels, Class ID 3)
        ├── valid/                  (17 labels, Class ID 3)
        └── test/                   (6 labels, Class ID 3)
```

### 5.2 Quality Assurance Checklist
- [x] **Zero Cross-Split Leakage:** Verified via perceptual hashing ($D_H > 4$ across all splits).
- [x] **100% Image-Label Pairing:** Exactly 115 images and 115 labels in Source 2.
- [x] **YOLO Coordinate Boundaries:** All bounding boxes strictly conform to $x_c, y_c \in [0, 1]$ and $w, h \in (0, 1]$.
- [x] **Class Ontology Consistency:** 100% of labels map to Prototype Class ID **`3`**.
- [x] **Visual Evidence Generated:** 30 high-resolution visual samples rendered under [`docs/stage2_7_1_radiator_visual_review/`](file:///D:/AI-Vehicle-Diagnosis-System/docs/stage2_7_1_radiator_visual_review/).
- [x] **Automated Regression Suite:** 7 dedicated unit/integration tests in [`tests/test_radiator_cleaning.py`](file:///D:/AI-Vehicle-Diagnosis-System/tests/test_radiator_cleaning.py) passing in 3.37s.
