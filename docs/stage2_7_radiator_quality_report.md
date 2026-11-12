# Stage 2.7: Radiator Dataset Quality Engineering & Risk Evaluation Report

**Project Title:** AI-Powered Multimodal Vehicle Fault Detection and Diagnosis System  
**Stage:** Stage 2.7 – Radiator Dataset Quality Engineering  
**Component Focus:** Class 3 (`radiator`)  
**Audited Sources:** `Engine Parts Detector.v1i.yolov8.zip` & `engine-bay-parts.v1i.yolov8.zip`  
**Distribution CSV:** [`docs/stage2_7_radiator_class_distribution.csv`](file:///D:/AI-Vehicle-Diagnosis-System/docs/stage2_7_radiator_class_distribution.csv)  
**Detailed Audit Log:** [`docs/stage2_7_radiator_dataset_audit.md`](file:///D:/AI-Vehicle-Diagnosis-System/docs/stage2_7_radiator_dataset_audit.md)  
**Visual Review Samples:** [`docs/stage2_7_radiator_visual_review/`](file:///D:/AI-Vehicle-Diagnosis-System/docs/stage2_7_radiator_visual_review/)  
**Date:** November 11, 2026  

---

## 1. Quality Engineering Scorecards

Both candidate dataset archives were evaluated across seven computer vision quality criteria:

| Quality Dimension | Evaluation Standard | Source 1: Engine Parts Detector v1 | Source 2: engine-bay-parts v1 |
|---|---|:---:|:---:|
| **1. Domain Authenticity** | Real automotive passenger vehicle engine compartments | **100% PASS** (Authentic engine bays) | **100% PASS** (Authentic engine bays) |
| **2. Semantic Boundary (Task 12)** | Identifies full radiator assembly rather than only the cap/bottle | **48.3% DEFECT** (51.7% label only the radiator cap) | **99.2% PASS** (Separates radiator from cap and reservoir) |
| **3. Coordinate & Pairing Validity** | Normalized YOLO bbox format $[0, 1]$ and 1:1 image/label matching | **99.75% PASS** (1 polygon in non-radiator label) | **100% PASS** (Zero syntax errors) |
| **4. Resolution Quality** | Minimum $400 \times 400$ px resolution | **95.25% PASS** (19 undersized images) | **95.0% PASS** (10 undersized images) |
| **5. Split Integrity** | Disjoint train/valid/test partitions without data leakage | **75.0% DEFECT** (13 cross-split leaking pairs) | **50.0% DEFECT** (100% train only; val/test empty) |
| **6. Duplicate Discipline** | Zero identical or perceptual near-duplicate redundancy | **94.5% WARNING** (1 SHA-256 group; 21 near-dups) | **97.5% PASS** (0 SHA-256 dups; 5 internal pairs) |
| **7. Composition Consistency** | Single engine-compartment scene per photograph | **100% PASS** (All single scenes) | **99.5% WARNING** (1 two-panel collage: `819_jpg`) |
| **OVERALL QUALITY SCORE** | Weighted composite score (0–100) | **76 / 100 (REQUIRES REMEDIATION)** | **88 / 100 (HIGH SEMANTIC QUALITY)** |

---

## 2. In-Depth Risk Assessment & Root Cause Analysis

### 2.1 Critical Risk 1: The "Radiator vs Radiator Cap" Semantic Divergence (Task 12)
- **The Finding:** 
  - In Source 1 (`Engine Parts Detector v1`), **51.7% of radiator bounding boxes (139 of 269)** have an area smaller than $0.005$ (mean area $0.00715$, mean aspect ratio $0.98$). Visual review reveals that these boxes annotate **only the circular pressure cap or neck filler**.
  - In Source 2 (`engine-bay-parts v1`), the mean box area is **$0.04459$ ($6.2\times$ larger)** with a mean aspect ratio of **$8.34$**. Visual review confirms that Source 2 annotates the **complete front slam panel, top tank, and radiator cooling core**.
- **Root Cause:** 
  - Source 2 included dedicated classes for `Radiator Cap` (Class 10) and `Engine Coolant Reservoir` (Class 3), leaving Class 9 to strictly represent the radiator core assembly.
  - Source 1 omitted a cap class entirely, causing annotators to treat the cap as a proxy label for the radiator.
- **Model Impact If Unresolved:** 
  - Training a single `radiator` detector on uncleaned pooled data would train the model on mutually contradictory features: a tiny $20 \times 20$ px circular cap vs a wide $500 \times 60$ px horizontal grille-support bar. This bimodal aspect ratio distribution ($0.98$ vs $8.34$) severely degrades IoU convergence and causes frequent false-negative / split-detection errors.

---

### 2.2 Critical Risk 2: Cross-Split Near-Duplicate Leakage (Source 1)
- **The Finding:** Source 1 contains **13 cross-split perceptual near-duplicate pairs** ($D_H \le 4$), linking images across training, validation, and test sets.
- **Root Cause:** Roboflow automated hash-splitting failed to identify near-identical photos taken from slight perspective shifts or different camera exposures during the same photography session.
- **Model Impact:** Validation and test mAP metrics on Source 1 would be artificially inflated, masking overfitting on vehicle bay geometries.

#### Cross-Split Leakage Inventory:

| Pair # | Split A Image | Split B Image | Hamming Dist ($D_H$) |
|:---:|---|---|:---:|
| 1 | `test/...800px-Honda_K24A_Engine_01...` | `train/...1280px-Honda_K24A_Engine_01...` | 0 (Identical capture, different resize) |
| 2 | `test/...1a92c257942f0d0cdf22c114b0d...` | `train/...Honda_City_V_Ext-26...` | 0 (Duplicate upload) |
| 3 | `test/...2020_honda_civic_sedan_46...` | `train/...HD-wallpaper-2020-honda...` | 4 (Same photoshoot angle) |
| 4 | `train/...best-honda-engines-07...` | `valid/...1c01390de2ba4b1b84c35aa45...` | 0 (Identical capture) |
| 5 | `train/...best-honda-engines-07...` | `valid/...0e7643ff-FD2-Civic-Type-R...` | 1 (Same vehicle, slight crop) |
| 6–13 | 8 additional cross-split pairs connecting `train` and `valid` splits | | $0 \le D_H \le 3$ |

---

### 2.3 Moderate Risk 3: Empty Validation and Test Partitions (Source 2)
- **The Finding:** Source 2 contains 200 total images (118 with radiator), with 100% placed into `train/`. `valid/` and `test/` are empty.
- **Model Impact:** Source 2 cannot be evaluated for generalization in its raw exported state. It requires a stratified 80 / 15 / 5 split before integration.

---

### 2.4 Moderate Risk 4: Multi-Panel Collage Image (Source 2)
- **The Finding:** `train/images/819_jpg.rf.ad6427498d2d7e6daaac820841956f91.jpg` is a horizontal two-panel collage containing two separate engine compartments and two radiator annotations.
- **Model Impact:** Artificial dividing line in the middle introduces edge artifacts into convolutional feature maps.

---

## 3. Recommended Remediation Strategy (Stage 2.7.1)

To prepare candidate radiator data for safe integration into the 5-class prototype, the following remediation pipeline is recommended:

```
Candidate Radiator Pool (378 images / 388 bboxes)
   │
   ├── 1. Semantic Boundary Filtering
   │     ├── Retain all 118 Source 2 images (Full core annotations, AR=8.34)
   │     ├── Filter Source 1: Isolate 129 boxes with area in [0.005, 0.05]
   │     └── Exclude or re-label 139 Source 1 "cap-only" boxes (area < 0.005)
   │
   ├── 2. Exact & Near-Duplicate Deduplication
   │     ├── Remove 1 duplicate in train (`68ac5874d3800199...`)
   │     └── Re-partition Source 1 so all 13 near-duplicate pairs reside in train
   │
   ├── 3. Collage & Defect Handling
   │     ├── Exclude or split `819_jpg` multi-panel collage
   │     └── Repair 11-token polygon in `0e7643ff-FD2-Civic-Type-R...`
   │
   └── 4. Stratified Leakage-Safe Partitioning
         ├── Source 2: Generate 80% train (94 imgs) / 15% val (18 imgs) / 5% test (6 imgs)
         └── Keep near-duplicate clusters intact within single splits
```

---

## 4. Verification & Testing Confirmation

The radiator dataset audit pipeline has been automated and verified via pytest:
- Test suite: [`tests/test_radiator_audit.py`](file:///D:/AI-Vehicle-Diagnosis-System/tests/test_radiator_audit.py)
- **Status:** **9 of 9 tests passing** in 0.31s.
- Full project test suite: **47 of 47 tests passing** with 0 regressions.
