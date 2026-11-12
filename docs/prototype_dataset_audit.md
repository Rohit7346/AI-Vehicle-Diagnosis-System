# Prototype Dataset Filtering & Remapping Audit Report
**Source Dataset:** Engine Parts Detector v3 Improved Demo Model (CC BY 4.0)
**Source Directory:** `D:\AI-Vehicle-Diagnosis-System\data\raw\public_dataset_sample`
**Output Directory:** `D:\AI-Vehicle-Diagnosis-System\data\prototype_5class`
**Date:** November 4, 2026

---
## 1. Class Remapping & Ontology Verification

| Original Class ID | Original Class Name | Remapped Prototype ID | Prototype Class Name | Decision / Semantic Rule |
|---|---|---|---|---|
| 0 | Air Coolant | — | — | **EXCLUDED** (Out of 5-class prototype scope) |
| 1 | Battery | 1 | battery | **ACCEPTED (1:1 Remap)**: 12V automotive battery casing and terminals |
| 2 | Fuse-Box | — | — | **EXCLUDED** (Auxiliary electrical box, out of scope) |
| 3 | Main Engine | — | — | **EXCLUDED**: Engine block/casting. **NEVER** mapped to `engine_bay` |
| 4 | Radiator | 3 | radiator | **ACCEPTED (Remapped 4 -> 3)**: Front cooling radiator core & tanks |
| 5 | Windscreen Washer | — | — | **EXCLUDED** (Fluid reservoir, out of scope) |
| — | *(Missing)* | 0 | engine_bay | **UNPOPULATED** (0 instances, not invented) |
| — | *(Missing)* | 2 | alternator | **UNPOPULATED** (0 instances, not invented) |
| — | *(Missing)* | 4 | exhaust_manifold | **UNPOPULATED** (0 instances, not invented) |

## 2. Dataset Split Preservation & Image Counts

| Split | Source Images | Retained Images | Skipped Images | Battery (ID 1) | Radiator (ID 3) | Total Retained Annotations | Excluded Annotations |
|---|---|---|---|---|---|---|---|
| **train** | 1,339 | **985** | 354 | 945 | 587 | **1,532** | 2,432 |
| **val** | 137 | **101** | 35 | 99 | 62 | **161** | 263 |
| **test** | 83 | **61** | 22 | 59 | 31 | **90** | 142 |
| **Total** | **1,559** | **1,147** | **411** | **1,103** | **680** | **1,783** | **2,837** |

*Note: Validation image count updated from 102 to 101 and Battery annotations from 1,104 to 1,103 following Stage 2.1 deduplication.*

## 3. Integrity & Quality Guardrail Checks

| Check Description | Metric Value | Verification Status |
|---|---|---|
| **Empty Label Files Created** | 0 | **PASSED (0 empty files created)** |
| **Missing Image / Label Pairs** | 0 | **PASSED (100% paired, 0 missing)** |
| **Annotation Syntax Errors** | 0 | **PASSED (0 syntax/parsing errors)** |
| **Invalid Bounding Boxes** | 0 | **PASSED (0 out-of-bounds, 0 degenerate)** |
| **Pipeline Cross-Split Leakage** | 0 cross-split duplicates | **PASSED (Strict split isolation)** |

## 4. Upstream Source Dataset Leakage Finding

> [!NOTE]
> During cryptographic SHA-256 hash auditing (`AnnotationValidator.detect_data_leakage`), a single identical source image was detected in both `train` and `val`:
> - **Train:** `istockphoto-953763812-612x612_jpg.rf.7d2678a0a53155decffaf575c9276744.jpg`
> - **Val:** `istockphoto-953763812-612x612_jpg.rf.2a0e3dc60a7d4104071c71fdf0efb53b.jpg`
> - **SHA-256:** `e75fee7735d1d9e29c1507adb0a27c7075bc2833ce6bb151ccfcc803db3bffcc`
> 
> This duplicate originated upstream in the public Roboflow archive. Per Task 5 requirement (*'Preserve train, validation and test splits. Do not move images between splits'*), this split structure was preserved exactly. It can be safely purged from `val` during full model training if strict zero-leakage deduplication is desired.

## 5. Skip Decision Audit Sample

Below is a representative audit of images skipped because they contain zero accepted prototype annotations:

- **[train]** `06hr923fn1tnbjyh0mgxn5pamolk270d_jpg.rf.00f8842643edd43fb1cdcd0232b0c124.jpg`: *No accepted prototype annotations (only excluded classes)*
- **[train]** `06hr923fn1tnbjyh0mgxn5pamolk270d_jpg.rf.80e7648609f152ba2bbe20f3999b77d1.jpg`: *No accepted prototype annotations (only excluded classes)*
- **[train]** `06hr923fn1tnbjyh0mgxn5pamolk270d_jpg.rf.ce3d05de8146b0a7239fb533550280f3.jpg`: *No accepted prototype annotations (only excluded classes)*
- **[train]** `2007-Honda-Civic-LX-1-8L-4-Cyl-Sedan-4-Door-_Fuse-Engine-Part-2_png.rf.948ef8844b1bba99a9815901d15f863e.jpg`: *No accepted prototype annotations (only excluded classes)*
- **[train]** `2007-Honda-Civic-LX-1-8L-4-Cyl-Sedan-4-Door-_Fuse-Engine-Part-2_png.rf.c801e9b26e59cb4dfe767e64d7ef494d.jpg`: *No accepted prototype annotations (only excluded classes)*
- **[val]** `31-cNZmQqDL_jpg.rf.6f54ac09fdb3df34027dbc40b813a942.jpg`: *No accepted prototype annotations (only excluded classes)*
- **[val]** `3178eq4pDpL-_SR600-315_PIWhiteStrip-BottomLeft-0-35_SCLZZZZZZZ_FMpng_BG255-255-255_png.rf.705d15f090a54ac5a8896a2d24ac7e8e.jpg`: *No accepted prototype annotations (only excluded classes)*
- **[val]** `319Fa7GDjtL_jpg.rf.cea048f6ebbbc1b1dcf63e175fe2fd12.jpg`: *No accepted prototype annotations (only excluded classes)*

