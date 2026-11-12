# Stage 2.3: Annotation Quality Control & Component Inventory Report

**Project Title:** AI-Powered Multimodal Vehicle Fault Detection and Diagnosis System  
**Stage:** Stage 2.3 – Multi-Class Annotation and Workshop Data Collection  
**Date:** November 4, 2026  
**Auditor Engine:** `ml/datasets/audit_staged_images.py`  
**Current Dataset Target:** 5-Class Prototype Object Detector (`yolov8n`)  

---

## 1. Executive Summary & Component Inventory

Stage 2.3 establishes the multi-class annotation framework and workshop collection infrastructure for the 5-class prototype. The system strictly separates raw staging from the clean prototype dataset:

| Class ID | Component Class | Prototype Status | Staging / Intake Status | Collection Source | Immediate Action Required |
|:---:|---|:---:|:---:|---|---|
| **0** | `engine_bay` | 0 annotations | **193 verified images staged** | `Car Open Hood` (CC BY 4.0) | Import to Roboflow & annotate perimeter |
| **1** | `battery` | **1,103 annotations** | Complete & verified | Engine Parts Detector v3 (CC BY 4.0) | None (Locked & verified, zero leakage) |
| **2** | `alternator` | 0 annotations | **0 public (rejected)**<br>Target: 150–200 workshop | Dual-labeling + Workshop Intake | Dual-label on 193 engine bays + workshop capture |
| **3** | `radiator` | **680 annotations** | Complete & verified | Engine Parts Detector v3 (CC BY 4.0) | None (Locked & verified, zero leakage) |
| **4** | `exhaust_manifold` | 0 annotations | **Target: 250 workshop** | Workshop Intake Photography | Execute 250-image workshop photography plan |

---

## 2. Pre-Import Quality Audit of Staged Engine Bay Images

The 193 engine bay images staged in [`data/raw/stage2_collection/engine_bay/images/`](file:///d:/AI-Vehicle-Diagnosis-System/data/raw/stage2_collection/engine_bay/images/) were rigorously evaluated:

| Quality Dimension | Test Method | Threshold / Standard | Detected Flaws | Audit Result |
|---|---|---|:---:|:---:|
| **Bitstream Integrity** | PIL byte stream decode | 100% decode | 0 | **PASSED (100%)** |
| **Minimum Resolution** | Dimensional validation | $\ge 400 \times 400$ px | 0 | **PASSED (100%)** |
| **Blur Threshold** | Laplacian variance $\sigma^2$ | $\sigma^2 \ge 60.0$ | 0 | **PASSED (100%)** |
| **Exposure Balance** | Mean grayscale luminance | $\mu \in [25.0, 240.0]$ | 0 | **PASSED (100%)** |
| **Exact Duplicates** | SHA-256 cryptographic hash | Zero collisions | 0 | **PASSED (100%)** |
| **Near-Duplicates** | 64-bit dHash Hamming distance | $D_H > 4$ | 0 | **PASSED (100%)** |

### Complete Ingestion Accounting:
- **Total Raw Images Inspected:** 633 images (from `Car Open Hood.v1i.folder.zip`).
- **Accepted High-Quality Staged:** **193 unique images**.
- **Total Rejected Candidates:** 440 images.
  - Closed-hood vehicle exteriors: 15 images.
  - Undersized resolution ($< 400$ px): 87 images.
  - Excessive blur ($\sigma^2 < 60.0$): 3 images.
  - Near-duplicate augmented copies: 350 images.

---

## 3. Roboflow Annotation Quality Control Protocol (Task 5)

To maintain gold-standard ground truth quality during the Roboflow annotation session in the `Vehicle Component Detection` project, all annotations must pass the following six quality gates:

### Gate 1: Exact Zero-Indexed Class Ontology
Ensure that bounding boxes use the exact lowercase label names configured in `ml/config/prototype_5class.yaml`:
- `0: engine_bay`
- `1: battery`
- `2: alternator`
- `3: radiator`
- `4: exhaust_manifold`

*(Do NOT create new labels like `engine`, `motor`, `car_hood`, `ac_compressor`, or `starter`).*

### Gate 2: Bounding Box Tightness & Boundary Discipline
- **Tight Fit:** Bounding boxes must touch the outermost pixels of the target hardware without loose whitespace or background padding.
- **Engine Bay Boundary:** Frame the internal cavity defined by the inner fender aprons, firewall bulkhead, and front crossmember. Never include the exterior painted hood skin, front bumper, or ground.
- **Alternator Boundary:** Include the cylindrical slotted stator housing and the front serpentine drive pulley.

### Gate 3: Multi-Component Co-Occurrence
A single photograph may contain multiple target classes. In a typical open-hood photograph:
- `engine_bay` is annotated as the bounding enclosure.
- `battery`, `radiator`, and `alternator` are annotated as internal sub-components nested inside or adjacent to the engine bay.
- Annotators must systematically check each image for all five classes.

### Gate 4: Occlusion & Partial Visibility Threshold
- **Threshold Rule:** If a component is $\ge 30\%$ visible and identifiable by its distinctive mechanical features, annotate the visible portion.
- **Occluded Rule:** If a component is $> 70\%$ covered by an engine decorative cover, intake resonator, or wiring harness, **do NOT fabricate a label.**

### Gate 5: Human Review Queue for Uncertain Objects
- Any ambiguous component (e.g. cannot verify whether a lower accessory is an alternator or an AC compressor) must NOT be assigned a speculative label.
- Flag the image filename in `dual_labeling_metadata.csv` with `review_status: FLAGGED_FOR_AUDIT` for senior automotive review.

### Gate 6: Zero Synthetic Annotation Tolerance
- Ground truth must reflect genuine optical evidence.
- No automated heuristics or synthetic bounding box generators may be applied without 100% human visual sign-off.

---

## 4. Remaining Dataset Gaps & Forward Integration Roadmap

| Component Class | Current Staged Volume | Target Deficit | Planned Remediation | Target Completion Date |
|---|:---:|:---:|---|---|
| **Class 0: `engine_bay`** | 193 staged | 157 | Roboflow annotation of staged 193 + supplemental workshop capture | Immediate |
| **Class 2: `alternator`** | 0 staged | 200 | Dual-labeling on 193 engine bays + 150–200 workshop photos | Workshop Intake Phase |
| **Class 4: `exhaust_manifold`** | 0 staged | 250 | Direct 250-image workshop photography plan | Workshop Intake Phase |

### Verification Protocol Before Final Dataset Integration:
1. Complete human annotation and export from Roboflow in YOLOv8 PyTorch format.
2. Run `ml/datasets/audit_staged_images.py` on all newly captured workshop batches.
3. Run `AnnotationValidator.validate_dataset()` and `AnnotationValidator.detect_data_leakage()` before merging into `data/prototype_5class/`.
4. Verify locked 80% train / 15% val / 5% test partition balance.
