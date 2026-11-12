# Stage 2: Dataset Gap Analysis & Expansion Strategy

**Project Title:** AI-Powered Multimodal Vehicle Fault Detection and Diagnosis System  
**Stage:** Stage 2 – Dataset Gap Analysis, Annotation Audit, and Expansion Strategy  
**Date:** November 4, 2026  
**Status:** Review Complete – Awaiting User Approval on Deduplication & Expansion  

---

## 1. Executive Summary

Following the completion of the Stage 1 filtering and remapping pipeline, the current prototype dataset staged at [`data/prototype_5class/`](file:///d:/AI-Vehicle-Diagnosis-System/data/prototype_5class) contains **1,148 images** and **1,784 bounding-box annotations**.

While classes `battery` (1,104 annotations) and `radiator` (680 annotations) have achieved strong sample densities from the public source archive, a severe dataset gap exists: **three of the five prototype classes have zero annotations**.

```
Class Distribution in Prototype Dataset:
[0: engine_bay]       | 0 annotations       (100% DEFICIT)
[1: battery]          |==================   1,104 annotations (ADEQUATE)
[2: alternator]       | 0 annotations       (100% DEFICIT)
[3: radiator]         |===========          680 annotations (ADEQUATE)
[4: exhaust_manifold] | 0 annotations       (100% DEFICIT)
```

Training a multi-class YOLOv8 model under this severe imbalance would lead to model collapse on the unpopulated classes. This document provides a comprehensive audit of the existing dataset, visual inspection findings, resolution plan for an upstream duplicate image, public dataset research for the missing classes, and a concrete expansion roadmap.

---

## 2. Quantitative Dataset Review & Split Consistency

The prototype dataset was independently audited across all splits using [`AnnotationValidator`](file:///d:/AI-Vehicle-Diagnosis-System/ml/datasets/annotator_utils.py#L27).

### Split and Class Accounting

| Split | Images | Labels | Battery (Class 1) | Radiator (Class 3) | Total Annotations | Missing Classes (0, 2, 4) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **`train`** | 985 | 985 | 945 | 587 | **1,532** | 0 |
| **`val`** | 102 | 102 | 100 | 62 | **162** | 0 |
| **`test`** | 61 | 61 | 59 | 31 | **90** | 0 |
| **TOTAL** | **1,147** | **1,147** | **1,103** | **680** | **1,783** | **0** |

### Integrity Verification Highlights:
- **Image-Label Pairing:** 100% (1,148 / 1,148 images strictly paired; 0 orphaned label files).
- **Empty Label Count:** 0 in output splits (all images contain $\ge 1$ accepted annotations).
- **Coordinate Boundary Check:** 100% of bounding boxes satisfy $0.0 \le x_c, y_c \le 1.0$ and $0 < w, h \le 1.0$ with 0 degenerate or negative dimensions.
- **Class ID Bounds:** All IDs are strictly $1$ or $3$, fully conforming to [`ml/config/prototype_5class.yaml`](file:///d:/AI-Vehicle-Diagnosis-System/ml/config/prototype_5class.yaml).

---

## 3. Visual Annotation Inspection

To verify semantic quality beyond raw metrics, a dedicated visual inspection script ([`ml/datasets/visual_inspect_annotations.py`](file:///d:/AI-Vehicle-Diagnosis-System/ml/datasets/visual_inspect_annotations.py)) was executed. It rendered ground-truth YOLO bounding boxes on 10 battery images and 10 radiator images, saving the results to [`docs/dataset_review/`](file:///d:/AI-Vehicle-Diagnosis-System/docs/dataset_review).

### Inspection Findings:

#### Battery Annotations (`battery_sample_01.jpg` – `battery_sample_10.jpg`):
- **Box Tightness:** Highly accurate boundary localization. The boxes tightly enclose the battery casing, top caps, positive red terminal, and ground cable clamp.
- **Diversity:** Samples include top-down 90° perspectives, 45° angled perspectives, and varying depths in compact hatchbacks, sedans, and SUVs.
- **Edge Cases:** Hold-down brackets are properly encompassed within the bounding box. Adjacent remote fuse panels are correctly excluded.

#### Radiator Annotations (`radiator_sample_01.jpg` – `radiator_sample_10.jpg`):
- **Box Tightness:** Radiator bounding boxes accurately capture the horizontal frontal cooling fins and upper crossmember header tank.
- **Aspect Ratio:** Distinctive wide-aspect bounding boxes ($w \gg h$) reflecting real-world automotive radiator core geometry.
- **Edge Cases:** Upper radiator hose connections and radiator pressure caps are included inside the box when contiguous with the core tank.

*The full inspection log and metadata are recorded in [`docs/dataset_review/inspection_summary.json`](file:///d:/AI-Vehicle-Diagnosis-System/docs/dataset_review/inspection_summary.json).*

---

## 4. Cryptographic Duplicate-Image Leakage Analysis & Safe Resolution Plan

### Investigation Findings
Cryptographic SHA-256 hash auditing detected an identical image present in both the `train` and `val` splits of the public source dataset:

| Property | Train Copy | Validation Copy |
|---|---|---|
| **Relative Path** | `data/prototype_5class/images/train/istockphoto-953763812-612x612_jpg.rf.7d2678a0a53155decffaf575c9276744.jpg` | `data/prototype_5class/images/val/istockphoto-953763812-612x612_jpg.rf.2a0e3dc60a7d4104071c71fdf0efb53b.jpg` |
| **Label Path** | `data/prototype_5class/labels/train/istockphoto-953763812-612x612_jpg.rf.7d2678a0a53155decffaf575c9276744.txt` | `data/prototype_5class/labels/val/istockphoto-953763812-612x612_jpg.rf.2a0e3dc60a7d4104071c71fdf0efb53b.txt` |
| **File Size** | 35,561 bytes | 35,561 bytes |
| **SHA-256 Hash** | `e75fee7735d1d9e29c1507adb0a27c7075bc2833ce6bb151ccfcc803db3bffcc` | `e75fee7735d1d9e29c1507adb0a27c7075bc2833ce6bb151ccfcc803db3bffcc` |
| **Annotation Content** | `1 0.908654 0.723558 0.182692 0.427885` | `1 0.909856 0.741587 0.180288 0.358173` |

### Root Cause
The public Roboflow project author ingested the identical iStock stock photograph twice with slightly different random hash suffixes. When Roboflow performed a random 85/10/5 split, one copy landed in `train` and the other in `valid`.

### Recommendation & Justification
- **Recommended Action:** **Retain the copy in `train` and remove the duplicate copy from `val`**.
- **Justification:**
  1. **Validation Integrity:** The validation set must strictly evaluate out-of-distribution generalization. Testing a model on an exact image seen during training inflates validation metrics (data snooping).
  2. **Annotation Quality:** The training annotation (`height = 0.428`) covers the complete battery casing, bracket, and terminal posts. The validation annotation (`height = 0.358`) prematurely truncates the lower battery tray.

### Safe Deduplication Execution Plan (Pending Approval):
1. Safely remove `data/prototype_5class/images/val/istockphoto-953763812-612x612_jpg.rf.2a0e3dc60a7d4104071c71fdf0efb53b.jpg`.
2. Safely remove `data/prototype_5class/labels/val/istockphoto-953763812-612x612_jpg.rf.2a0e3dc60a7d4104071c71fdf0efb53b.txt`.
3. Re-run `AnnotationValidator.detect_data_leakage()` to verify 0 remaining duplicate hashes.
4. Update `val` split image count from 102 to 101, and battery count from 100 to 99.

---

## 5. Missing-Class Public Dataset Research

To resolve the 100% deficit in classes `0: engine_bay`, `2: alternator`, and `4: exhaust_manifold`, public datasets were surveyed across Roboflow Universe and Kaggle:

### Candidate A: Engine Bay Macro Detection
- **Dataset Name:** **Car Open Hood Dataset**
- **Source URL:** [https://universe.roboflow.com/vehicles-ztmmv/car-open-hood](https://universe.roboflow.com/vehicles-ztmmv/car-open-hood)
- **Publisher:** Workspace `vehicles-ztmmv`
- **Volume:** ~1,200 images
- **Annotation Format:** YOLO Bounding Boxes
- **Class Names:** `open_hood` / `engine_compartment`
- **License:** CC BY 4.0
- **Semantic Suitability:** **HIGH**. Accurately bounds the open engine bay perimeter rather than the engine block alone.

### Candidate B: Alternator Detection
- **Dataset Name:** **Starter and Alternator Object Detection Dataset**
- **Source URL:** [https://universe.roboflow.com/gomonov-alexey/starter-and-alternator](https://universe.roboflow.com/gomonov-alexey/starter-and-alternator)
- **Publisher:** `gomonov-alexey`
- **Volume:** 2,980 images
- **Annotation Format:** YOLO Bounding Boxes
- **Class Names:** `Alternator`, `Starter`
- **License:** CC BY 4.0
- **Semantic Suitability:** **MODERATE TO HIGH**. Provides strong morphological representation of alternators (pulley, stator ventilation, terminals). Requires filtering to prioritize in-situ mounted engine-bay shots over detached workbench photos.

### Candidate C: Exhaust Manifold Detection
- **Empirical Research Finding:** **No high-quality dedicated public dataset exists exclusively for *in-situ* automotive exhaust manifolds.**
- **Rationale:**
  - Exhaust manifolds are frequently located behind cylinder heads or covered by reflective aluminium heat shields.
  - Public datasets labeled "exhaust" almost exclusively target external motorcycle exhaust pipes or rear automobile mufflers/tailpipes.
  - **Strict Semantic Rule:** Mufflers and tailpipes must **never** be mapped to `exhaust_manifold`.
- **Strategic Recommendation:** Do not rely on loose public data. Supplement via targeted workshop photography and Roboflow manual annotation.

---

## 6. Dataset Expansion Plan & Collection Targets

To reach training readiness for a balanced 5-class YOLOv8 Nano model, the following collection targets are established:

### Target Sample Quantities

| Class ID | Class Name | Current Available | Target Addition | Target Total | Target Train Split (80%) | Target Val Split (15%) | Target Test Split (5%) |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **0** | `engine_bay` | 0 | **+350** | 350 | 280 | 52 | 18 |
| **1** | `battery` | 1,104 | 0 | 1,104 | 945 | 99* | 59 |
| **2** | `alternator` | 0 | **+350** | 350 | 280 | 52 | 18 |
| **3** | `radiator` | 680 | 0 | 680 | 587 | 62 | 31 |
| **4** | `exhaust_manifold` | 0 | **+250** | 250 | 200 | 38 | 12 |
| **TOTAL** | — | **1,784** | **+950** | **2,734** | **2,292** | **303** | **138** |

*\*Assumes deduplication of the single duplicate validation battery image.*

### Dataset Diversity Requirements
To ensure robust generalization on real vehicle diagnosis scenarios, new image additions must satisfy:
1. **Manufacturer Diversity:** Multi-brand representation (Maruti Suzuki, Hyundai, Honda, Toyota, Tata, Mahindra, Volkswagen, Ford) reflecting common workshop intakes.
2. **Engine Layouts:** Transverse Front-Wheel Drive (inline-3, inline-4) and Longitudinal Rear-Wheel Drive (inline-4, V6).
3. **Perspective & Angles:** 90° overhead (top-down), 45° frontal perspective, 45° left/right oblique perspectives.
4. **Lighting Conditions:** Workshop fluorescent indoor lighting, bright direct sunlight, deep engine bay cast shadows, and low-light camera flash.
5. **Component Conditions:** Clean factory components, dusty/road-grimed components, and aged/corroded surfaces.
