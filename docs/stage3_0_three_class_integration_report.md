# Stage 3.0: Three-Class Dataset Integration Report

**Project Title:** AI-Powered Multimodal Vehicle Fault Detection and Diagnosis System  
**Stage:** Stage 3.0 – Three-Class Dataset Integration and Final Training Readiness Audit  
**Target Architecture:** YOLOv8 Object Detection (Ultralytics)  
**Integrated Dataset Root:** [`data/processed/stage3_0_three_class/`](file:///D:/AI-Vehicle-Diagnosis-System/data/processed/stage3_0_three_class/)  
**Configuration File:** [`data/processed/stage3_0_three_class/data.yaml`](file:///D:/AI-Vehicle-Diagnosis-System/data/processed/stage3_0_three_class/data.yaml)  
**Distribution Summary:** [`docs/stage3_0_class_distribution.csv`](file:///D:/AI-Vehicle-Diagnosis-System/docs/stage3_0_class_distribution.csv)  
**Duplicate Resolution Registry:** [`docs/stage3_0_duplicate_resolution.csv`](file:///D:/AI-Vehicle-Diagnosis-System/docs/stage3_0_duplicate_resolution.csv)  
**Training Readiness Audit:** [`docs/stage3_0_training_readiness_report.md`](file:///D:/AI-Vehicle-Diagnosis-System/docs/stage3_0_training_readiness_report.md)  
**Integration Test Suite:** [`tests/test_three_class_integration.py`](file:///D:/AI-Vehicle-Diagnosis-System/tests/test_three_class_integration.py)  
**Execution Date:** November 11, 2026  

---

## 1. Executive Summary & Integration Scope

Stage 3.0 establishes the first unified multi-class object detection dataset for the Vehicle Component Detection pipeline. Prior stages focused on rigorous class-by-class data curation, semantic remediation, and duplicate filtering:
- **Stage 2.5 & 2.5.1:** Battery audit and cleaning (465 automotive battery images; malformed polygon resolved; cross-split near duplicates clustered).
- **Stage 2.6:** Engine bay validation (191 annotated engine compartments; 1 hard negative sample).
- **Stage 2.7 & 2.7.1:** Radiator audit and semantic remediation (100% of Source 1 cap-only boxes excluded; 115 wide radiator cooling assemblies retained; 2 collages excluded).

### Core Deliverables Achieved:
1. **Unified 3-Class Ontology Established:**
   - Class `0`: `engine_bay`
   - Class `1`: `battery`
   - Class `2`: `radiator`
2. **Unified Dataset Integration:**
   - Successfully merged 696 unique physical images and 771 bounding boxes into a standardized YOLOv8 directory structure at [`data/processed/stage3_0_three_class/`](file:///D:/AI-Vehicle-Diagnosis-System/data/processed/stage3_0_three_class/).
   - 100% image-to-label pairing integrity (696 images : 696 label files).
   - Generated valid YOLOv8 [`data.yaml`](file:///D:/AI-Vehicle-Diagnosis-System/data/processed/stage3_0_three_class/data.yaml).
3. **Multi-Class Co-occurrence Consolidation:**
   - 76 physical vehicle images from Source 2 (`engine-bay-parts`) contain both an automotive battery and a radiator cooling assembly.
   - Automatically merged distinct source label annotations into unified multi-line label text files without overwriting.
4. **Annotation Format Harmonization:**
   - 185 engine bay polygon annotations exported from Roboflow were deterministically converted into standard 5-token YOLO bounding boxes $(x_c, y_c, w, h)$.
   - All bounding box coordinates validated strictly within $[0.0, 1.0]$.
5. **Absolute Elimination of Data Leakage:**
   - Performed exhaustive perceptual hash ($dHash$) auditing across all split boundaries (113,900 cross-split image pairs evaluated).
   - Reassigned the Chevrolet Spin validation near-duplicate to train (`RES-SPIN-01`).
   - Realigned 23 Source 2 multi-class split discrepancies into unified target splits (`RES-REASSIGN-02` through `RES-REASSIGN-23`).
   - Zero cross-split leakage pairs ($D_H \le 4$) exist in the integrated dataset.

---

## 2. Unified Class Mapping & Provenance Architecture

```
+----------------------------------------------------------------------------------------------------+
|                                    STAGE 3.0 INTEGRATION PIPELINE                                  |
+----------------------------------------------------------------------------------------------------+
|                                                                                                    |
|  [Stage 2.6 Engine Bay]                                                                            |
|  - Source: Vehicle Component Detection                                                             |
|  - 191 images / 191 boxes / 1 hard negative                                                        |
|  - Source Class ID: 0 -----------------------------------------> Class 0: engine_bay               |
|  - Roboflow polygons converted to standard YOLO 5-token boxes                                      |
|                                                                                                    |
|  [Stage 2.5.1 Battery]                                                                             |
|  - Source 1: Engine Parts Detector v1 (117 images / 117 boxes)                                     |
|  - Source 2: engine-bay-parts v1 (348 images / 348 boxes)                                          |
|  - Source Class ID: 1 -----------------------------------------> Class 1: battery                  |
|                                                                                                    |
|  [Stage 2.7.1 Radiator]                                                                            |
|  - Source 2: engine-bay-parts v1 (115 images / 115 boxes)                                          |
|  - Source 1: Excluded 269 cap-only boxes                                                           |
|  - Prototype Class ID: 3 --------------------------------------> Class 2: radiator                 |
|                                                                                                    |
|  [Multi-Class Reconciliation]                                                                      |
|  - 76 shared images between Battery and Radiator in Source 2                                       |
|  - Merged into unified multi-box label files (Class 1 + Class 2)                                   |
|                                                                                                    |
+----------------------------------------------------------------------------------------------------+
|                                  OUTPUT: data/processed/stage3_0_three_class/                      |
|                                  - train: 480 images (527 bboxes)                                  |
|                                  - val:   146 images (167 bboxes)                                  |
|                                  - test:  70 images  (77 bboxes)                                   |
|                                  - TOTAL: 696 unique images | 771 bboxes                           |
+----------------------------------------------------------------------------------------------------+
```

### Class Ontology Remapping Matrix

| Unified Class ID | Class Name | Component Description | Source Dataset | Source Class ID | Remapping Rule |
|:---:|---|---|---|:---:|---|
| **`0`** | `engine_bay` | Visible engine compartment under open hood | Stage 2.6 Export | `0` | Direct map ($0 \rightarrow 0$); convert polygon to bbox |
| **`1`** | `battery` | Automotive 12V lead-acid / starter battery | Stage 2.5.1 Cleaned | `1` | Direct map ($1 \rightarrow 1$) |
| **`2`** | `radiator` | Complete radiator core and header assembly | Stage 2.7.1 Cleaned | `3` | Remap from prototype ID ($3 \rightarrow 2$) |

> [!NOTE]
> Prototype classes `2` (`alternator`) and `4` (`exhaust_manifold`) are deferred to Stage 4 collection as planned. The 3-class training ontology is compact, high-precision, and directly models the primary macro structure (`engine_bay`) and two essential service components (`battery`, `radiator`).

---

## 3. Integrated Dataset Distribution

The dataset was partitioned using deterministic, group-aware assignment to ensure zero data leakage while preserving representative class distributions across splits:

### 3.1 Split Distribution Summary Table

| Partition | Total Images | Image Pct | Total BBoxes | Engine Bay (`0`) | Battery (`1`) | Radiator (`2`) | Multi-Class Images | Multi-Box Images | Hard Negatives |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **`train`** | **480** | 68.97% | **527** | 136 | 312 | 79 | 47 | 47 | 0 |
| **`val`** | **146** | 20.98% | **167** | 37 | 103 | 27 | 21 | 21 | 0 |
| **`test`** | **70** | 10.06% | **77** | 18 | 50 | 9 | 8 | 8 | 1 |
| **TOTAL** | **696** | **100.0%** | **771** | **191** | **465** | **115** | **76** | **76** | **1** |

### 3.2 Detailed Class Representation Breakdown

- **Engine Bay (Class 0):**
  - Total Bounding Boxes: 191
  - Distribution: Train: 136 (71.2%), Val: 37 (19.4%), Test: 18 (9.4%).
  - Hard Negative: 1 image (`Ford-Ranger-XLT-4x4-3-2-Autoentusiastas-22_jpg.rf.961a87612f0e300ceab9b386ee3aa6ee.jpg`) in `test` split containing an empty 0-byte `.txt` label file to test background false-positive resistance.
- **Battery (Class 1):**
  - Total Bounding Boxes: 465
  - Distribution: Train: 312 (67.1%), Val: 103 (22.2%), Test: 50 (10.8%).
  - Sourced from Engine Parts Detector v1 (117) and engine-bay-parts v1 (348).
- **Radiator (Class 2):**
  - Total Bounding Boxes: 115
  - Distribution: Train: 79 (68.7%), Val: 27 (23.5%), Test: 9 (7.8%).
  - Sourced exclusively from engine-bay-parts v1 (verified wide radiator cores).
- **Multi-Class Co-occurrence:**
  - 76 vehicle images depict both an automotive battery and a radiator assembly within the same visual scene.
  - Distribution: Train: 47 images, Val: 21 images, Test: 8 images.
  - Multi-class label files contain exactly 2 lines: one for Class 1 and one for Class 2.

---

## 4. Technical Integration Challenges & Remediations

### 4.1 Multi-Class Image Reconciliation
In Source 2 (`engine-bay-parts v1`), battery and radiator annotations originated from the same raw image pool. In prior stages (2.5.1 and 2.7.1), these classes were cleaned and partitioned independently. Consequently, 23 images had split discrepancies between the two standalone datasets (e.g., an image was assigned to `train` in the battery dataset but `val` in the radiator dataset due to independent Roboflow random hashing).

**Remediation:**
1. Detected all shared image filenames across the cleaned battery and radiator directories.
2. Formatted a master split resolution rule: If an image is assigned to `val` or `test` in either source, it was promoted to that evaluation split, and its corresponding near-duplicate cluster was locked to prevent cross-split leakage.
3. Created multi-box label files by appending remapped Class 2 bounding boxes to Class 1 files:
   ```text
   # Sample multi-class label file: 1018_jpg.rf.859b48c3b01a6136ef829ec9788f62f5.txt
   1 0.441865 0.771875 0.177063 0.134896
   2 0.505342 0.380729 0.407986 0.057292
   ```
4. All 23 split reconciliations were cataloged in [`docs/stage3_0_duplicate_resolution.csv`](file:///D:/AI-Vehicle-Diagnosis-System/docs/stage3_0_duplicate_resolution.csv) (`RES-REASSIGN-02` through `RES-REASSIGN-23`).

### 4.2 Roboflow Polygon-to-Bounding-Box Conversion
Stage 2.6 engine bay annotations were exported from Roboflow with segmentation masks / polygons (token lengths $> 5$, e.g., 29 tokens per line). Standard YOLOv8 object detection (`yolov8n.pt`) requires 5-token bounding box format: `class x_center y_center width height`.

**Remediation:**
1. Parsed arbitrary $2k$-polygon point sets $(x_1, y_1, x_2, y_2, \dots, x_k, y_k)$.
2. Extracted axis-aligned bounding bounds:
   $$x_{\min} = \min(x_i), \quad x_{\max} = \max(x_i), \quad y_{\min} = \min(y_i), \quad y_{\max} = \max(y_i)$$
3. Computed normalized center coordinates and dimensions:
   $$x_c = \frac{x_{\min} + x_{\max}}{2}, \quad y_c = \frac{y_{\min} + y_{\max}}{2}, \quad w = x_{\max} - x_{\min}, \quad h = y_{\max} - y_{\min}$$
4. Clamped all normalized coordinates strictly to $[0.0, 1.0]$.
5. Converted all 185 polygon files into valid 5-token YOLO bounding boxes.

### 4.3 Windows Path Length Limitation ($> 260$ Characters)
Several Roboflow engine bay filenames generated excessively long paths when nested inside `data/processed/stage3_0_three_class/images/train/` (up to 277 characters), triggering `FileNotFoundError: [Errno 2]` in standard Windows APIs:
- e.g., `compartimento-do-motor-de-coleta-toyota-hilux-isolado-no-fundo-branco-visao-superior-do-motor-aberto-do-carro_jpg.rf.18e3c8a76797b6eda69827af966ea857.jpg`

**Remediation:**
Integrated `make_long_path()` utility prepending the Windows extended-path prefix `\\?\` to all file copy, write, and PIL hash read operations, ensuring 100% I/O stability on Windows 11.

---

## 5. Duplicate Detection & Cross-Split Leakage Audit

A comprehensive duplicate audit was performed across all 696 images in the integrated dataset using SHA-256 (exact duplicate check) and perceptual difference hashing ($dHash$, near-duplicate threshold $D_H \le 4$).

### 5.1 Exact Duplicate Analysis
- **Internal Exact Duplicates:** 0 found. Every image file across `train`, `val`, and `test` has a unique SHA-256 hash.
- **Cross-Split Exact Duplicates:** 0 found.

### 5.2 Near-Duplicate Analysis & Group Clustering
- **Chevrolet Spin Resolution (`RES-SPIN-01`):**
  - Identified in Stage 2.6 audit: `sob-o-capo-todas-as-versoes-do-spin-...408a...jpg` (train) and `...683c...jpg` (val) shared $D_H = 4$.
  - In Stage 3.0 integration, the validation copy was deterministically reassigned to `train`. Both copies now reside strictly in `train`.
- **Near-Duplicate Cluster Resolution (`RES-CLUSTER-01`):**
  - `710_jpg` (master $3662\times2122$) and `747_jpg` (web preview $1080\times625$) share perceptual identity ($D_H = 0$).
  - Both images were locked to `train` split, eliminating cross-split contamination.

### 5.3 Global Cross-Split Leakage Evaluation
Every image in each split was compared against every image in every other split:
- **`train` vs `val`:** $480 \times 146 = 70,080$ comparisons $\rightarrow$ **0 pairs with $D_H \le 4$**.
- **`train` vs `test`:** $480 \times 70 = 33,600$ comparisons $\rightarrow$ **0 pairs with $D_H \le 4$**.
- **`val` vs `test`:** $146 \times 70 = 10,220$ comparisons $\rightarrow$ **0 pairs with $D_H \le 4$**.
- **Total Cross-Split Comparisons:** **113,900 pairs**.
- **Result:** **ZERO cross-split data leakage detected.**

---

## 6. Dataset Structure & Configuration Validation

### 6.1 Directory Tree Verification
```
data/processed/stage3_0_three_class/
|-- data.yaml
|-- images/
|   |-- train/    [480 files]
|   |-- val/      [146 files]
|   `-- test/     [70 files]
`-- labels/
    |-- train/    [480 files, 527 bboxes]
    |-- val/      [146 files, 167 bboxes]
    `-- test/     [70 files, 77 bboxes, 1 empty label]
```

### 6.2 Configuration File (`data.yaml`)
```yaml
path: D:/AI-Vehicle-Diagnosis-System/data/processed/stage3_0_three_class
train: images/train
val: images/val
test: images/test

nc: 3
names: ['engine_bay', 'battery', 'radiator']
```

The configuration follows official Ultralytics YOLOv8 specifications with absolute root path and relative split paths, ensuring seamless cross-platform execution.

---

## 7. Automated Test Suite Results

A dedicated automated test suite was authored at [`tests/test_three_class_integration.py`](file:///D:/AI-Vehicle-Diagnosis-System/tests/test_three_class_integration.py) covering 10 comprehensive validation categories:

| Test ID | Test Function | Target Property | Result |
|---|---|---|:---:|
| `TC-3.0-01` | `test_dataset_directory_structure_and_counts` | Verifies existence of all 6 subdirectories and exact image counts (480/146/70) | **PASS** |
| `TC-3.0-02` | `test_image_label_pairing_and_stems` | Verifies 1:1 image-to-label pairing and identical file stems across all splits | **PASS** |
| `TC-3.0-03` | `test_valid_class_ids_and_counts` | Verifies all class IDs $\in \{0, 1, 2\}$ and exact bbox counts (191 / 465 / 115) | **PASS** |
| `TC-3.0-04` | `test_normalized_coordinate_boundaries` | Asserts all coordinates $(x_c, y_c, w, h) \in [0.0, 1.0]$ and dimensions $> 0$ | **PASS** |
| `TC-3.0-05` | `test_empty_label_negative_sample` | Asserts exactly 1 empty label file exists in `test` (Ford Ranger negative) | **PASS** |
| `TC-3.0-06` | `test_multi_class_and_multi_box_images` | Asserts exactly 76 multi-class images exist with proper Class 1 & 2 coexistence | **PASS** |
| `TC-3.0-07` | `test_zero_cross_split_data_leakage` | Evaluates $dHash$ across all 113,900 cross-split pairs; asserts 0 pairs $D_H \le 4$ | **PASS** |
| `TC-3.0-08` | `test_yaml_configuration_validity` | Validates YAML structure, `nc: 3`, class names, and valid path resolution | **PASS** |
| `TC-3.0-09` | `test_image_readability_and_channels` | Asserts all 696 images are valid, uncorrupted RGB/Grayscale images via PIL | **PASS** |
| `TC-3.0-10` | `test_reproducibility_and_duplicate_registry` | Asserts duplicate resolution registry contains all resolved split cases | **PASS** |

**Test Execution Summary:**
- Suite: `pytest tests/test_three_class_integration.py -v`
- Total Tests: **10**
- Passed: **10**
- Failed: **0**
- Execution Duration: ~23.8 seconds

---

## 8. Summary Table of Retained Datasets & Provenance

| Component Class | Unified ID | Primary Source | License | Original Box Count | Retained Box Count | Retained Image Count |
|---|:---:|---|---|:---:|:---:|:---:|
| `engine_bay` | `0` | Vehicle Component Detection (Stage 2.6) | Roboflow / CC BY 4.0 | 191 | 191 | 191 (+1 negative) |
| `battery` | `1` | Engine Parts Detector v1 + engine-bay-parts v1 | CC BY 4.0 / MIT | 465 | 465 | 465 |
| `radiator` | `2` | engine-bay-parts v1 (Stage 2.7.1) | MIT | 119 | 115 | 115 |
| **INTEGRATED** | `0, 1, 2` | **Unified Multi-Class Pipeline** | **Open Source (MIT / CC BY 4.0)** | **775** | **771** | **696 unique** |

---

## 9. Next Steps

With the three-class dataset fully constructed, validated, and verified through automated tests, the pipeline is ready for Stage 3.1:
1. Conduct formal Training Readiness Audit ([`docs/stage3_0_training_readiness_report.md`](file:///D:/AI-Vehicle-Diagnosis-System/docs/stage3_0_training_readiness_report.md)).
2. Formalize GO decision for Stage 3.1 YOLOv8 Nano model training.
