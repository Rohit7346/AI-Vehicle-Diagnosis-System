# Stage 2.2: Alternator Image Collection & Curation Specification

**Project Title:** AI-Powered Multimodal Vehicle Fault Detection and Diagnosis System  
**Stage:** Stage 2.2 – Missing Component Image Collection  
**Target Class:** Class 2: `alternator`  
**Target Ingestion Volume:** 350 in-situ images  
**Storage Destination:** `data/raw/stage2_collection/alternator/images/`  
**Original Annotations Staged:** `data/raw/stage2_collection/alternator/original_annotations/`  
**Metadata Registry:** `data/raw/stage2_collection/alternator/source_metadata.csv`  
**Date:** November 4, 2026  

---

## 1. Candidate Source Overview & Provenance

| Parameter | Value | Justification / Notes |
|---|---|---|
| **Source Repository** | [Roboflow Universe: Starter and Alternator](https://universe.roboflow.com/gomonov-alexey/starter-and-alternator) | Community automotive parts dataset |
| **Workspace / Author** | `gomonov-alexey` | Creator profile |
| **License** | **CC BY 4.0** | Commercial and research use permitted with attribution |
| **Total Scale** | 2,980 images | Contains both alternators and starter motors |
| **Task Type** | Object Detection | Provides 2D bounding boxes in YOLOv8 PyTorch `.txt` format |
| **Original Classes** | `Alternator`, `Start`, `Starter` | Class indices must be mapped; non-alternator classes discarded |

---

## 2. In-Situ Curation Protocol & Strict Exclusion Filters

Empirical verification established that 80–85% of images in this archive depict detached alternators on wooden workbenches, parts racks, or white product backdrops. To prevent domain-shift failure during real engine bay diagnosis, strict selective curation is enforced:

### Inclusion Criteria:
1. **Mounted In-Situ Context:** Alternator must be physically installed on an engine block inside or adjacent to a vehicle engine bay.
2. **Drive Belt Interaction:** The serpentine/accessory drive belt or front drive pulley must be visible.
3. **High Bounding-Box Quality:** Tight bounding box around the alternator cylindrical casing and front pulley.
4. **Original Class Filter:** Must carry original class label `Alternator`.

### Exclusion Criteria:
1. **Starters Discarded:** All images labeled `Start` or `Starter` are unconditionally excluded.
2. **Detached Workbench Objects:** Discard loose components resting on tables, shelves, wooden pallets, or plain studio backgrounds.
3. **Ambiguous Accessories:** Discard images where the alternator is indistinguishable from the lower-mounted air conditioning compressor or power steering pump.
4. **Duplicate & Near-Duplicate Curation:** Filter out identical copies and heavily augmented rotational duplicates using dHash ($\le 4$).

---

## 3. Original Annotation Preservation

Unlike the final prototype dataset where classes are remapped to $0..4$, raw source annotations from this candidate must be preserved separately without modification:
- Source `.txt` files containing original class IDs and coordinates are staged in:
  `data/raw/stage2_collection/alternator/original_annotations/`
- This ensures human verification can audit the original bounding box against the image before any remapping into prototype Class 2 (`alternator`).

---

## 4. Metadata Registry Schema

Every audited candidate image is registered in [`data/raw/stage2_collection/alternator/source_metadata.csv`](file:///d:/AI-Vehicle-Diagnosis-System/data/raw/stage2_collection/alternator/source_metadata.csv) with the following header:

```csv
filename,original_class_id,original_class_name,source_url,license,dataset_version,original_filename,selection_decision
```

- `filename`: Local staged filename
- `original_class_id`: Upstream source class integer
- `original_class_name`: `Alternator`, `Start`, or `Starter`
- `source_url`: `https://universe.roboflow.com/gomonov-alexey/starter-and-alternator`
- `license`: `CC BY 4.0`
- `dataset_version`: Upstream version tag
- `original_filename`: Raw filename from upstream archive
- `selection_decision`: `ACCEPTED_IN_SITU` or `REJECTED_<BENCHTOP|STARTER|BLURRY>`

---

## 5. Empirical Archive Verification & In-Situ Audit Results

The candidate archive `Starter and Alternator.v2i.yolov8.zip` (15.5 MB) was unpacked and evaluated against project constraints:

| Metric / Check | Value | Notes |
|---|:---:|---|
| **Archive Total Images** | 2,093 | Partitioned across train (1,466), valid (419), test (208) |
| **Archive Total Labels** | 2,093 | Preserved in `original_annotations/` |
| **Label Class 0 (`Alternator`)** | 1,175 bboxes | Present across 1,086 images |
| **Label Class 1 (`Starter`)** | 1,253 bboxes | Present across 1,008 images |
| **Starter Images Excluded** | 1,007 | Excluded per project specification (`REJECTED_EXCLUDED_CLASS_STARTER`) |
| **Resolution Audit (Class 0)** | 100% at $200 \times 200$ px | Fails minimum resolution threshold of $400 \times 400$ px |
| **Context Audit (Class 0)** | 100% isolated catalog shots | Detached components on white e-commerce backgrounds |
| **Original Annotations Staged** | **2,093** | Full preservation in [`data/raw/stage2_collection/alternator/original_annotations/`](file:///d:/AI-Vehicle-Diagnosis-System/data/raw/stage2_collection/alternator/original_annotations/) |
| **Accepted In-Situ Images** | **0** | Excluded to prevent model corruption / domain collapse |

### Curation Decision & Integrity Rationale:
Every alternator candidate in this public archive is a $200 \times 200$ catalog thumbnail of a detached component with bounding boxes spanning 95–99% of the frame. Ingesting these would train the YOLOv8 model to look for white squares with detached metal scrap rather than mounted alternators driven by serpentine belts deep inside complex engine bays.

To preserve scientific rigor:
1. All 2,093 original annotation files are preserved under `original_annotations/` for traceability.
2. All 2,093 records are documented in [`source_metadata.csv`](file:///d:/AI-Vehicle-Diagnosis-System/data/raw/stage2_collection/alternator/source_metadata.csv).
3. Zero isolated catalog images were accepted into the staging directory.
4. **Gap Remediation Strategy:** Co-labeling alternators directly on the 193 staged `engine_bay` images (where alternators are naturally installed in-situ) supplemented by direct workshop photography.
