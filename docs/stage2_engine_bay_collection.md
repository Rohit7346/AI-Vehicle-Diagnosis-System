# Stage 2.2: Engine Bay Image Collection & Staging Specification

**Project Title:** AI-Powered Multimodal Vehicle Fault Detection and Diagnosis System  
**Stage:** Stage 2.2 – Missing Component Image Collection  
**Target Class:** Class 0: `engine_bay`  
**Target Ingestion Volume:** 350 images  
**Storage Destination:** `data/raw/stage2_collection/engine_bay/images/`  
**Metadata Registry:** `data/raw/stage2_collection/engine_bay/source_metadata.csv`  
**Date:** November 4, 2026  

---

## 1. Candidate Source Overview & Provenance

| Parameter | Value | Justification / Notes |
|---|---|---|
| **Source Repository** | [Roboflow Universe: Car Open Hood](https://universe.roboflow.com/vehicles-ztmmv/car-open-hood) | Public community repository |
| **Workspace Slug** | `vehicles-ztmmv` | Creator workspace: `vehicles` |
| **Dataset Title** | `car-open-hood` (v1/v2) | Automotive open-hood monitoring dataset |
| **License** | **CC BY 4.0** | Commercial and academic use permitted with attribution |
| **Total Scale** | ~1,200 images | Partitioned across `OpenHood`, `NotOpenHood`, `Unlabeled` |
| **Direct Bounding Boxes** | **None** | Source is an **image classification dataset**, not an object detector |

---

## 2. Ingestion & Filtering Rules

Because the candidate source provides whole-image labels rather than coordinate annotations, the dataset serves as an unannotated image pool for Class 0 bootstrapping:

### Inclusion Criteria:
1. **Clearly Identifiable Engine Compartment:** Image must clearly capture the open hood cavity bounded by inner fender aprons, firewall bulkhead, and front crossmember / radiator support.
2. **Vehicle Diversity:** Select across diverse body styles: sedans, compact hatchbacks, SUVs, pickup trucks, and minivans.
3. **Perspectives:** Include 90° overhead top-down angles, 45° frontal-quarter angles, and lateral oblique perspectives.
4. **Resolution:** Minimum image dimensions of $400 \times 400$ pixels (preferred $\ge 640 \times 640$).

### Exclusion Criteria (Negative Filters):
1. **`NotOpenHood` Images:** Discard all images where the hood is closed or only the exterior bodywork is visible.
2. **Engine Block Close-Ups:** Discard macro close-ups framing solely the valve cover or intake plenum where the outer perimeter of the engine bay is cut off.
3. **Severe Motion Blur or Darkness:** Discard images with Laplacian variance $< 60.0$ or mean luminance $< 25.0$.
4. **Exact Duplicates:** Filter via SHA-256; discard identical copies.

---

## 3. Metadata Registry Schema

Metadata for every selected image is tracked in [`data/raw/stage2_collection/engine_bay/source_metadata.csv`](file:///d:/AI-Vehicle-Diagnosis-System/data/raw/stage2_collection/engine_bay/source_metadata.csv) with the following header:

```csv
filename,source_dataset,original_image_identifier,source_url,license,dataset_version,selection_decision
```

- `filename`: Local filename inside `data/raw/stage2_collection/engine_bay/images/`
- `source_dataset`: `vehicles-ztmmv/car-open-hood`
- `original_image_identifier`: Upstream Roboflow filename or asset UUID
- `source_url`: `https://universe.roboflow.com/vehicles-ztmmv/car-open-hood`
- `license`: `CC BY 4.0`
- `dataset_version`: Upstream version tag (e.g. `v1`)
- `selection_decision`: `ACCEPTED_OPEN_HOOD` or `REJECTED_<REASON>`

---

## 5. Empirical Ingestion & Audit Results

The candidate archive `Car Open Hood.v1i.folder.zip` (85.8 MB) was unpacked and comprehensively evaluated via `ml/datasets/audit_staged_images.py`:

| Metric / Check | Value | Notes |
|---|:---:|---|
| **Archive Total Files** | 638 | 633 image files, folder structure |
| **`NotOpenHood` Partition** | 15 images | Exterior vehicle photos with closed hood |
| **`OpenHood` Partition** | 618 images | Interior engine compartment photographs |
| **Underlying Base Scenes** | 211 unique stems | Source dataset consists of 3x augmented variations |
| **Corrupted Images** | 0 | 100% decoded successfully |
| **Undersized Images (<400px)** | 87 | Excluded (e.g. 320x240, 500x375 thumbnails) |
| **Blurry Images ($\sigma^2 < 60.0$)** | 3 | Excluded (Laplacian variance 54.6–54.9) |
| **Extreme Luminance** | 0 | All valid images within $[25.0, 240.0]$ |
| **Near-Duplicate Rejections** | 350 | Redundant augmented variations pruned via dHash ($D_H \le 4$) |
| **Accepted High-Quality Base Images** | **193** | Sharpest unique view per vehicle scene staged |

### Current Staging Status:
- **Staged Images:** 193 files in [`data/raw/stage2_collection/engine_bay/images/`](file:///d:/AI-Vehicle-Diagnosis-System/data/raw/stage2_collection/engine_bay/images/)
- **Audit Compliance:** 100% pass rate across 0 corrupted, 0 undersized, 0 blurry, 0 duplicate images.
- **Audit Registry:** 648 records tracked in [`data/raw/stage2_collection/engine_bay/source_metadata.csv`](file:///d:/AI-Vehicle-Diagnosis-System/data/raw/stage2_collection/engine_bay/source_metadata.csv).
- **Target Ingestion Gap:** Sourced 193 of target 350 images. Sourcing remaining ~157 distinct engine bay perspectives will occur via additional workshop photography or high-resolution open datasets.
