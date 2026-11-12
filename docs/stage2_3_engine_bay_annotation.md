# Stage 2.3: Engine Bay Image Import & Dual-Labeling Protocol

**Project Title:** AI-Powered Multimodal Vehicle Fault Detection and Diagnosis System  
**Stage:** Stage 2.3 – Multi-Class Annotation and Workshop Data Collection  
**Component Focus:** Class 0 (`engine_bay`) and Co-occurring Class 2 (`alternator`)  
**Target Ingestion Volume:** 193 Verified Images  
**Source Dataset:** [Roboflow Universe: Car Open Hood (`vehicles-ztmmv/car-open-hood`)](https://universe.roboflow.com/vehicles-ztmmv/car-open-hood) (License: CC BY 4.0)  
**Image Directory:** [`data/raw/stage2_collection/engine_bay/images/`](file:///d:/AI-Vehicle-Diagnosis-System/data/raw/stage2_collection/engine_bay/images/)  
**Upload ZIP Archive:** [`data/raw/stage2_collection/engine_bay/engine_bay_193_images.zip`](file:///d:/AI-Vehicle-Diagnosis-System/data/raw/stage2_collection/engine_bay/engine_bay_193_images.zip) (27.73 MB)  
**Manifest Files:** [`manifest.json`](file:///d:/AI-Vehicle-Diagnosis-System/data/raw/stage2_collection/engine_bay/manifest.json) | [`manifest.csv`](file:///d:/AI-Vehicle-Diagnosis-System/data/raw/stage2_collection/engine_bay/manifest.csv)  
**Tracking Metadata:** [`dual_labeling_metadata.csv`](file:///d:/AI-Vehicle-Diagnosis-System/data/raw/stage2_collection/engine_bay/dual_labeling_metadata.csv)  
**Date:** November 4, 2026  

---

## 1. Image Verification & Manifest Summary

All 193 staged images have been decoded, checked for corruption, verified for resolution ($W, H \ge 400$ px), and audited for uniqueness (zero SHA-256 exact duplicates, zero perceptual near-duplicates with $D_H \le 4$):

| Quality Metric | Ingestion Standard | Audit Result | Status |
|---|---|:---:|:---:|
| **Verified Image Count** | Exactly 193 images | 193 files | **VERIFIED** |
| **Image Readability** | 100% valid bitstream decode | 0 corrupted | **PASSED** |
| **Minimum Resolution** | Width $\ge 400$ px, Height $\ge 400$ px | Min: $605 \times 403$ px<br>Max: $4032 \times 3024$ px | **PASSED** |
| **Sharpness Boundary** | Laplacian variance $\sigma^2 \ge 60.0$ | Range: $118.4$ to $4661.4$<br>Mean: $1048.2$ | **PASSED** |
| **Exposure / Luminance** | Mean grayscale intensity $\in [25.0, 240.0]$ | Range: $62.1$ to $185.3$ | **PASSED** |
| **Exact Cryptographic Duplicates** | Zero SHA-256 collisions | 0 collisions | **PASSED** |
| **Perceptual Near-Duplicates** | 64-bit dHash Hamming distance $> 4$ | 0 pairs with $D_H \le 4$ | **PASSED** |

### Vehicle Make Diversity in Staged Set
Filename inspection confirms broad automotive brand diversity across the 193 images:
- **Identified Manufacturers:** Chevrolet (17), Toyota (15), Renault (13), Ford (10), Fiat (10), Honda (7), Volkswagen/VW (12), BYD (5), Hyundai (4), GM (3), Mercedes-Benz (3), Peugeot (1), Audi (1), BMW (1), Jeep (1).
- **Generic / Unspecified Multi-Vehicle Sets:** 90 images.

---

## 2. Roboflow Project Import Instructions

Because automated browser injection and headless scraping are restricted by Roboflow's Cloudflare challenge security layer and require user credentials, import the images manually using the prepared batch package:

### Import Steps:
1. Open your web browser and log in to [Roboflow](https://roboflow.com/).
2. Navigate to your existing project:
   **`Vehicle Component Detection`**  
   *(Do NOT create a new project; maintain the unified five-class workspace).*
3. In the left-hand navigation sidebar, click on **Upload Data**.
4. You may upload the images using either of two prepared options:
   - **Option A (Single Archive Drag-and-Drop — Recommended):** Drag and drop the packaged ZIP file:  
     `D:\AI-Vehicle-Diagnosis-System\data\raw\stage2_collection\engine_bay\engine_bay_193_images.zip`
   - **Option B (Direct Folder Selection):** Click **Select Folder** and choose:  
     `D:\AI-Vehicle-Diagnosis-System\data\raw\stage2_collection\engine_bay\images\`
5. In the Roboflow upload dialog:
   - Confirm all **193 images** are recognized.
   - Do **not** apply automated train/val/test splits during upload. Assign them to **Unannotated** / **Assign for Labeling**.
6. Click **Save and Continue** to stage the images in your Roboflow annotation queue.

---

## 3. Dual-Labeling Annotation Protocol

Each image in the 193-image pool must undergo systematic dual-component inspection:

```
                      193 Open-Hood Images
                               │
               ┌───────────────┴───────────────┐
               ▼                               ▼
       A. Class 0: engine_bay         B. Class 2: alternator
     (Mandatory: 1 bbox / image)      (Selective: 0 or 1 bbox)
```

### A. Class 0: `engine_bay` Bounding Box Rules
1. **Perimeter Enclosure:** The bounding box must encapsulate the macroscopic under-hood mechanical volume.
2. **Inner Boundaries:** Align the box with the inner fender aprons (left and right), the firewall bulkhead (rear/top), and the radiator core support crossmember (front/bottom).
3. **Strict Negative Exclusions:**
   - **DO NOT** include the exterior hood/bonnet skin or outer painted body panels.
   - **DO NOT** include the front grille, headlights, front bumper cover, or ground/workshop floor.
   - **DO NOT** label solely the engine block / cylinder head cover as the `engine_bay`. The class represents the complete compartment cavity.
4. **Cardinality:** Exactly **one** `engine_bay` bounding box per valid open-hood photograph.

### B. Class 2: `alternator` Bounding Box Rules
1. **Verification of Presence:** Inspect the front accessory drive layout of the engine. An alternator is present in the foreground on many vehicles, but may be deeply recessed, obscured by intake piping, or located below the frame.
2. **Identification Criteria:**
   - Cylindrical cast-aluminum stator housing with ventilation slots/vanes.
   - Front drive pulley engaged with the serpentine accessory belt.
   - Distinct mounting ears bolted to the engine block or bracket.
3. **Anti-Confusion Guidelines:**
   - **AC Compressor:** Mounted lower down on the engine block, larger diameter, identifiable by refrigerant lines/manifolds. **Do NOT** label as an alternator.
   - **Power Steering Pump:** Often higher up with a plastic fluid reservoir directly attached. **Do NOT** label as an alternator.
   - **Idler / Tensioner Pulleys:** Smooth or ribbed standalone pulleys without an electrical generator body. **Do NOT** label as an alternator.
4. **Selective Labeling Rule:**
   - If the alternator is clearly visible: Draw a tight bounding box around its casing and pulley.
   - If the alternator is obscured, hidden beneath engine covers, or identity is uncertain: **DO NOT CREATE A LABEL.**
   - If ambiguous: Flag the image for human review in `dual_labeling_metadata.csv`.

---

## 4. Dual-Labeling Metadata Registry

All 193 images are initialized in [`data/raw/stage2_collection/engine_bay/dual_labeling_metadata.csv`](file:///d:/AI-Vehicle-Diagnosis-System/data/raw/stage2_collection/engine_bay/dual_labeling_metadata.csv) with the schema:

```csv
filename,original_source,source_url_or_workshop_id,license_or_permission,capture_date,vehicle_make_model,component_visibility,engine_bay_status,alternator_status,annotation_status,review_status
```

During human annotation, the annotator updates:
- `engine_bay_status`: `ANNOTATED_VALID`
- `alternator_status`: `VISIBLE_AND_ANNOTATED`, `ABSENT_OR_OCCLUDED`, or `UNCERTAIN_NEEDS_REVIEW`
- `annotation_status`: `ANNOTATED`
- `review_status`: `APPROVED` or `FLAGGED_FOR_AUDIT`
