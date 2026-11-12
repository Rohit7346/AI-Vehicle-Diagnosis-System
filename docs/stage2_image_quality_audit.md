# Stage 2.2: Image Quality & Duplicate Audit Specification

**Project Title:** AI-Powered Multimodal Vehicle Fault Detection and Diagnosis System  
**Stage:** Stage 2.2 – Missing Component Image Collection & Quality Auditing  
**Date:** November 4, 2026  
**Auditor Engine:** `ml/datasets/audit_staged_images.py`  

---

## 1. Audit Framework & Technical Methodology

To ensure only high-quality, representative, and non-redundant photographs enter the training pipeline, an automated quality assurance engine ([`ml/datasets/audit_staged_images.py`](file:///d:/AI-Vehicle-Diagnosis-System/ml/datasets/audit_staged_images.py)) is implemented. Every candidate image staged in `data/raw/stage2_collection/` undergoes six sequential audit checks:

### 1. File Integrity & Decoding Validation
- **Method:** Strict stream decode via `Image.open().load()`.
- **Threshold:** Zero tolerance. Corrupted bitstreams, truncated JPEGs, or unreadable files are immediately flagged and rejected.

### 2. Resolution & Dimension Boundary
- **Method:** Pixel dimension validation $(W, H)$.
- **Threshold:** Minimum resolution of $400 \times 400$ pixels. Images below this threshold lack fine-grained texture necessary for YOLOv8 edge feature maps.

### 3. Cryptographic Exact Duplicate Detection
- **Method:** 256-bit SHA-256 cryptographic hashing of raw file bytes.
- **Threshold:** Zero exact duplicate tolerance. Identical hashes are grouped; only the primary copy is retained, while redundant copies are rejected.

### 4. Perceptual Near-Duplicate Detection (dHash)
- **Method:** 64-bit Difference Hash (dHash) computed by resizing images to $9 \times 8$ grayscale and calculating directional horizontal gradients:
  $$\text{bit}_{r, c} = (P_{r, c} > P_{r, c+1})$$
- **Similarity Metric:** Bitwise Hamming distance between 64-bit hash integers:
  $$D_H(h_1, h_2) = \sum (h_1 \oplus h_2)$$
- **Threshold:** Pairs with $D_H \le 4$ are flagged as near-duplicates (representing minor crops, aggressive JPEG recompression, or burst-mode camera shots of the same angle).

### 5. Blurriness Detection via Laplacian Variance
- **Method:** Second-order finite difference approximation of the 2D Laplacian operator $\nabla^2 I$ on grayscale image arrays:
  $$\nabla^2 I(x, y) \approx -4 I(x, y) + I(x-1, y) + I(x+1, y) + I(x, y-1) + I(x, y+1)$$
- **Metric:** Variance of the Laplacian response $\sigma^2(\nabla^2 I)$.
- **Threshold:** Images with $\sigma^2 < 60.0$ are rejected as motion-blurred or out-of-focus.

### 6. Exposure & Luminance Extremes Audit
- **Method:** Mean pixel intensity across the grayscale luminance spectrum $[0.0, 255.0]$.
- **Threshold:**
  - $\mu < 25.0$: Rejected as underexposed / pitch-black engine bay cavity.
  - $\mu > 240.0$: Rejected as overexposed / washed-out flash glare.

---

## 2. Class-by-Class Staging Status & Collection Audit

```
Stage 2.2 Staging Pipeline Status (Empirically Verified):
├── Class 0: engine_bay
│   ├── Target Volume:    350 images
│   ├── Archive Evaluated:Car Open Hood.v1i.folder.zip (633 images)
│   ├── Staging Folder:   data/raw/stage2_collection/engine_bay/images/
│   ├── Accepted Staged:  193 unique high-quality open-hood images
│   ├── Rejected Count:   440 images (15 closed-hood, 87 undersized, 3 blurry, 350 near-duplicates)
│   ├── Metadata File:    data/raw/stage2_collection/engine_bay/source_metadata.csv (648 records)
│   └── Auditor Result:   100% PASSED (0 corrupted, 0 undersized, 0 blurry, 0 duplicate)
├── Class 2: alternator
│   ├── Target Volume:    350 images (in-situ mounted)
│   ├── Archive Evaluated:Starter and Alternator.v2i.yolov8.zip (2,093 images & labels)
│   ├── Staging Folder:   data/raw/stage2_collection/alternator/images/
│   ├── Raw Annotations:  data/raw/stage2_collection/alternator/original_annotations/ (2,093 files)
│   ├── Accepted Staged:  0 images (100% fail in-situ and 400x400 minimum resolution criteria)
│   ├── Rejected Count:   2,093 images (1,007 starters, 1,086 undersized 200x200 detached catalog photos)
│   ├── Metadata File:    data/raw/stage2_collection/alternator/source_metadata.csv (2,093 records)
│   └── Integrity Action: Excluded detached thumbnails to prevent model training corruption
└── Class 4: exhaust_manifold
    ├── Target Volume:    250 images (workshop in-situ)
    ├── Staging Folder:   data/raw/stage2_collection/exhaust_manifold/images/
    ├── Metadata File:    data/raw/stage2_collection/exhaust_manifold/source_metadata.csv
    └── Sourcing Status:  Intake schema ready; awaiting direct workshop photography
```

---

## 3. Rejection Taxonomy & Quality Control Protocol

When candidate image batches are unpacked into their respective staging folders, the auditor classifies any rejected image into one of five explicit rejection categories:

| Rejection Category | Detection Mechanism | Action Taken |
|---|---|---|
| `REJECTED_CORRUPTED` | PIL `UnidentifiedImageError` or EOF error | File deleted from staging; logged in metadata. |
| `REJECTED_UNDERSIZED` | Image dimensions $< 400 \times 400$ | File excluded from annotation import. |
| `REJECTED_BLURRY` | Laplacian variance $\sigma^2 < 60.0$ | File excluded; requires sharper camera capture. |
| `REJECTED_EXPOSURE` | Mean luminance $< 25$ or $> 240$ | File excluded; lighting unsuitable for computer vision. |
| `REJECTED_DUPLICATE` | SHA-256 match or dHash $D_H \le 4$ | Redundant copy discarded; primary image retained. |
| `REJECTED_NON_IN_SITU` | Visual/metadata audit (e.g. bench-top alternator) | Isolated object rejected to avoid domain shift. |

---

## 4. Roboflow Annotation Preparation Checklist

Once candidate images pass the automated quality audit:
1. **Directory Isolation:** Ensure raw candidate collections remain inside `data/raw/stage2_collection/` and are **never** commingled with `data/prototype_5class/`.
2. **Roboflow Project Import:**
   - Upload accepted `engine_bay` images in batches of 100.
   - Upload accepted in-situ `alternator` images in batches of 100.
   - Upload accepted `exhaust_manifold` images in batches of 50.
3. **Five-Class Sequential Zero-Indexed Ontology:**
   - `0: engine_bay`
   - `1: battery`
   - `2: alternator`
   - `3: radiator`
   - `4: exhaust_manifold`
4. **Partition Discipline:** Maintain the locked 80% train / 15% val / 5% test split ratio without altering the locked 61 test samples from Stage 1.
