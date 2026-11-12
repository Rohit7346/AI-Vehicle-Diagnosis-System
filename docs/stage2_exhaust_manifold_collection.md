# Stage 2.2: Exhaust Manifold Workshop Image Collection Protocol

**Project Title:** AI-Powered Multimodal Vehicle Fault Detection and Diagnosis System  
**Stage:** Stage 2.2 – Missing Component Image Collection  
**Target Class:** Class 4: `exhaust_manifold`  
**Target Ingestion Volume:** 250 in-situ vehicle images  
**Storage Destination:** `data/raw/stage2_collection/exhaust_manifold/images/`  
**Metadata Registry:** `data/raw/stage2_collection/exhaust_manifold/source_metadata.csv`  
**Date:** November 4, 2026  

---

## 1. Domain Context & Collection Objectives

Due to the complete absence of open-source in-situ automotive exhaust manifold datasets, a targeted manual collection is required. Exhaust manifolds occupy a critical diagnostic role in vehicle fault detection (exhaust leaks, cracked runners, failed gaskets, missing studs, glowing/overheating headers).

### Strict Semantic Boundary:
- **`exhaust_manifold`:** Cast iron or tubular steel manifold assembly bolted directly to the cylinder head exhaust ports. Factory heat shields bolted directly over the manifold are included in the definition.

### Negative Distractor Exclusions:
- **Mufflers & Silencers:** Completely rejected.
- **Tailpipes & Exhaust Tips:** Completely rejected.
- **Motorcycle Exhausts:** Completely rejected.
- **Underbody Catalytic Converters:** Rejected unless an integrated manifold-converter unit mounted directly to the cylinder head.

---

## 2. Structured Workshop Diversity Quota (250 Images)

To ensure the model learns robust visual features across varying workshop intake scenarios, images must be collected to meet predefined diversity buckets:

```
Total Target: 250 In-Situ Images
├── Engine Layouts:
│   ├── Transverse Front-Facing Manifold: 90 images (36%)
│   ├── Transverse Rear-Facing Manifold:  70 images (28%)
│   ├── Longitudinal Inline-4 / Inline-6:  50 images (20%)
│   └── V-Engine Bank 1 / Bank 2:          40 images (16%)
├── Surface & Shielding States:
│   ├── Factory Stamped Aluminium Shield: 110 images (44%)
│   ├── Exposed Oxidized / Cast Iron:      80 images (32%)
│   ├── Tubular Stainless / Aftermarket:   30 images (12%)
│   └── Oil-Grimed / Corroded Surface:     30 images (12%)
└── Illumination Environments:
    ├── Fluorescent Workshop Bay:         100 images (40%)
    ├── Outdoor Ambient Daylight:          80 images (32%)
    ├── High-Contrast Direct Sunlight:     40 images (16%)
    └── Mobile Inspection LED Flash:       30 images (12%)
```

---

## 3. Legal Provenance & Intellectual Property Compliance

In strict compliance with university academic and research ethics:
1. **Permitted Sources:**
   - Physical vehicle intake photography in university automotive laboratories, authorized service centers, or local multi-brand repair workshops.
   - Creative Commons (CC0 / CC BY) automotive photography repositories with explicit commercial and academic reuse rights.
2. **Explicit Prohibition on Scraping:**
   - Automated scraping of copyrighted dealership inventories, forum galleries, or auction portals in violation of their Terms of Service is strictly prohibited.
3. **Privacy Masking:**
   - Vehicle registration number plates and customer vehicle identification numbers (VIN) visible in peripheral framing must be blurred or cropped out before ingestion.

---

## 4. Metadata Registry Schema

Every collected image is registered in [`data/raw/stage2_collection/exhaust_manifold/source_metadata.csv`](file:///d:/AI-Vehicle-Diagnosis-System/data/raw/stage2_collection/exhaust_manifold/source_metadata.csv) with the following header:

```csv
filename,source,permission_license,capture_date,selection_decision,engine_layout,shielding_state
```

- `filename`: Local filename inside `data/raw/stage2_collection/exhaust_manifold/images/`
- `source`: e.g. `Auto_Lab_Intake_Vehicle_04` or `Open_Access_Repo`
- `permission_license`: `Workshop_Permission_Granted` or `CC_BY_4.0`
- `capture_date`: ISO format date (`YYYY-MM-DD`)
- `selection_decision`: `ACCEPTED_IN_SITU` or `REJECTED_<REASON>`
- `engine_layout`: `Transverse_Front`, `Transverse_Rear`, `Longitudinal`, `V_Engine`
- `shielding_state`: `Aluminium_Heat_Shield`, `Exposed_Cast_Iron`, `Tubular_Header`

---

## 5. Pre-Annotation Quality Checklist

Prior to importing collected images into Roboflow:
1. Run [`ml/datasets/audit_staged_images.py`](file:///d:/AI-Vehicle-Diagnosis-System/ml/datasets/audit_staged_images.py) to confirm zero file corruption, resolution $\ge 400 \times 400$, and zero exact/near duplicate images.
2. Verify that at least 20% of the exhaust manifold assembly is unambiguously identifiable through wiring harnesses and plumbing.
3. Follow [`docs/annotation_guidelines.md`](file:///d:/AI-Vehicle-Diagnosis-System/docs/annotation_guidelines.md) for bounding box tightness.
