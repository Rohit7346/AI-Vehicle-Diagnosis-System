# Exhaust Manifold Collection & Dataset Split Plan

**Project Title:** AI-Powered Multimodal Vehicle Fault Detection and Diagnosis System  
**Stage:** Stage 2.1 – Manual Image Collection & Split Allocation Strategy  
**Target Class:** Class 4: `exhaust_manifold`  
**Target Collection Volume:** 250 in-situ vehicle photographs  
**Date:** November 4, 2026  

---

## 1. Collection Objectives & Semantic Definition

Because empirical research confirmed that no public object-detection dataset provides reliable *in-situ* automotive exhaust manifold annotations, a structured manual collection plan is established.

### Strict Semantic Scope:
- **`exhaust_manifold`:** Cast iron or tubular stainless-steel exhaust manifold/header bolted directly to the cylinder head exhaust ports, collecting burnt gases into the exhaust downpipe. Includes factory-stamped aluminium heat shields bolted directly over the manifold.

### Explicit Exclusion Criteria (Negative Distractors):
- **Mufflers & Silencers:** Intermediate resonators or rear silencer boxes (located under the floorpan or rear bumper).
- **Tailpipes & Tips:** Chrome or steel exhaust tips exiting the rear bumper.
- **Motorcycle Exhausts:** Exposed two-wheeler chrome headers or external exhaust canisters.
- **Standalone Catalytic Converters:** Under-chassis catalytic converters that are separated from the cylinder head by downpipes.

---

## 2. Image Diversity Matrix (250 Target Images)

To ensure the trained YOLOv8 Nano model generalizes effectively across commercial vehicles entering diagnostic service bays, the 250 images must be collected across predefined diversity quotas:

### A. Powertrain & Engine Layout Distribution
| Engine Configuration | Manifold Position & Characteristics | Target Image Count |
|---|---|:---:|
| **Transverse FWD (Front Manifold)** | Manifold mounted at front of engine, directly facing radiator/grille. Highly visible (e.g., Hyundai Kappa, Maruti K12). | **90 images** |
| **Transverse FWD (Rear Manifold)** | Manifold mounted at rear of engine between block and firewall. Often shadowed or partially occluded (e.g., Honda R-series, Toyota 2NZ). | **70 images** |
| **Longitudinal RWD (Inline)** | Side-mounted manifold along engine block (e.g., Mahindra mHawk, Tata 2.2 Dicor). | **50 images** |
| **V-Engine (V6 / V8)** | Transverse or longitudinal bank 1 or bank 2 visible manifolds. | **40 images** |

### B. Shielding & Surface Condition Distribution
| Surface / Shielding Condition | Visual Attributes | Target Image Count |
|---|---|:---:|
| **Factory Heat Shield Installed** | Stamped textured aluminium or galvanized steel heat shield with embossed warning icons. | **110 images** |
| **Exposed Cast Iron** | Dark grey, charcoal, or surface-oxidized cast iron manifold runners. | **80 images** |
| **Tubular Stainless / Aftermarket** | Welded tubular steel runners with heat-tinted straw/blue discoloration. | **30 images** |
| **Grimed / Oil-Soiled / Corroded** | Real-world workshop condition with road dust, salt corrosion, or mild valve-cover oil residue. | **30 images** |

### C. Perspective & Lighting Protocols
- **Angles:** 40% top-down overhead (90°), 40% front-quarter oblique (45°), 20% tight macro angle into the firewall gap.
- **Lighting:**
  - 100 images: Workshop fluorescent indoor illumination.
  - 80 images: Bright ambient natural outdoor daylight.
  - 40 images: High-contrast direct sun creating deep engine bay shadows.
  - 30 images: Low-light under-hood photography utilizing smartphone / inspection lamp LED flash.

---

## 3. Roboflow Batch Ingestion & Annotation Protocol

1. **Batch Preparation:** Images must be captured at $\ge 1280 \times 720$ resolution and grouped into 5 batches of 50 images (`exhaust_manifold_batch_01` to `batch_05`).
2. **Pre-Upload Review:** Review raw photos against the exclusion criteria. Discard motion-blurred or unidentifiable photos.
3. **Bounding Box Execution (per [`docs/annotation_guidelines.md`](file:///d:/AI-Vehicle-Diagnosis-System/docs/annotation_guidelines.md)):**
   - Class ID: Select `4: exhaust_manifold`.
   - Box Extent: Enclose the exhaust runners and factory heat shield. Tightly border the cylinder head mounting flange down to the downpipe collector flange.
   - Occlusion Rule: If $\ge 20\%$ of the manifold is visible through wiring harnesses, draw the box. If $< 20\%$ is visible, leave unannotated.

---

## 4. Dataset Split Allocation Strategy

### Core Architectural Safeguards:
1. **Preserve Existing Test Set:** The current 61 test images in `data/prototype_5class/images/test/` must remain **completely untouched**. We do not shuffle existing test data.
2. **Grouped Allocation by Vehicle / Session:** If multiple photos are taken of the same vehicle (e.g., from different angles or lighting), **all photos of that specific vehicle must be assigned to the same split**. Never place Angle A of Vehicle #12 in `train` and Angle B in `val` (prevents near-duplicate leakage).

### New Image Allocation Formulation:
The 250 newly curated `exhaust_manifold` images will be partitioned following the standard **80% Train / 15% Val / 5% Test** ratio:

| Partition | Allocation Ratio | Allocated Image Count | Target Use Case |
|---|:---:|:---:|---|
| **`train`** | 80% | **200 images** | Feature extraction and weight optimization in YOLOv8 Nano. |
| **`val`** | 15% | **38 images** | Hyperparameter tuning, early stopping, and mAP evaluation. |
| **`test`** | 5% | **12 images** | Held-out benchmark evaluation on completely unseen vehicles. |
| **TOTAL** | **100%** | **250 images** | — |

---

## 5. Image Counts vs. Annotation Counts

In object detection datasets, **image count** and **annotation count** are distinct metrics:
- An image containing an open engine bay may contain multiple component labels simultaneously.
- When annotating the 250 new images, annotators will also label any co-occurring prototype components visible in the same frame (e.g., `0: engine_bay`, `1: battery`, `3: radiator`):

```
Single In-Situ Image Example:
  - 1 x Image File (exhaust_sample_01.jpg)
  - 3 x Annotations:
      [0] 0.500 0.500 0.950 0.900  -> engine_bay
      [4] 0.350 0.450 0.280 0.220  -> exhaust_manifold
      [1] 0.780 0.320 0.160 0.190  -> battery
```

This multi-label density enriches contextual co-occurrence learning for YOLOv8 without requiring additional raw image volume.
