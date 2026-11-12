# Stage 2.3: Exhaust Manifold Workshop Collection Specification

**Project Title:** AI-Powered Multimodal Vehicle Fault Detection and Diagnosis System  
**Stage:** Stage 2.3 – Multi-Class Annotation and Workshop Data Collection  
**Target Class:** Class 4 (`exhaust_manifold`)  
**Target Collection Volume:** 250 Usable In-Situ Workshop Photographs  
**Aligned Strategy:** [`docs/exhaust_manifold_collection_plan.md`](file:///d:/AI-Vehicle-Diagnosis-System/docs/exhaust_manifold_collection_plan.md)  
**Image Directory:** [`data/raw/stage2_collection/exhaust_manifold/images/`](file:///d:/AI-Vehicle-Diagnosis-System/data/raw/stage2_collection/exhaust_manifold/images/)  
**Metadata Registry:** [`source_metadata.csv`](file:///d:/AI-Vehicle-Diagnosis-System/data/raw/stage2_collection/exhaust_manifold/source_metadata.csv)  
**Date:** November 4, 2026  

---

## 1. Scope & Technical Context

Exhaust manifolds represent one of the most critical thermal and structural components in vehicle powertrain diagnosis. They operate under extreme thermal cycling (ambient to $850^\circ\text{C}$), making them prone to warping, cracking, stud shearing, and exhaust gas leakage.

Because no high-quality public dataset provides granular, in-situ vehicle exhaust manifold bounding boxes, Class 4 is sourced directly via controlled workshop photography across authorized repair workshops, independent garages, and automotive inspection bays.

---

## 2. Exhaust Manifold Structural Taxonomy

During image intake, photographers must catalog components according to the four primary automotive manifold architectures:

| Manifold Architecture | Physical Characteristics | Diagnostic Relevance |
|---|---|---|
| **1. Cast-Iron Log Manifold** | Heavy-gauge, dark gray/brown textured iron casting collecting exhaust from 3–4 cylinder ports into a single collector flange. | Prone to stress fractures, hair-line cracks, and gasket leaks. |
| **2. Tubular Stainless Header** | Welded individual mandrel-bent stainless-steel or mild-steel primary tubes converging into a merged collector. | Prone to weld seam failure and thermal discoloration. |
| **3. Integrated Cylinder Head** | Manifold runners cast internally inside the aluminum cylinder head; only the single exhaust outlet port and turbo flange are external. | Modern downsized engines; diagnostic focus is on turbo flange studs and heat shielding. |
| **4. Heat-Shielded Manifold** | Stamped galvanized steel or embossed aluminum thermal shield enclosing the runners to protect the engine bay from radiant heat. | Most common real-world configuration (~65% of production vehicles); detector must learn both the shield perimeter and visible runners. |

---

## 3. Workshop Capture Diversity Matrix (Target: 250 Images)

To ensure robust detection across diverse engine compartments, photographs must be distributed according to this composition matrix:

```
                      250 Exhaust Manifold Target
                                   │
      ┌────────────────┬───────────┴───────────┬────────────────┐
      ▼                ▼                       ▼                ▼
Transverse Front  Transverse Rear         Longitudinal      V-Engine Banks
  (75 images)       (50 images)           (50 images)        (45 images)
```

| Engine Configuration | Target Count | Recommended Camera Angles | Optical Challenges |
|---|:---:|---|---|
| **Transverse Front-Facing** | 75 | 45° downward frontal, low-angle through radiator gap | Direct access; watch for glare off heat shield |
| **Transverse Rear-Facing** | 50 | High overhead reaching over engine cover toward firewall | Low ambient light; require LED inspection torch |
| **Longitudinal Engine (I4 / I6)** | 50 | Lateral oblique 45° framing cylinder head flank | Clear lateral view along inner fender |
| **V-Engine (V6 / V8 Banks)** | 45 | Bank 1 (right) and Bank 2 (left) individual frames | High engine cover obstruction |
| **Condition Diversity** | 30 | Oily residue, severe iron rust, cracked shield/casting | Critical for multimodal diagnosis |

---

## 4. Strict Quality & Exclusion Protocol

### Strict Inclusion Criteria:
1. Manifold or primary heat shield must be physically bolted to the cylinder head exhaust ports.
2. Image resolution must be $\ge 1280 \times 720$ pixels (native phone/camera resolution $\ge 1920 \times 1080$; min acceptable $400 \times 400$).
3. Image must be sharp ($\sigma^2 \ge 60.0$) with adequate exposure (avoid pitch-black shadows behind engine block).

### Explicit Exclusions:
1. **Underbody Exhaust Components:** Discard rear mufflers, intermediate resonators, and tailpipes.
2. **Motorcycle / Marine Exhausts:** Strictly automotive passenger cars, light trucks, and SUVs.
3. **Detached Downpipes / Catalytic Converters:** Discard isolated catalytic converters unless attached directly to the manifold collector.
4. **Scrap-Yard Loose Parts:** Discard detached manifolds resting on workbenches or scrap heaps.

---

## 5. Metadata Schema & Intake Registry

All staged images are transferred to `data/raw/stage2_collection/exhaust_manifold/images/` and cataloged in [`data/raw/stage2_collection/exhaust_manifold/source_metadata.csv`](file:///d:/AI-Vehicle-Diagnosis-System/data/raw/stage2_collection/exhaust_manifold/source_metadata.csv) with the following schema:

```csv
filename,original_source,source_url_or_workshop_id,license_or_permission,capture_date,vehicle_make_model,engine_layout,shielding_state,manifold_material,component_visibility,annotation_status,review_status
```

### Privacy & Vehicle Masking:
- Vehicle registration plates and VIN barcodes must be cropped out or digitally masked before annotation.
- Workshop personnel, customers, or proprietary service records must not appear in the photograph.
- Signed workshop intake consent must be logged in project records.
