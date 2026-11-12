# Stage 2.3: Alternator Sourcing & Workshop Collection Specification

**Project Title:** AI-Powered Multimodal Vehicle Fault Detection and Diagnosis System  
**Stage:** Stage 2.3 – Multi-Class Annotation and Workshop Data Collection  
**Target Class:** Class 2 (`alternator`)  
**Target Ingestion Volume:** 150–200 In-Situ Workshop Images + Co-occurring Dual-Labeled Open-Hood Candidates  
**Image Directory:** [`data/raw/stage2_collection/alternator/images/`](file:///d:/AI-Vehicle-Diagnosis-System/data/raw/stage2_collection/alternator/images/)  
**Original Annotations Preserved:** [`data/raw/stage2_collection/alternator/original_annotations/`](file:///d:/AI-Vehicle-Diagnosis-System/data/raw/stage2_collection/alternator/original_annotations/) (2,093 files)  
**Metadata Registries:** [`source_metadata.csv`](file:///d:/AI-Vehicle-Diagnosis-System/data/raw/stage2_collection/alternator/source_metadata.csv) (Public archive audit) | [`workshop_metadata.csv`](file:///d:/AI-Vehicle-Diagnosis-System/data/raw/stage2_collection/alternator/workshop_metadata.csv) (Workshop intake)  
**Date:** November 4, 2026  

---

## 1. Dual-Path Sourcing Strategy

Because the candidate public dataset (`Starter and Alternator.v2i.yolov8.zip`) consisted entirely of $200 \times 200$ pixel isolated product catalog thumbnails on white backgrounds, our quality audit rejected all 1,086 alternator candidates to protect the prototype model from severe domain shift and false detections.

To build a high-accuracy, real-world vehicle diagnosis detector, Class 2 is populated via a two-path strategy:

```
                          Class 2: alternator
                                   │
         ┌─────────────────────────┴─────────────────────────┐
         ▼                                                   ▼
Path 1: In-Situ Dual-Labeling                     Path 2: Workshop Photography
- 193 staged open-hood images                     - Direct workshop garage intake
- Annotate visible mounted alternators            - Target: 150–200 high-res photos
- Real engine bay spatial context                 - Diverse engine layouts & angles
```

---

## 2. Path 1: Dual-Labeling on Staged Engine Bay Images

During the Stage 2.3 Roboflow annotation session for the 193 engine bay images:
1. Every image is scrutinized for a visible mounted alternator.
2. If clearly identifiable, draw a bounding box labeled `alternator`.
3. If partially obscured by engine cover or wiring harness, include only the visible casing and pulley assembly.
4. Record visibility status in [`dual_labeling_metadata.csv`](file:///d:/AI-Vehicle-Diagnosis-System/data/raw/stage2_collection/engine_bay/dual_labeling_metadata.csv):
   - `VISIBLE_AND_ANNOTATED`: Confirmed clear alternator.
   - `ABSENT_OR_OCCLUDED`: Hidden, omitted, or mounted out of view.
   - `UNCERTAIN_NEEDS_REVIEW`: Ambiguous component requiring secondary mechanic review.

---

## 3. Path 2: Workshop Alternator Photography Protocol (Task 3)

### Target Volume & Scope
- **Target:** 150–200 usable, high-resolution in-situ photographs across operational repair garages.
- **Minimum Resolution:** $1280 \times 720$ pixels (or native smartphone camera resolution $\ge 1920 \times 1080$; min acceptable $400 \times 400$).
- **Frame Composition:** Contextual framing showing the alternator bolted to the engine block with surrounding serpentine belt and bracket assembly visible.

### Diversity Matrix:
| Dimension | Stratification Target | Notes |
|---|---|---|
| **Engine Layout** | - 60% Transverse (Front-Wheel Drive)<br>- 30% Longitudinal (Rear/All-Wheel Drive)<br>- 10% V-Engine (Transverse or Longitudinal) | Ensures model generalizes across engine orientations |
| **Viewing Angles** | - 90° Top-Down Overhead<br>- 45° Front-Quarter Oblique<br>- Low-angle / Side-profile between radiator & block | Prevents perspective overfitting |
| **Lighting Context** | - Natural daylight (open workshop bay door)<br>- Overhead fluorescent shop lighting<br>- Directional inspection LED torch / work light | Validates detector in varied maintenance environments |
| **Surface Condition** | - Clean / recently replaced OEM unit<br>- Light dust / road film<br>- Heavy oil contamination / grease buildup<br>- Surface oxidation / aluminum corrosion | Crucial for realistic fault detection scenarios |
| **Vehicle Brands** | Asian (Toyota, Hyundai, Honda), European (VW, Renault, BMW), American (Ford, Chevrolet) | Diverse alternator casing styles and belt routes |

### Explicit Inclusions:
1. Alternator body including stator housing, front drive pulley, and visible electrical connector.
2. Serpentine drive belt engaging the alternator pulley.
3. Tensioner or adjustment bracket where visible.

### Strict Exclusions:
1. **Detached Workbench Parts:** Loose scrap-yard alternators sitting on tables or floors.
2. **AC Compressors:** Larger cylindrical units with heavy refrigerant manifold lines mounted lower on the engine block.
3. **Power Steering Pumps:** Units with plastic fluid reservoirs directly integrated.
4. **Starter Motors:** Compact cylindrical motors tucked under intake manifolds engaging the transmission flywheel.
5. **Idler Pulleys:** Isolated pulleys without the slotted aluminum alternator casing.

---

## 4. Metadata Registry & Workshop Intake Workflow

Every captured workshop photograph is transferred to:
`data/raw/stage2_collection/alternator/images/`

And logged in [`data/raw/stage2_collection/alternator/workshop_metadata.csv`](file:///d:/AI-Vehicle-Diagnosis-System/data/raw/stage2_collection/alternator/workshop_metadata.csv) with the schema:

```csv
filename,original_source,source_url_or_workshop_id,license_or_permission,capture_date,vehicle_make_model,engine_layout,component_visibility,pulley_and_belt_visible,wear_and_dirt_level,annotation_status,review_status
```

### Privacy & Consent Compliance:
- Workshop owner / service manager consent must be obtained prior to photography.
- Mask or crop customer license plates and vehicle identification numbers (VIN).
- Customer personally identifiable information (PII) must not appear in any image.
