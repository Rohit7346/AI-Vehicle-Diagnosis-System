# Annotation Guidelines: Vehicle Component Detection Prototype

**Project:** AI-Powered Multimodal Vehicle Fault Detection and Diagnosis System  
**Stage:** Stage 2 – Annotation Standards & Quality Control  
**Target Format:** Ultralytics YOLOv8 Bounding Boxes (`<class_id> <x_center> <y_center> <width> <height>`)  
**Classes:** 5-Component Prototype Hierarchy  

---

## 1. General Bounding Box Principles

To train a robust YOLOv8 Nano object detector, annotations must adhere to strict geometric and semantic rules:

1. **Tight Boundaries:** Bounding boxes must tightly border the outermost visible pixels of the component. Avoid loose boxes containing excessive background clutter.
2. **No Empty Margins:** Do not pad the bounding box to make it square. The box should match the natural aspect ratio of the component.
3. **Occlusion Handling:**
   - **Partial Occlusion ($\ge 20\%$ visible):** If at least 20% of the component is visible and identifiable, draw the bounding box to enclose the entire identifiable extent of the component (including the occluded section only if its contour is unambiguously inferrable).
   - **Severe Occlusion ($< 20\%$ visible):** If less than 20% of the component is visible, **DO NOT ANNOTATE**. Annotating heavily obscured slivers creates false positive hallucinations during training.
4. **Truncation at Image Edges:** If a component is cut off by the edge of the photograph, bound only the portion visible within the frame. Do not project coordinates outside the image boundary $[0.0, 1.0]$.
5. **No Synthetic Extrapolations:** Only annotate physical hardware visible in the photograph. Do not label components that you "know are there" but cannot see.

---

## 2. Component-by-Component Annotation Specifications

### Class 0: `engine_bay`

| Attribute | Specification |
|---|---|
| **Class ID** | `0` |
| **Semantic Meaning** | The complete vehicle engine compartment perimeter enclosure under an opened hood. |
| **Inclusions (Inside the Box)** | The entire cavity bounded by the inner fender walls, firewall/bulkhead, and front crossmember / radiator support. |
| **Exclusions (Outside the Box)** | The exterior front bumper, exterior front fenders, hood exterior panel, windshield, and ground underneath the car. |
| **Critical Anti-Pattern** | **DO NOT** annotate the engine block or plastic engine decorative cover alone. `engine_bay` represents the macro enclosure context. |
| **Acceptable Image Types** | Wide-angle under-hood photographs capturing the entire open compartment. |
| **Unsuitable Image Types** | Extreme macro close-ups where the perimeter walls of the engine compartment are completely out of frame. |

---

### Class 1: `battery`

| Attribute | Specification |
|---|---|
| **Class ID** | `1` |
| **Semantic Meaning** | 12V automotive starter battery unit. |
| **Inclusions (Inside the Box)** | Rectangular plastic battery casing, top cell caps, positive (+) and negative (-) terminal posts, cable terminal clamps, and metal hold-down crossbar bracket. |
| **Exclusions (Outside the Box)** | Standalone fuse/relay boxes mounted adjacent to the battery tray, ECU enclosures, or long wire runs extending beyond the battery base. |
| **Critical Anti-Pattern** | Do not label a fuse box lid as a battery just because it is black and located next to the battery. |
| **Acceptable Image Types** | Clear view of battery top casing and terminals from top-down or 45° angle. |
| **Unsuitable Image Types** | Battery hidden completely beneath an opaque plastic aftermarket cover where no terminal or casing is visible. |

---

### Class 2: `alternator`

| Attribute | Specification |
|---|---|
| **Class ID** | `2` |
| **Semantic Meaning** | Serpentine-belt-driven AC electrical generator mounted to the engine block. |
| **Inclusions (Inside the Box)** | Cylindrical cast aluminium housing, stator ventilation slots, front drive pulley, serpentine belt loop over the pulley, and visible electrical output connector. |
| **Exclusions (Outside the Box)** | Air conditioning (AC) compressor (typically mounted lower), power steering pump, water pump, or belt tensioner idler pulleys. |
| **Critical Anti-Pattern** | Do not confuse the alternator with the AC compressor. The alternator features visible stator wire coils through ventilation slots and a smaller pulley. |
| **Acceptable Image Types** | Front or side engine view showing the belt-driven accessory drive with alternator clearly identifiable. |
| **Unsuitable Image Types** | Deep underbody shots where only a belt is visible, or disconnected scrap parts photographed without engine context. |

---

### Class 3: `radiator`

| Attribute | Specification |
|---|---|
| **Class ID** | `3` |
| **Semantic Meaning** | Front engine cooling radiator assembly. |
| **Inclusions (Inside the Box)** | Radiator core (cooling fins and tubes), upper radiator tank, upper hose inlet neck, and radiator pressure cap. |
| **Exclusions (Outside the Box)** | Coolant expansion reservoir tank, air conditioning condenser (unless physically integrated), front grille mesh, or front bumper cover. |
| **Critical Anti-Pattern** | Do not label the translucent coolant overflow tank as the radiator. The radiator is the large finned heat exchanger mounted behind the grille. |
| **Acceptable Image Types** | Front-facing overhead or forward view into the core support showing the radiator top tank and upper fins. |
| **Unsuitable Image Types** | Exterior front grille photos where the radiator fins are completely hidden in darkness. |

---

### Class 4: `exhaust_manifold`

| Attribute | Specification |
|---|---|
| **Class ID** | `4` |
| **Semantic Meaning** | Exhaust collector header bolted directly to the cylinder head exhaust ports. |
| **Inclusions (Inside the Box)** | Cast iron or tubular steel runners collecting exhaust gases from cylinder head ports, manifold mounting flange, and attached heat shield (if factory installed). |
| **Exclusions (Outside the Box)** | Downpipe, catalytic converter (unless an integrated manifold-converter unit), intermediate exhaust pipes, resonators, and rear mufflers / tailpipes. |
| **Critical Anti-Pattern** | **DO NOT** label a muffler or rear exhaust tip as an exhaust manifold. Only the manifold attached to the cylinder head qualifies. |
| **Acceptable Image Types** | Clear view of the engine cylinder head exhaust side (often between engine block and firewall, or at the front of transverse engines). |
| **Unsuitable Image Types** | Undercarriage photos showing only the exhaust pipe leading to the muffler. |

---

## 3. Ambiguity & Edge-Case Decision Matrix

| Scenario | Decision Rule |
|---|---|
| **Component covered by factory heat shield** | If the heat shield directly covers the exhaust manifold and defines its external form, enclose the heat shield and mark as `exhaust_manifold`. |
| **Battery inside protective thermal blanket** | If the battery is enclosed in a fabric thermal sleeve or plastic OEM box, annotate the entire enclosed unit as `battery`. |
| **Multiple identical components** | If a twin-turbo or V-engine has dual exhaust manifolds or dual batteries, create **separate bounding boxes** for each instance. |
| **Dust, grease, or road grime** | Annotate dusty or grimy components normally. Diversity in surface condition is essential for real-world workshop deployment. |
| **Blurry or motion-blurred image** | If the component edges cannot be distinguished within 10 pixels, discard the image. Do not guess boundaries. |

---

## 4. Roboflow Annotation Setup Checklist

When importing and annotating batches on Roboflow:
1. Verify class ontology indices match zero-indexed prototype order:
   - `0: engine_bay`
   - `1: battery`
   - `2: alternator`
   - `3: radiator`
   - `4: exhaust_manifold`
2. Maintain an 80% / 15% / 5% train/val/test split across all added batches.
3. Apply standard preprocessing: Auto-Orient, Resize to 640x640 (Stretch or Fit with black borders).
4. Export as **YOLOv8 PyTorch** format for seamless staging into `data/prototype_5class/`.
