# Dataset Collection & Annotation Protocol

## 1. Ethical & Academic Rigor Guidelines
- Random web scraping without attribution and quality controls is strictly forbidden.
- All vehicular imagery must satisfy rigorous physical inspection parameters to ensure model generalization across workshop environments, road debris, and variable illumination.

---

## 2. Imaging Protocols & Environmental Controls
To minimize domain shift between training and field deployment:

1. **Resolution & Optical Quality:**
   - Minimum capture resolution: `1920x1080` (1080p).
   - Optical focus must be locked onto the target subassembly (no motion blur).
2. **Illumination Conditions:**
   - Direct daylight (diffuse overcast preferred to minimize harsh specular glares).
   - Workshop artificial lighting (LED work lamps / inspection torches).
   - Low-light undercarriage imagery with controlled flashlight compensation.
3. **Capture Angles:**
   - Orthogonal (direct normal view, $90^\circ \pm 10^\circ$).
   - Oblique (grazing angle, $45^\circ \pm 15^\circ$) for depth analysis (e.g., brake pad remaining thickness, tread wear).

---

## 3. Class Definitions & Annotation Ontology

### Subsystem Classes (Component Detection)
| Class ID | Class Label | Visual Identifier & Scope |
| :--- | :--- | :--- |
| `0` | `engine_bay` | Entire visible bay or major block assembly |
| `1` | `battery` | Battery casing, tie-down brackets, and cell terminals |
| `2` | `alternator` | Outer casing, pulley, serpentine contact |
| `3` | `radiator` | Core matrix, cooling tubes, and upper tank |
| `4` | `brake_rotor` | Steel/cast disc surface, hat, and edge |
| `5` | `brake_caliper` | Caliper body, bleeder screw, and brake pad backing |
| `6` | `tire_tread` | Circumferential grooves and contact patches |
| `7` | `tire_sidewall` | Lateral tire face from rim lip to shoulder |
| `8` | `suspension_strut` | Coil spring, damper body, and dust boot |
| `9` | `exhaust_manifold` | Cast iron/steel runner pipes and catalytic joint |

### Defect Classes (Fault Detection)
| Class ID | Class Label | Defect Threshold / Severity Criterion |
| :--- | :--- | :--- |
| `0` | `fluid_leak_oil` | Viscous black/brown moisture pooling or coating surfaces |
| `1` | `fluid_leak_coolant` | Bright green, pink, or crystallized crust along hoses/radiator |
| `2` | `fluid_leak_brake` | Clear/amber oily fluid coating caliper seams or hydraulic hoses |
| `3` | `metal_corrosion` | Pitting rust, oxidized flaky scale (> 10% surface area) |
| `4` | `belt_cracking` | Transverse cracks separated by less than 15mm along belt ribs |
| `5` | `hose_deterioration` | Visible rubber microcracking, dry rot, or localized ballooning |
| `6` | `terminal_oxidation` | White/cyan lead sulfate powder buildup on battery posts |
| `7` | `rotor_grooving` | Concentric circular scoring ridges visible or lip > 1.5mm |
| `8` | `pad_wear_critical` | Remaining friction pad lining below 3mm |
| `9` | `tire_tread_depth_low` | Tread depth indicator bars flush with tread surface (< 1.6mm) |
| `10` | `tire_sidewall_bulge` | Localized outward bubble indicative of broken internal cords |
| `11` | `structural_dent_crack` | Sharp fracture lines or plastic deformation exceeding tolerance |

---

## 4. Train / Validation / Test Splits
- **Training Set (70%):** Model parameter optimization.
- **Validation Set (15%):** Hyperparameter tuning, early stopping criterion.
- **Test Set (15%):** Held-out academic benchmark for publication and final evaluation.
- **Splits must be stratified across vehicle makes and conditions** to prevent leakage across splits.
