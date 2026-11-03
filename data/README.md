# Dataset Management & Taxonomy Specification

This directory holds all data used for training, validating, and testing the **AI-Powered Multimodal Vehicle Fault Detection System**.

> **IMPORTANT ARCHITECTURAL RULE:**
> No raw or processed image files are committed to Git. Only `.gitkeep` markers and this documentation are tracked.

---

## 1. Directory Structure

```
data/
├── raw/                  # Original unprocessed sensor feeds
│   ├── images/           # Uncompressed vehicle imagery (JPEG / PNG)
│   └── audio/            # (Future) Engine acoustic signatures (WAV / FLAC)
├── processed/            # Canonical splits ready for YOLO / PyTorch pipelines
│   ├── train/            # Standardized 640x640 images for training
│   ├── val/              # Validation split
│   └── test/             # Held-out benchmark split
└── annotations/          # Label coordinates matching processed images
    ├── train/            # YOLO format .txt (or COCO format .json)
    ├── val/
    └── test/
```

---

## 2. Target Classes & Taxonomy

The multimodal diagnostic pipeline is split into **two distinct visual perception tasks**:

### Task A: Vehicle Component Localization
Detects and isolates functional sub-assemblies of the vehicle:
1. `engine_bay` - Global view of powertrain compartment
2. `battery` - 12V lead-acid / traction battery pack
3. `alternator` - Alternator assembly & belt tensioner
4. `radiator` - Heat exchanger & cooling fins
5. `brake_rotor` - Brake disc assembly
6. `brake_caliper` - Hydraulic caliper & pads
7. `tire_tread` - Outer contact surface of wheel
8. `tire_sidewall` - Outer lateral wall of tire
9. `suspension_strut` - Shock absorber / spring assembly
10. `exhaust_manifold` - Header / catalytic / tailpipe junction

### Task B: Visible Fault & Defect Classification
Localizes specific physical abnormalities, wear patterns, or damage:
1. `fluid_leak_oil` - Engine / gear oil discoloration and seepage
2. `fluid_leak_coolant` - Green/pink ethylene glycol crystalline deposits
3. `fluid_leak_brake` - Brake fluid seepage near lines/calipers
4. `metal_corrosion` - Surface oxidation, scaling, or structural rust
5. `belt_cracking` - Transverse cracks or fraying on serpentine/timing belts
6. `hose_deterioration` - Bulging, cracking, or softening in coolant/vacuum lines
7. `terminal_oxidation` - White/blue sulfate powder on battery terminals
8. `rotor_grooving` - Deep circular scoring or heat hairline cracks on brake disc
9. `pad_wear_critical` - Brake friction material < 3mm remaining
10. `tire_tread_depth_low` - Tire wear indicator bars exposed (< 1.6mm)
11. `tire_sidewall_bulge` - Structural cord fracture in tire wall
12. `structural_dent_crack` - Impact damage, fracture, or weld failure

---

## 3. Annotation Formats

### YOLO Annotation Standard (Default)
Each image `image_0001.jpg` in `processed/train/` has a matching `image_0001.txt` in `annotations/train/`:
```
<class_id> <x_center> <y_center> <width> <height>
```
*Coordinates are normalized between `0.0` and `1.0`.*

---

## 4. Verification Check Before Training
Before initiating model training with `python -m ml.training.train_components` or `train_faults`:
1. Place raw photos into `data/raw/images/`.
2. Run `ml.datasets.preprocessor` to produce normalized 640x640 images into `data/processed/`.
3. Annotate using tools such as CVAT, LabelImg, or Roboflow and place `.txt` files into `data/annotations/`.
4. Run `python -m ml.datasets.annotator_utils --verify` to guarantee bounding box coordinate boundaries and image-label synchronization.
