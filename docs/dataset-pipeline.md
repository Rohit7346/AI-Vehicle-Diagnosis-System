# Ultralytics YOLO Dataset Pipeline Specification

**Subsystem**: Stage 1 Vehicle Component Detection (YOLOv8 Nano)  
**Standard**: Canonical Ultralytics YOLO Dual-Directory Architecture  
**Hardware Target**: NVIDIA GeForce RTX 3050 Laptop GPU (4GB VRAM)  
**Document Path**: `docs/dataset-pipeline.md`

---

## 1. Pipeline Architecture Overview

To achieve seamless, 100% native compatibility with Ultralytics YOLO without fragile custom path hooks, the dataset pipeline implements the standard **canonical dual-directory layout**:

```
data/
├── raw/                      # Untouched high-res sensor ingest
│   ├── images/               # Raw field inspection photographs
│   └── audio/                # Acoustic engine signatures (future expansion)
├── images/                   # Preprocessed 640x640 RGB images
│   ├── train/                # 70% Training split
│   ├── val/                  # 15% Validation split
│   └── test/                 # 15% Held-out academic benchmark split
└── labels/                   # YOLO normalized bounding-box coordinates
    ├── train/                # Matching .txt files for train
    ├── val/                  # Matching .txt files for val
    └── test/                 # Matching .txt files for test
```

### 1.1 Ultralytics Automatic Label Resolution
Ultralytics YOLO uses the internal path resolver `img2label_paths()`, which automatically maps:
$$\text{Image Path: } \texttt{data/images/}\langle\text{split}\rangle\texttt{/foo.jpg} \implies \text{Label Path: } \texttt{data/labels/}\langle\text{split}\rangle\texttt{/foo.txt}$$

By matching `data/images/` to `data/labels/`, dataset loading is completely automated with zero cache lookup warnings.

---

## 2. Canonical Class Ontology & Index Mapping

The 10 vehicle component classes are locked and synchronized across configuration files, validator utilities, and the inference engine:

| Class ID | Class Label | Physical Scope & Visual Boundaries |
| :---: | :--- | :--- |
| `0` | `engine_bay` | Global powertrain bay, cylinder block, and upper intake |
| `1` | `battery` | 12V lead-acid / traction battery housing, tie-downs, and terminals |
| `2` | `alternator` | Alternator housing, pulley wheel, and serpentine belt contact |
| `3` | `radiator` | Cooling fins matrix, upper/lower tanks, and fan shroud |
| `4` | `brake_rotor` | Steel/cast disc friction contact ring and center hat |
| `5` | `brake_caliper` | Hydraulic caliper body, bleeder nipple, and pad carrier |
| `6` | `tire_tread` | Circumferential tread grooves and outer contact patch |
| `7` | `tire_sidewall` | Lateral tire wall from rim flange to tread shoulder |
| `8` | `suspension_strut` | Damper body, coil spring, bump stop, and strut top |
| `9` | `exhaust_manifold` | Cast iron/stainless runner pipes and collector flange |

> [!IMPORTANT]
> Class IDs are fixed $[0 \dots 9]$. Changing or re-indexing classes will corrupt cross-split validation and break the spatial IoA correlation engine.

---

## 3. Annotation Syntax & Geometric Boundaries

Every label file `foo.txt` corresponding to `foo.jpg` contains one space-separated line per detected object:

```
<class_id> <x_center> <y_center> <width> <height>
```

### Verification Rules Enforced by `AnnotationValidator`:
1. **Column Count**: Exactly 5 space-separated numeric values per line.
2. **Class ID**: Integer $\in [0, 9]$.
3. **Normalized Coordinates**:
   $$0.0 \le x_{\text{center}} \le 1.0, \quad 0.0 \le y_{\text{center}} \le 1.0$$
4. **Dimension Sanity**:
   $$0.005 \le w \le 1.0, \quad 0.005 \le h \le 1.0$$
5. **Corner Boundaries**:
   $$x_{\min} = x_c - \frac{w}{2} \ge 0.0, \quad x_{\max} = x_c + \frac{w}{2} \le 1.0$$
   $$y_{\min} = y_c - \frac{h}{2} \ge 0.0, \quad y_{\max} = y_c + \frac{h}{2} \le 1.0$$
6. **One-to-One Pairing**: Every image must have a corresponding `.txt` file (empty `.txt` files denote verified negative samples without target components).

---

## 4. Academic Integrity & Dataset Safeguards

### 4.1 Data Leakage Prevention
The [`AnnotationValidator.detect_data_leakage()`](file:///d:/AI-Vehicle-Diagnosis-System/ml/datasets/annotator_utils.py) audit checks for:
- **Filename Collision**: Identical image file stems shared between `train`, `val`, or `test`.
- **Content Hash Collision**: SHA-256 hash duplication across splits (detects renamed duplicate images).
- If any overlap is found, the pre-flight gate halts training immediately with a `[ABORT]` signal.

### 4.2 Class Representation Gate
- Every target class ($0$ to $9$) must have representative samples in both the training split ($\ge 5$ annotations) and the validation split ($\ge 2$ annotations).
- The pipeline explicitly forbids starting training if any class has 0 annotations.

---

## 5. Configuration Isolation Architecture

In accordance with modular design principles, dataset definitions are strictly decoupled from training hyperparameters:

### 5.1 Dataset Specification ([`ml/config/components_dataset.yaml`](file:///d:/AI-Vehicle-Diagnosis-System/ml/config/components_dataset.yaml))
Contains **only** data paths and class ontology:
```yaml
path: ../../data
train: images/train
val: images/val
test: images/test

names:
  0: engine_bay
  1: battery
  2: alternator
  3: radiator
  4: brake_rotor
  5: brake_caliper
  6: tire_tread
  7: tire_sidewall
  8: suspension_strut
  9: exhaust_manifold
```

### 5.2 Training Hyperparameters ([`ml/config/hyperparameters_yolov8n.yaml`](file:///d:/AI-Vehicle-Diagnosis-System/ml/config/hyperparameters_yolov8n.yaml))
Contains **only** optimization and training parameters tuned for the 4GB RTX 3050:
- Model: `yolov8n.pt` (3.2M params)
- Input Size: `640x640`
- Batch Size: `16` (~1.8 GB VRAM footprint)
- Compute: `device: 0` (CUDA GPU)
- Mixed Precision: `amp: true` (FP16 Tensor Cores)
- Optimizer: `AdamW` (`lr0: 0.001`, `weight_decay: 0.0005`)
- Patience: `20` (Early stopping)

---

## 6. Checkpoint Verification & Synchronization

When Ultralytics executes training, artifacts are saved to:
```
runs/detect/<name>/
├── weights/
│   ├── best.pt              # Highest validation mAP@50-95 checkpoint
│   └── last.pt              # Final epoch state for training resumption
├── confusion_matrix.png     # Class confusion matrix
├── results.png              # Loss curves & mAP trajectory
└── args.yaml                # Serialized hyperparameter snapshot
```

### Automated Inference Synchronization:
Upon successful completion, [`train_components.py`](file:///d:/AI-Vehicle-Diagnosis-System/ml/training/train_components.py) verifies `best.pt` file integrity and automatically synchronizes it:
$$\texttt{runs/detect/component\_yolov8n/weights/best.pt} \xrightarrow{\text{copy}} \texttt{ml/weights/component\_detector\_best.pt}$$

Once placed in `ml/weights/`, the FastAPI backend (`GET /api/v1/health`) and React Operator Dashboard automatically detect the checkpoint and transition model readiness to **Loaded / Ready**.

---

## 7. Exact Step-by-Step Execution Guide

### Step 1: Inspect Dataset Status & Generate Summary Report
Run the audit utility to review image counts, label synchronization, and class distributions:
```powershell
.venv\Scripts\python.exe -m ml.datasets.annotator_utils
```

### Step 2: Populate Raw Images & Run Preprocessing
1. Place raw photos into `data/raw/images/`.
2. Annotate the preprocessed images using LabelImg, CVAT, or Roboflow with classes 0–9.
3. Save resulting 640x640 images into `data/images/train/` and `data/images/val/`.
4. Save matching `.txt` files into `data/labels/train/` and `data/labels/val/`.

### Step 3: Run Full Pre-Flight Validation
```powershell
.venv\Scripts\python.exe -c "from ml.datasets.annotator_utils import AnnotationValidator; from pathlib import Path; print(AnnotationValidator.audit_full_dataset(Path('data')))"
```

### Step 4: Run Smoke-Test (3 Epochs, Lightweight Verification)
```powershell
.venv\Scripts\python.exe -m ml.training.train_components --smoke-test
```

### Step 5: Execute Full Stage 1 Model Training (Upon Approval)
```powershell
.venv\Scripts\python.exe -m ml.training.train_components --data ml/config/components_dataset.yaml --hyp ml/config/hyperparameters_yolov8n.yaml
```
