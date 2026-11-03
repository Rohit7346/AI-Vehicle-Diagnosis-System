# AI-Powered Multimodal Vehicle Fault Detection and Diagnosis System

An engineering-grade, production-ready monorepo platform designed for autonomous vehicular subsystem inspection, component localization, and visible defect diagnosis.

Developed as a **Final-Year Engineering Research Project**, this system bridges computer vision, edge-ready inference backends, and responsive operator dashboards into a decoupled, reproducible architecture.

---

## 1. System Architecture & Dual-Stage Perception

Rather than treating automotive defect inspection as a black-box classification problem, this system utilizes a **hierarchical two-stage visual perception pipeline**:

```
                                  [ Raw Inspection Image ]
                                              │
                                              ▼
                             [ Image Preprocessor (CLAHE, Resize) ]
                                              │
                     ┌────────────────────────┴────────────────────────┐
                     │                                                 │
                     ▼                                                 ▼
        [ Stage 1: Component Detector ]                 [ Stage 2: Fault Detector ]
          • Ultralytics YOLOv8m                           • Ultralytics YOLOv8m
          • 10 Automotive Subsystems                      • 12 Visible Defect Classes
                     │                                                 │
                     └────────────────────────┬────────────────────────┘
                                              │
                                              ▼
                        [ Spatial Correlation Engine (IoA >= 0.30) ]
                                              │
                                              ▼
                        [ Health Scoring & Prescriptive Actions ]
                                              │
                                              ▼
                                   [ Diagnostic Report ]
```

### Stage 1: Subsystem Component Localization
Identifies and bounds functional assemblies:
1. `engine_bay` - Main powertrain compartment
2. `battery` - 12V / traction battery casing and tie-downs
3. `alternator` - Alternator assembly and pulley
4. `radiator` - Core heat exchanger matrix and fins
5. `brake_rotor` - Brake disc assembly and friction contact face
6. `brake_caliper` - Caliper housing and pad bracket
7. `tire_tread` - Outer wheel tread surface and circumferential grooves
8. `tire_sidewall` - Outer lateral wall from bead to shoulder
9. `suspension_strut` - Coil spring and damper tube
10. `exhaust_manifold` - Exhaust runner pipes and catalytic junction

### Stage 2: Visible Defect & Anomaly Classification
Detects physical wear patterns, structural cracks, and fluid leakage:
1. `fluid_leak_oil` (HIGH) - Engine/gear oil seepage
2. `fluid_leak_coolant` (HIGH) - Coolant crust/puddles
3. `fluid_leak_brake` (CRITICAL) - Brake fluid line/caliper leaks
4. `metal_corrosion` (MODERATE) - Structural/surface oxidation
5. `belt_cracking` (HIGH) - Serpentine/timing belt transverse splits
6. `hose_deterioration` (MODERATE) - Coolant/vacuum line dry rot
7. `terminal_oxidation` (LOW) - Sulfate buildup on battery posts
8. `rotor_grooving` (CRITICAL) - Deep scoring on brake rotor
9. `pad_wear_critical` (CRITICAL) - Brake pad lining < 3mm
10. `tire_tread_depth_low` (HIGH) - Tread wear indicator flush (< 1.6mm)
11. `tire_sidewall_bulge` (CRITICAL) - Tire cord separation/delamination
12. `structural_dent_crack` (HIGH) - Sharp impact deformation or fracture

---

## 2. Monorepo Directory Layout

```
AI-Vehicle-Diagnosis-System/
├── .env.example                  # Environment configuration template
├── .gitignore                    # Rigorous ignore rules (weights, data, venv)
├── requirements.txt              # Production Python dependencies
├── requirements-dev.txt          # Developer, linting, and testing dependencies
├── README.md                     # Comprehensive architecture and setup documentation
│
├── data/                         # Dataset root (tracked via .gitkeep)
│   ├── README.md                 # Taxonomy specification & annotation guide
│   ├── raw/                      # Untouched original high-res imagery & audio
│   │   ├── images/               # High-res raw photos
│   │   └── audio/                # Acoustic engine signatures (future expansion)
│   ├── processed/                # Normalized 640x640 images
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── annotations/              # YOLO label files (<class_id> <x> <y> <w> <h>)
│       ├── train/
│       ├── val/
│       └── test/
│
├── docs/                         # Academic and engineering specifications
│   ├── architecture.md           # Deep dive into system design and dataflows
│   ├── dataset_specifications.md # Photography protocols, lighting, and classes
│   ├── model_methodology.md      # Dual-stage equations, IoA, and metrics
│   └── api_specifications.md     # OpenAPI / REST contracts and payloads
│
├── ml/                           # Standalone Machine Learning Core
│   ├── config/                   # Detection YAML configs and dataclasses
│   │   ├── base_config.py
│   │   ├── component_config.yaml
│   │   └── fault_config.yaml
│   ├── datasets/                 # Ingestion, CLAHE preprocessor, annotation validator
│   │   ├── collector.py
│   │   ├── preprocessor.py
│   │   └── annotator_utils.py
│   ├── models/                   # Modular YOLO detector wrappers
│   │   ├── base_model.py
│   │   ├── component_detector.py
│   │   └── fault_detector.py
│   ├── training/                 # Training pipelines with pre-flight gates
│   │   ├── train_components.py
│   │   └── train_faults.py
│   ├── evaluation/               # Benchmarking metrics (mAP, IoU, confusion matrix)
│   │   ├── metrics.py
│   │   └── evaluator.py
│   ├── inference/                # Unified diagnostic orchestrator & Pydantic schemas
│   │   ├── schemas.py
│   │   └── engine.py
│   └── weights/                  # Trained model checkpoints (.pt, .onnx)
│       └── README.md
│
├── backend/                      # FastAPI Asynchronous REST Backend
│   ├── run.py                    # Local development launcher
│   └── app/
│       ├── main.py               # Application factory, CORS, lifespan
│       ├── core/                 # App settings (Pydantic BaseSettings)
│       │   ├── config.py
│       │   └── security.py
│       ├── schemas/              # Request and response models
│       │   └── diagnosis.py
│       ├── services/             # Service layer bridging FastAPI and ML engine
│       │   └── diagnosis_service.py
│       └── api/v1/               # Aggregated version 1 API routes
│           ├── router.py
│           └── endpoints/
│               ├── health.py     # System telemetry & model readiness probe
│               └── diagnosis.py  # Image upload & inference controller
│
├── frontend/                     # React + Vite + Tailwind CSS Dashboard
│   ├── package.json
│   ├── vite.config.js            # Reverse proxy configuration to FastAPI
│   ├── tailwind.config.js
│   ├── index.html
│   └── src/
│       ├── App.jsx               # Navigation, state management & periodic health polling
│       ├── index.css             # Tailwind base and dark theme rules
│       ├── services/api.js       # Axios client for backend communication
│       ├── components/           # Modular visual components
│       │   ├── Navbar.jsx
│       │   ├── Sidebar.jsx
│       │   ├── SystemStatus.jsx
│       │   ├── ImageUploader.jsx
│       │   ├── DiagnosisResultCard.jsx
│       │   ├── ComponentList.jsx
│       │   └── FaultBadge.jsx
│       └── pages/                # Dashboard views
│           ├── Dashboard.jsx
│           ├── DiagnosticsPage.jsx
│           ├── ModelStatusPage.jsx
│           └── ResearchDocsPage.jsx
│
├── scripts/                      # Operational utilities
│   └── verify_environment.py     # Complete environment audit & check
└── tests/                        # Comprehensive unit & integration test suite
    ├── test_preprocessor.py
    ├── test_annotator.py
    ├── test_inference_engine.py
    └── test_api_endpoints.py
```

---

## 3. Technology Stack

- **Computer Vision & AI/ML**: Python 3.13, PyTorch, Ultralytics YOLOv8, OpenCV, Pillow, NumPy
- **Backend Service**: Python FastAPI, Uvicorn, Pydantic v2, Pydantic-Settings
- **Frontend Dashboard**: React 19, Vite 6, Tailwind CSS 3.4, Lucide Icons, Axios
- **Database Readiness**: SQLAlchemy 2.0 & PostgreSQL (prepared for next phase)
- **Testing & Code Quality**: Pytest, Pytest-Asyncio, HTTPX

---

## 4. Setup & Installation Instructions

### Prerequisites
- Python 3.10+ (Python 3.13 supported)
- Node.js v20+ and npm
- Git

### Step 1: Clone Repository & Create Virtual Environment
```bash
# Clone the repository
git clone https://github.com/your-username/AI-Vehicle-Diagnosis-System.git
cd AI-Vehicle-Diagnosis-System

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate
```

### Step 2: Install Python Dependencies
```bash
# Install core runtime dependencies
pip install -r requirements.txt

# Or install development and test dependencies
pip install -r requirements-dev.txt
```

### Step 3: Install Frontend Dependencies
```bash
cd frontend
npm install
cd ..
```

### Step 4: Configure Environment Variables
```bash
cp .env.example .env
```

---

## 5. Development Workflow & Running the System

### 1. Verify Environment Integrity
Run the built-in diagnostic audit script:
```bash
python scripts/verify_environment.py
```

### 2. Start the Backend API (FastAPI)
From the repository root:
```bash
python backend/run.py
# Server starts on http://localhost:8000
# OpenAPI Interactive Documentation: http://localhost:8000/docs
```

### 3. Start the Frontend Dashboard (React + Vite)
In a separate terminal:
```bash
cd frontend
npm run dev
# Dashboard launches on http://localhost:5173
```

---

## 6. Dataset Preparation & Model Training Protocol

> **ACADEMIC INTEGRITY NOTICE:**
> The repository explicitly refuses to emit simulated, fake, or mocked predictions.
> The inference engine strictly checks for real trained weights. If weights are missing, the system reports uninitialized status (`HTTP 503` with an actionable diagnosis message).

### Workflow to Train Models:
1. **Collect Images**: Place authentic inspection photographs into `data/raw/images/`.
2. **Audit Raw Data**:
   ```bash
   python -c "from ml.datasets.collector import DatasetCollector; print(DatasetCollector().audit_raw_data())"
   ```
3. **Preprocess Imagery**: Run CLAHE contrast enhancement and letterbox resizing into `data/processed/train/`.
4. **Annotate**: Label targets using tools such as CVAT or LabelImg and store `.txt` files into `data/annotations/train/`.
5. **Verify Annotations**:
   ```bash
   python -c "from ml.datasets.annotator_utils import AnnotationValidator; from pathlib import Path; print(AnnotationValidator.validate_split(Path('data/processed/train'), Path('data/annotations/train'), 10))"
   ```
6. **Train Component Detector**:
   ```bash
   python -m ml.training.train_components --epochs 100 --batch-size 16
   ```
7. **Train Fault Detector**:
   ```bash
   python -m ml.training.train_faults --epochs 120 --batch-size 16
   ```
8. **Place Checkpoints**: Copy trained `best.pt` checkpoints to:
   - `ml/weights/component_detector_best.pt`
   - `ml/weights/fault_detector_best.pt`

---

## 7. Automated Testing Suite

Execute the comprehensive test suite covering preprocessing, label parsing, inference safeguards, and API contracts:

```bash
pytest -v
```

All 8 tests run automatically and validate:
- Image CLAHE and letterbox padding mathematics
- YOLO annotation syntax and coordinate boundaries
- Safeguards preventing fake prediction emission when weights are missing
- FastAPI `/health` telemetry and `/diagnose/image` handling

---

## 8. Research Roadmap & Future Extensions

- [x] Decoupled monorepo architecture (ML, Backend, Frontend)
- [x] Dual-stage vision perception contracts & configuration
- [x] Image preprocessing pipeline (CLAHE & Letterbox)
- [x] Pre-flight dataset verification gates
- [x] FastAPI REST API with health and image ingestion
- [x] React dashboard with dark-theme telemetry
- [ ] Multimodal acoustic engine fault classification (`data/raw/audio`)
- [ ] Persistent telemetry & report storage with PostgreSQL and Alembic
- [ ] TensorRT / ONNX Runtime edge deployment for embedded inspection devices
