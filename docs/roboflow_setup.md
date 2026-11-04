# Roboflow Project Setup & Configuration Specification

**Project Title:** AI-Powered Multimodal Vehicle Fault Detection and Diagnosis System  
**Stage:** Stage 1 – Vehicle Component Detection Prototype  
**Date of Audit & Setup Documentation:** November 3, 2026  
**Document Status:** Pending User Authentication (Manual Project Creation Required)  

---

## 1. Project Specifications

| Parameter | Configuration Value | Justification / Notes |
|---|---|---|
| **Platform** | [Roboflow](https://roboflow.com/) | Cloud-based collaborative annotation and dataset version control platform |
| **Project Name** | `Vehicle Component Detection` | Dedicated repository for Stage 1 component object detection |
| **Project Type** | `Object Detection` | Bounding box localization of distinct vehicle hardware components |
| **Annotation Type** | `Bounding Boxes` (Rectangular 2D) | Required for YOLOv8 coordinate format (`x_center`, `y_center`, `width`, `height`) |
| **Model Objective** | `YOLOv8 Nano (yolov8n)` | Ultra-lightweight edge detector optimized for NVIDIA RTX 3050 (4GB VRAM) |
| **Target Framework** | Ultralytics YOLOv8 PyTorch | Native compatibility with `ml/training/train_components.py` |

---

## 2. Five-Class Prototype Ontology

The Stage 1 prototype focuses on 5 core under-hood mechanical and electrical components. These classes must be configured in Roboflow in exactly this sequential order to align with zero-indexed YOLO class IDs:

| Class ID | Class Name | Exact Semantic Definition | Explicit Exclusions & Anti-Patterns |
|---|---|---|---|
| **0** | `engine_bay` | The complete vehicle engine compartment perimeter enclosure viewed with the hood open. | **DO NOT** annotate the engine block alone. Must represent the macroscopic under-hood bay area containing the powertrain assembly. |
| **1** | `battery` | The 12V automotive starter battery, rectangular casing, positive/negative terminals, clamp cables, and hold-down bracket. | Exclude standalone fuse boxes or ECU housings adjacent to the battery unless physically integrated into the terminal post. |
| **2** | `alternator` | Serpentine-belt-driven AC electrical generator unit mounted to the front/side of the engine block, including casing, front pulley, and stator ventilation vents. | Exclude the AC compressor, water pump, or belt tensioner idler pulleys. |
| **3** | `radiator` | Front engine cooling radiator assembly, comprising core cooling fins, top/bottom tanks, upper radiator hose connection, and radiator cap. | Exclude the coolant overflow expansion reservoir tank, AC condenser, or auxiliary oil cooler unless physically bounding the main radiator assembly. |
| **4** | `exhaust_manifold` | Cast iron or tubular stainless-steel exhaust manifold/header assembly bolted directly to the cylinder head exhaust ports. | **DO NOT** map to downpipes, catalytic converters, intermediate resonators, or rear mufflers/tailpipes. Only the manifold attached to the cylinder head qualifies. |

### Architectural Safeguard: Preservation of 10-Class Master Ontology
The production system defines a 10-class canonical architecture in `ml/config/components_dataset.yaml`:
- `0: engine_bay`, `1: battery`, `2: alternator`, `3: radiator`, `4: brake_rotor`, `5: brake_caliper`, `6: tire_tread`, `7: tire_sidewall`, `8: suspension_strut`, `9: exhaust_manifold`.

The 5-class prototype isolates classes `{0, 1, 2, 3, 9}` remapped to contiguous IDs `{0, 1, 2, 3, 4}` in `ml/config/prototype_5class.yaml`. The 10-class dataset configuration remains completely untouched.

---

## 3. Automation Audit & Setup Status

### Audit Finding
- **Current Status:** **PENDING USER AUTHENTICATION**
- **Automated Execution Barrier:**
  1. Automated headless project creation via API or browser requires an authenticated Roboflow user session or an active `ROBOFLOW_API_KEY`.
  2. Inspection of the local environment (`.env`, Python `.venv`, plugin accounts) confirms that no Roboflow credentials or authentication tokens exist on this system.
  3. Roboflow Universe and web endpoints enforce Cloudflare bot protection, preventing non-interactive scripted logins.
- **Academic & Operational Integrity:** We do **not** claim the project has already been created in the cloud. The project configuration has been fully prepared locally, and simple manual creation steps are provided below.

---

## 4. Manual Setup Guide (Step-by-Step)

Follow these precise steps to instantiate the project on Roboflow:

### Step 1: Account Login
1. Navigate to [https://roboflow.com/](https://roboflow.com/) in your web browser.
2. Sign in to your existing account (or create a free account using your student/developer email).

### Step 2: Create the Project
1. In your Roboflow dashboard, select your workspace (or click **Create Workspace**, e.g., `AI-Vehicle-Diagnosis`).
2. Click the **Create New Project** button.
3. Configure project attributes:
   - **Project Name:** `Vehicle Component Detection`
   - **Project Type:** Select **Object Detection**
   - **What are you detecting?:** Enter `Vehicle Components`
   - **Annotation Group:** Bounding Boxes
4. Click **Create Project**.

### Step 3: Configure the 5 Classes
In the project settings / ontology view, add the 5 classes in order:
1. `engine_bay`
2. `battery`
3. `alternator`
4. `radiator`
5. `exhaust_manifold`

*(Roboflow assigns class IDs automatically based on class creation order. Verify that the order matches the table above).*

### Step 4: Obtain Your Roboflow API Key (Optional for CLI sync)
1. In the upper-right corner of the Roboflow dashboard, click your avatar and select **Roboflow Settings**.
2. Click **Workspaces** -> Select your workspace -> **Roboflow API**.
3. Copy your **Private API Key**.
4. In your project root (`d:\AI-Vehicle-Diagnosis-System\.env`), append:
   ```env
   ROBOFLOW_API_KEY=your_private_api_key_here
   ROBOFLOW_WORKSPACE=your_workspace_slug
   ROBOFLOW_PROJECT=vehicle-component-detection
   ```

### Step 5: Exporting Datasets to the Monorepo
When annotations are finalized:
1. In Roboflow, navigate to **Generate** -> **Create Version**.
2. Keep preprocessing standard (e.g., Auto-Orient, Resize to 640x640).
3. Click **Export Dataset** -> Select format: **YOLOv8**.
4. Select **download zip to computer**.
5. Extract the downloaded package into `data/prototype_5class/` (or into `data/raw/public_dataset_sample/` for initial inspection).
