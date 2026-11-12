# Stage 3.1: YOLOv8 Nano Baseline Model Training Report

**Project Title:** AI-Powered Multimodal Vehicle Fault Detection and Diagnosis System  
**Stage:** Stage 3.1 – YOLOv8 Nano Baseline Training  
**Architecture:** YOLOv8 Nano (`yolov8n.pt`, 3.2M parameters)  
**Hardware Platform:** NVIDIA GeForce RTX 3050 Laptop GPU (4 GB GDDR6)  
**Compute Framework:** PyTorch 2.6.0+cu124, Ultralytics 8.4.210  
**Training Run Root:** [`runs/detect/stage3_1_yolov8n/`](file:///D:/AI-Vehicle-Diagnosis-System/runs/detect/stage3_1_yolov8n/)  
**Configuration File:** [`runs/detect/stage3_1_yolov8n/args.yaml`](file:///D:/AI-Vehicle-Diagnosis-System/runs/detect/stage3_1_yolov8n/args.yaml)  
**Results CSV:** [`runs/detect/stage3_1_yolov8n/results.csv`](file:///D:/AI-Vehicle-Diagnosis-System/runs/detect/stage3_1_yolov8n/results.csv)  
**Loss & Metric Curves:** [`runs/detect/stage3_1_yolov8n/results.png`](file:///D:/AI-Vehicle-Diagnosis-System/runs/detect/stage3_1_yolov8n/results.png)  
**Dataset Specification:** [`data/processed/stage3_0_three_class/data.yaml`](file:///D:/AI-Vehicle-Diagnosis-System/data/processed/stage3_0_three_class/data.yaml)  
**Environment Audit:** [`docs/stage3_1_environment_report.md`](file:///D:/AI-Vehicle-Diagnosis-System/docs/stage3_1_environment_report.md)  
**Evaluation Report:** [`docs/stage3_1_evaluation_report.md`](file:///D:/AI-Vehicle-Diagnosis-System/docs/stage3_1_evaluation_report.md)  
**Execution Date:** November 11, 2026  

---

## 1. Executive Summary

Stage 3.1 trained the project's first baseline object detection model using the certified three-class dataset from Stage 3.0. A YOLOv8 Nano model (`yolov8n.pt`) was fine-tuned across 100 epochs on an NVIDIA GeForce RTX 3050 Laptop GPU.

All training operations strictly honored project constraints:
1. Sourced directly from [`data/processed/stage3_0_three_class/data.yaml`](file:///D:/AI-Vehicle-Diagnosis-System/data/processed/stage3_0_three_class/data.yaml) without altering raw data or prototype datasets.
2. Verified 1:1 image-label pairings and class mappings before training.
3. Applied conservative hyperparameters to remain within the 4 GB VRAM envelope.
4. Preserved all training curves, confusion matrices, logs, and model checkpoints (`best.pt`, `last.pt`).
5. Kept model binaries and runs excluded from Git tracking via `.gitignore`.

---

## 2. Training Hyperparameters & Execution Configuration

The complete hyperparameter profile from [`runs/detect/stage3_1_yolov8n/args.yaml`](file:///D:/AI-Vehicle-Diagnosis-System/runs/detect/stage3_1_yolov8n/args.yaml):

| Parameter | Configured Value | Operational Purpose |
|---|---|---|
| **Base Architecture** | `yolov8n.pt` (COCO pre-trained, 3.13M parameters) | Lightweight real-time edge deployment baseline |
| **Input Resolution** | `640 x 640` | Square letterboxed input preserving wide component aspect ratios |
| **Total Epochs** | `100` | Full training convergence trajectory |
| **Early Stopping Patience** | `20` epochs | Early-stop safeguard (model converged naturally through epoch 100) |
| **Batch Size** | `2` | Conservative allocation to prevent CUDA OOM on 4 GB VRAM |
| **Compute Device** | `0` (CUDA: NVIDIA RTX 3050) | Accelerated training with FP16 Automatic Mixed Precision (`amp=True`) |
| **Dataloader Workers** | `2` (initial) $\rightarrow$ `0` (resumed) | Adjusted to `0` to prevent Windows OpenCV worker memory fragmentation |
| **RAM Caching** | `cache=False` | Streamed disk reads avoiding physical memory saturation |
| **Random Seed** | `42` (`deterministic=True`) | Guaranteed algorithmic reproducibility across gradient updates |
| **Optimizer** | `AdamW` (`lr0=0.001429`, `momentum=0.937`) | Adaptive gradient optimization with decoupled weight decay |
| **Loss Function** | CIoU + Distribution Focal Loss (DFL) + BCE | Multi-task bounding box regression and classification |

---

## 3. Dataset Distribution & Input Split Accounting

The unified dataset was consumed strictly according to the Stage 3.0 distribution:

| Partition | Images Processed | Bounding Boxes | Engine Bay (`0`) | Battery (`1`) | Radiator (`2`) |
|---|:---:|:---:|:---:|:---:|:---:|
| **`train`** | 479 valid (1 long-path ignored) | 527 | 136 | 312 | 79 |
| **`val`** | 146 | 167 | 37 | 103 | 27 |
| **`test`** | 70 | 77 | 18 | 50 | 9 |
| **TOTAL** | **696 unique** | **771** | **191** | **465** | **115** |

*Note: 76 multi-class co-occurrences (Battery + Radiator in the same image) were correctly processed as multi-box labels.*

---

## 4. Hardware Telemetry & Resource Utilization

- **GPU Hardware:** NVIDIA GeForce RTX 3050 Laptop GPU (4,096 MiB GDDR6, compute capability 8.6).
- **VRAM Utilization:** Peak memory consumption reached **495 MiB** ($\approx 12.1\%$ of available VRAM).
- **GPU Duty Cycle:** Compute utilization averaged between 25% and 85% during batch iterations.
- **CUDA OOM Events:** **Zero CUDA out-of-memory errors** occurred. The conservative `batch=2` envelope safely prevented GPU memory saturation.
- **Operating Temperature:** Maintained steady between 49°C and 55°C.
- **Total Training Duration:** ~1.7 hours (~55 min for epochs 1–60 + 47.08 min for epochs 61–100).

---

## 5. Technical Challenges & Engineering Remediations

### 5.1 DataLoader Worker Memory Accumulation at Epoch 61
- **Observation:** At epoch 61, DataLoader worker process 1 threw `cv2.error: (-4:Insufficient memory) Failed to allocate 27433728 bytes in function 'cv::OutOfMemoryError'`.
- **Root Cause:** In Windows Python multiprocessing, PyTorch dataloader worker processes spawned via `spawn` accumulate image memory allocations across tens of thousands of image read iterations when using OpenCV 5.0 C-extensions.
- **Remediation:** Sourced checkpoint `runs/detect/stage3_1_yolov8n/weights/last.pt` and seamlessly resumed training with `workers=0`. In single-process mode, image loading runs in the main thread with immediate garbage collection, completely eliminating subprocess memory leaks and training to epoch 100 without interruption.

### 5.2 Extended Path Length in Train Split
- **Observation:** One image (`compartimento-do-motor-de-coleta-toyota-hilux-...269971302_jpg.rf.18e3c8a76797b6eda69827af966ea857.jpg`) has a 277-character filename that exceeded standard Windows MAX_PATH when scanned by OpenCV's native C++ reader.
- **Handling:** Ultralytics logged `ignoring corrupt image/label: [Errno 2] No such file or directory` and safely trained on the remaining 479 training images without failure.

---

## 6. Training Progression & Checkpoint Preservation

### 6.1 Key Epoch Checkpoint Metrics

| Epoch Milestone | Training Loss (Box / Cls / DFL) | Validation Precision | Validation Recall | Validation mAP@0.5 | Validation mAP@0.5:0.95 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **Epoch 1** | 1.923 / 4.042 / 1.948 | 0.272 | 0.329 | 0.281 | 0.128 |
| **Epoch 25** | 1.341 / 1.488 / 1.450 | 0.655 | 0.584 | 0.612 | 0.325 |
| **Epoch 50** | 1.182 / 1.121 / 1.385 | 0.652 | 0.651 | 0.642 | 0.332 |
| **Epoch 75** | 1.054 / 0.942 / 1.258 | 0.710 | 0.615 | 0.668 | 0.389 |
| **Epoch 100** | 0.985 / 0.812 / 1.194 | **0.732** | **0.623** | **0.676** | **0.412** |

### 6.2 Checkpoint Preservation
- **Best Weights Checkpoint:** [`runs/detect/stage3_1_yolov8n/weights/best.pt`](file:///D:/AI-Vehicle-Diagnosis-System/runs/detect/stage3_1_yolov8n/weights/best.pt) (6.50 MB, optimized FP16)
- **Last Weights Checkpoint:** [`runs/detect/stage3_1_yolov8n/weights/last.pt`](file:///D:/AI-Vehicle-Diagnosis-System/runs/detect/stage3_1_yolov8n/weights/last.pt) (6.50 MB)
- **Results Log:** [`runs/detect/stage3_1_yolov8n/results.csv`](file:///D:/AI-Vehicle-Diagnosis-System/runs/detect/stage3_1_yolov8n/results.csv)
- **Performance Curves:** [`runs/detect/stage3_1_yolov8n/results.png`](file:///D:/AI-Vehicle-Diagnosis-System/runs/detect/stage3_1_yolov8n/results.png)
- **Confusion Matrix:** [`runs/detect/stage3_1_yolov8n/confusion_matrix.png`](file:///D:/AI-Vehicle-Diagnosis-System/runs/detect/stage3_1_yolov8n/confusion_matrix.png)
- **Precision-Recall Curve:** [`runs/detect/stage3_1_yolov8n/BoxPR_curve.png`](file:///D:/AI-Vehicle-Diagnosis-System/runs/detect/stage3_1_yolov8n/BoxPR_curve.png)
- **F1 Score Curve:** [`runs/detect/stage3_1_yolov8n/BoxF1_curve.png`](file:///D:/AI-Vehicle-Diagnosis-System/runs/detect/stage3_1_yolov8n/BoxF1_curve.png)
