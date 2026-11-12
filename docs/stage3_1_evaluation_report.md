# Stage 3.1: YOLOv8 Nano Baseline Evaluation & Diagnostics Report

**Project Title:** AI-Powered Multimodal Vehicle Fault Detection and Diagnosis System  
**Stage:** Stage 3.1 – YOLOv8 Nano Baseline Evaluation  
**Architecture:** YOLOv8 Nano (`yolov8n.pt`, 3.13M parameters)  
**Evaluated Checkpoint:** [`runs/detect/stage3_1_yolov8n/weights/best.pt`](file:///D:/AI-Vehicle-Diagnosis-System/runs/detect/stage3_1_yolov8n/weights/best.pt)  
**Evaluation Summary JSON:** [`runs/detect/stage3_1_yolov8n/eval_summary.json`](file:///D:/AI-Vehicle-Diagnosis-System/runs/detect/stage3_1_yolov8n/eval_summary.json)  
**Visual Samples Directory:** [`docs/stage3_1_visual_results/`](file:///D:/AI-Vehicle-Diagnosis-System/docs/stage3_1_visual_results/)  
**Evaluation Script:** `scratch/evaluate_and_visualize_stage3_1.py`  
**Execution Date:** November 11, 2026  

---

## 1. Executive Summary & Evaluation Scope

Following the successful 100-epoch training of the baseline YOLOv8 Nano model on the unified three-class dataset, independent quantitative evaluations were executed on:
1. **Validation Split (`val`):** 146 images, 167 ground-truth bounding boxes.
2. **Test Split (`test`):** 70 images, 77 ground-truth bounding boxes (including 1 intentional hard negative).

The evaluation measured Precision ($P$), Recall ($R$), Mean Average Precision at IoU threshold 0.50 ($\text{mAP}@0.5$), and comprehensive $\text{mAP}@[0.5:0.95]$.

---

## 2. Quantitative Performance Metrics

### 2.1 Validation Split Evaluation (`val`, $N = 146$ images, 167 bboxes)

| Class ID | Component Name | Instances | Precision ($P$) | Recall ($R$) | mAP@0.5 | mAP@0.5:0.95 |
|:---:|---|:---:|:---:|:---:|:---:|:---:|
| `0` | `engine_bay` | 37 | **0.913** (91.3%) | **0.892** (89.2%) | **0.960** (96.0%) | **0.659** (65.9%) |
| `1` | `battery` | 103 | **0.891** (89.1%) | **0.716** (71.6%) | **0.788** (78.8%) | **0.457** (45.7%) |
| `2` | `radiator` | 27 | **0.391** (39.1%) | **0.259** (25.9%) | **0.270** (27.0%) | **0.116** (11.6%) |
| **ALL** | **Overall Macro** | **167** | **0.732** (73.2%) | **0.622** (62.2%) | **0.673** (67.3%) | **0.411** (41.1%) |

### 2.2 Unseen Test Split Evaluation (`test`, $N = 70$ images, 77 bboxes)

| Class ID | Component Name | Instances | Precision ($P$) | Recall ($R$) | mAP@0.5 | mAP@0.5:0.95 |
|:---:|---|:---:|:---:|:---:|:---:|:---:|
| `0` | `engine_bay` | 18 | **0.919** (91.9%) | **0.944** (94.4%) | **0.941** (94.1%) | **0.749** (74.9%) |
| `1` | `battery` | 50 | **0.892** (89.2%) | **0.660** (66.0%) | **0.786** (78.6%) | **0.385** (38.5%) |
| `2` | `radiator` | 9 | 0.000 (0.0%) | 0.000 (0.0%) | 0.039 (3.9%) | 0.009 (0.9%) |
| **ALL** | **Overall Macro** | **77** | **0.604** (60.4%) | **0.535** (53.5%) | **0.588** (58.8%) | **0.381** (38.1%) |

---

## 3. Detailed Component-Specific Diagnostics

### 3.1 Radiator Performance Disparity (Critical Finding)
As anticipated in the Stage 3.0 readiness audit, radiator detection exhibited lower performance:
- **Validation Split:** $\text{mAP}@0.5 = 0.270$, Precision $= 0.391$, Recall $= 0.259$.
- **Test Split:** $\text{mAP}@0.5 = 0.039$, Recall $= 0.000$ (all 9 test instances missed at confidence threshold 0.25).
- **Underlying Root Causes:**
  1. **Extreme Aspect Ratio:** Radiator cooling cores possess a mean aspect ratio $w/h \approx 8.34$ (very wide, flat horizontal strip across the front tie-bar). Standard YOLO anchor/feature grids downsample wide aspect ratios heavily.
  2. **Sample Scarcity:** Only 79 training instances and 9 test instances exist across the entire dataset. In contrast, `battery` had 312 training instances (4x larger).
  3. **Visual Obstruction:** Under standard engine bay lighting, radiator cores sit partially recessed beneath upper radiator support brackets and plastic shroud covers.
- **Remediation Plan for Stage 4:** Radiator requires dedicated workshop image collection with front tie-bar close-ups, class-specific focal loss weighting, and multi-scale training with high horizontal aspect ratio augmentations.

### 3.2 Battery Performance (Production Grade)
- **Validation:** $\text{mAP}@0.5 = 0.788$, Precision $= 0.891$, Recall $= 0.716$.
- **Test:** $\text{mAP}@0.5 = 0.786$, Precision $= 0.892$, Recall $= 0.660$.
- **Diagnostics:** The battery detector exhibits high precision ($\approx 89\%$). Missed detections (lower recall) occur primarily in dark, shadowed engine bay corners or when batteries are partially encased in insulating thermal jackets.

### 3.3 Engine Bay Performance (Exceptional Robustness)
- **Validation:** $\text{mAP}@0.5 = 0.960$, Precision $= 0.913$, Recall $= 0.892$.
- **Test:** $\text{mAP}@0.5 = 0.941$, Precision $= 0.919$, Recall $= 0.944$.
- **Diagnostics:** Engine bay detection achieves $\ge 94\%$ mAP on both splits. The network reliably identifies the complete perimeter of the open engine bay across sedans, hatchbacks, SUVs, and commercial vans.

---

## 4. Hard-Negative Evaluation & False-Positive Analysis

- **Evaluated Image:** `test/Ford-Ranger-XLT-4x4-3-2-Autoentusiastas-22_jpg.rf.55c7c78702e78433f5abd96c3a33e57f.jpg`
- **True Content:** Exterior 3/4 front view of a pickup truck with the hood completely **closed** (0 ground truth components).
- **Model Output:** Detected `engine_bay` with confidence **`0.901`** across the closed hood and grille region.
- **Diagnostic Conclusion:** The baseline model has learned strong correlations with the front hood/bumper area of vehicles, causing it to fire a high-confidence false positive when presented with a closed-hood vehicle front end.
- **Actionable Insight:** The training set currently consists almost exclusively of open-hood engine compartments. Adding closed-hood exterior vehicle images as explicit negative background samples during Stage 4 training will penalize and eliminate this false-positive mode.

---

## 5. Visual Inspection of Generated Test Predictions

Rendered images with bounding boxes, labels, and confidence scores are preserved in [`docs/stage3_1_visual_results/`](file:///D:/AI-Vehicle-Diagnosis-System/docs/stage3_1_visual_results/):

| Sample Reference | Target Scenario | Ground Truth Components | Model Detections | Correctness |
|---|---|---|---|:---:|
| [`visual_01_engine_bay_only.jpg`](file:///D:/AI-Vehicle-Diagnosis-System/docs/stage3_1_visual_results/visual_01_engine_bay_only.jpg) | Engine Bay Only | 1 `engine_bay` | `engine_bay` (conf: **0.838**) | **CORRECT** |
| [`visual_02_battery_only.jpg`](file:///D:/AI-Vehicle-Diagnosis-System/docs/stage3_1_visual_results/visual_02_battery_only.jpg) | Battery Only | 1 `battery` | `battery` (conf: **0.650**) | **CORRECT** |
| [`visual_03_radiator_only.jpg`](file:///D:/AI-Vehicle-Diagnosis-System/docs/stage3_1_visual_results/visual_03_radiator_only.jpg) | Radiator Only | 1 `radiator` | No detections ($> 0.25$) | **FALSE NEGATIVE** |
| [`visual_04_battery_and_radiator.jpg`](file:///D:/AI-Vehicle-Diagnosis-System/docs/stage3_1_visual_results/visual_04_battery_and_radiator.jpg) | Battery & Radiator | 1 `battery`, 1 `radiator` | No detections ($> 0.25$) | **FALSE NEGATIVE** |
| [`visual_05_multiple_components.jpg`](file:///D:/AI-Vehicle-Diagnosis-System/docs/stage3_1_visual_results/visual_05_multiple_components.jpg) | Multiple Components | 1 `battery`, 1 `radiator` | `battery` (conf: **0.805**) | **PARTIAL** |
| [`visual_06_hard_negative.jpg`](file:///D:/AI-Vehicle-Diagnosis-System/docs/stage3_1_visual_results/visual_06_hard_negative.jpg) | Hard Negative Background | 0 components (closed hood) | `engine_bay` (conf: **0.901**) | **FALSE POSITIVE** |

---

## 6. Generalization Limits & Scientific Disclaimers

> [!CAUTION]
> **Limitations of Baseline Evaluation:**
> 1. The test partition comprises 70 images and exactly 9 radiator instances. Strong claims of general open-world reliability cannot be substantiated from this small test set alone.
> 2. The radiator class requires significantly more real-world workshop samples and targeted aspect-ratio tuning before production reliance.
> 3. False-positive resistance on closed-hood exterior images requires training set augmentation with negative exterior scenes.

---

## 7. Model Inference Latency Benchmark

Measured on NVIDIA GeForce RTX 3050 Laptop GPU (4 GB VRAM) using FP16 mixed precision at $640 \times 640$:
- **Pre-processing Latency:** $\approx 0.8\text{ ms}$
- **Tensor Core Inference Latency:** $\approx 11.9\text{ ms}$
- **Post-processing (NMS) Latency:** $\approx 3.4\text{ ms}$
- **Total Pipeline Latency:** **$\approx 16.1\text{ ms}$ per image** ($\approx 62\text{ frames per second}$)

The model easily satisfies the real-time inference latency requirement ($\le 50\text{ ms}$) for the diagnostic API.
