# Stage 3.0: Training Readiness Audit & Final GO / NO-GO Assessment

**Project Title:** AI-Powered Multimodal Vehicle Fault Detection and Diagnosis System  
**Stage:** Stage 3.0 – Three-Class Dataset Integration and Final Training Readiness Audit  
**Target Milestone:** Stage 3.1 – YOLOv8 Nano 3-Class Model Training  
**Audit Target:** [`data/processed/stage3_0_three_class/`](file:///D:/AI-Vehicle-Diagnosis-System/data/processed/stage3_0_three_class/)  
**Integration Report:** [`docs/stage3_0_three_class_integration_report.md`](file:///D:/AI-Vehicle-Diagnosis-System/docs/stage3_0_three_class_integration_report.md)  
**Distribution Registry:** [`docs/stage3_0_class_distribution.csv`](file:///D:/AI-Vehicle-Diagnosis-System/docs/stage3_0_class_distribution.csv)  
**Duplicate Registry:** [`docs/stage3_0_duplicate_resolution.csv`](file:///D:/AI-Vehicle-Diagnosis-System/docs/stage3_0_duplicate_resolution.csv)  
**Automated Tests:** [`tests/test_three_class_integration.py`](file:///D:/AI-Vehicle-Diagnosis-System/tests/test_three_class_integration.py) (10/10 PASS)  
**Date:** November 11, 2026  

---

## 1. Executive Summary & Final Verdict

This audit assesses the physical, structural, semantic, and algorithmic readiness of the unified three-class dataset ([`data/processed/stage3_0_three_class/`](file:///D:/AI-Vehicle-Diagnosis-System/data/processed/stage3_0_three_class/)) before launching Stage 3.1 model training.

The dataset integrates three rigorously audited and cleaned source streams:
1. **Engine Bay (Class 0):** Stage 2.6 Roboflow export (191 annotations + 1 hard negative).
2. **Battery (Class 1):** Stage 2.5.1 curated automotive batteries (465 annotations).
3. **Radiator (Class 2):** Stage 2.7.1 curated radiator cooling assemblies (115 annotations).

### Formal Verdict: **GO** (Full Approval for Stage 3.1 Training)

| Assessment Dimension | Status | Key Justification |
|---|:---:|---|
| **Data Integrity & Pairing** | **SATISFIED** | 696 images, 696 label files; 100% paired with identical stems; 0 corrupted files. |
| **Annotation Format** | **SATISFIED** | 100% standard 5-token YOLO bounding boxes $(x_c, y_c, w, h)$; all coordinates in $[0.0, 1.0]$. |
| **Data Leakage & Splitting** | **SATISFIED** | Exhaustive $dHash$ audit ($N = 113,900$ cross-split pairs) verified **0 cross-split leakage pairs ($D_H \le 4$)**. |
| **Semantic Validity** | **SATISFIED** | Cap-only mislabeling in radiator source purged (100% Source 1 caps excluded); only genuine cores retained. |
| **Multi-Class Co-occurrence** | **SATISFIED** | 76 multi-class images harmonized into unified label files without overwrite or duplication. |
| **Automated Verification** | **SATISFIED** | 10 of 10 automated integration tests passed; full test suite passes (64/64 tests). |
| **Hardware & Environment** | **SATISFIED** | Ultralytics YOLOv8 installed in `.venv`; Windows long-path safety verified. |

---

## 2. Quantitative Volume & Class Distribution Analysis

### 2.1 Dataset Volume Summary

```
Total Unique Physical Images: 696
Total Bounding Box Objects:   771
Label Files:                  696 (695 annotated, 1 hard negative background)
```

| Split | Images | Image % | Total BBoxes | Engine Bay (0) | Battery (1) | Radiator (2) | Multi-Class Imgs | Multi-Box Imgs |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **`train`** | **480** | 69.0% | **527** | 136 | 312 | 79 | 47 | 47 |
| **`val`** | **146** | 21.0% | **167** | 37 | 103 | 27 | 21 | 21 |
| **`test`** | **70** | 10.1% | **77** | 18 | 50 | 9 | 8 | 8 |
| **TOTAL** | **696** | **100.0%** | **771** | **191** | **465** | **115** | **76** | **76** |

### 2.2 Class Balance Assessment & Training Risk Mitigation

The bounding-box count ratio across classes is approximately:
$$\text{battery } (465) : \text{engine\_bay } (191) : \text{radiator } (115) \approx 4.04 : 1.66 : 1.00$$

- **Battery (Class 1 - 465 bboxes):** Robust sample size with high intra-class variation (top-post, side-post, battery blankets, different engine bay quadrants).
- **Engine Bay (Class 0 - 191 bboxes):** Excellent macro-level representation across 191 diverse vehicle makes/models. High spatial consistency ($AR \approx 1.0 - 1.5$, large box area $> 0.50$).
- **Radiator (Class 2 - 115 bboxes):** Lowest relative count, but highly consistent morphology (wide rectangular aspect ratio across the front slam panel, mean $AR \approx 8.34$).
  
**Training Recommendations for Class Balance:**
1. **Focal Loss / CIoU Loss:** Standard Ultralytics YOLOv8 loss functions inherently down-weight easy background examples and balance multi-class detection via task-aligned assigner.
2. **Multi-Scale Augmentations:** Mosaic ($p = 1.0$), horizontal flips ($p = 0.5$), and scale jitter ($p = 0.5$) will synthetically expose the network to diverse radiator perspectives.
3. **No Artificial Oversampling Required:** Because 76 images contain co-occurring battery and radiator objects, the model will naturally learn contextual spatial relationships (radiator at front perimeter, battery in left/right quarter panel).

---

## 3. Image Resolution, Geometry & Aspect Ratio Profiles

Understanding the physical resolution and geometry across source datasets is essential for selecting optimal training parameters:

| Component Class | Mean BBox Area | Mean Aspect Ratio ($w/h$) | Native Image Resolutions | Geometric Characteristic |
|---|:---:|:---:|---|---|
| **`engine_bay`** | $\approx 0.620$ | $\approx 1.25$ | $512 \times 512$ (Roboflow square) | Macro bounding container covering $>60\%$ of frame |
| **`battery`** | $\approx 0.058$ | $\approx 1.15$ | $640 \times 640$ to $3662 \times 2122$ | Compact rectangular block, upper/side terminals |
| **`radiator`** | $\approx 0.045$ | $\approx 8.34$ | $640 \times 640$ to $3662 \times 2122$ | Very wide horizontal band along front upper tie-bar |

### Recommended Training Resolution: `--imgsz 640`
- Roboflow square images ($512 \times 512$) will undergo mild bilinear upsampling with automatic padding.
- High-resolution images ($> 1000\text{px}$) will be downsampled with letterbox padding, maintaining the distinctive $8.34$ aspect ratio of radiator cores without non-uniform stretching.
- 640px input resolution provides optimal trade-off between inference speed and feature preservation for the YOLOv8 Nano ($3.2\text{M}$ parameters) architecture.

---

## 4. Hard Negative & False-Positive Mitigation

- **Negative Sample:** Exactly 1 documented hard negative image (`Ford-Ranger-XLT-4x4-3-2-Autoentusiastas-22_jpg.rf.961a87612f0e300ceab9b386ee3aa6ee.jpg`) resides in `test/images/` paired with an empty 0-byte label file in `test/labels/`.
- **Purpose:** Ultralytics YOLOv8 utilizes empty label files as true background samples during evaluation to penalize hallucinated false positives (e.g., predicting an engine bay when looking at an exterior door or rear tailgate).
- **Test Set Purity:** The negative sample is restricted strictly to the test set, preventing spurious gradient noise during training while providing an objective evaluation benchmark.

---

## 5. Data Leakage & Independence Verification

Data leakage between train, validation, and test splits severely inflates validation metrics and causes production model failure. Stage 3.0 enforced rigorous protections:

1. **Exact Duplicate Prevention:**
   - 0 duplicate SHA-256 hashes exist across the entire 696-image collection.
2. **Perceptual Near-Duplicate Clustering:**
   - 29 near-duplicate situations and split realignments were audited and cataloged in [`docs/stage3_0_duplicate_resolution.csv`](file:///D:/AI-Vehicle-Diagnosis-System/docs/stage3_0_duplicate_resolution.csv).
   - Near-duplicate pairs (e.g., Chevrolet Spin `RES-SPIN-01`, cluster `RES-CLUSTER-01`) were co-located into `train`.
   - Source 2 split discrepancies were aligned to unified evaluation splits (`RES-REASSIGN-02` through `RES-REASSIGN-23`).
3. **Global Perceptual Verification:**
   - Evaluated 113,900 cross-split image pairs using difference hashing ($dHash$).
   - Minimum cross-split Hamming distance observed: $D_H \ge 5$.
   - **Zero cross-split leakage confirmed.**

---

## 6. Training Configuration & Baseline Specification for Stage 3.1

The unified dataset is ready for baseline execution in Stage 3.1. Below is the approved training specification:

### 6.1 Training Command Line Template
```powershell
& "D:\AI-Vehicle-Diagnosis-System\.venv\Scripts\python.exe" -m ultralytics train `
    model=yolov8n.pt `
    data="D:/AI-Vehicle-Diagnosis-System/data/processed/stage3_0_three_class/data.yaml" `
    epochs=100 `
    patience=20 `
    batch=16 `
    imgsz=640 `
    project="D:/AI-Vehicle-Diagnosis-System/models/runs/detect" `
    name="stage3_1_yolov8n_3class_baseline" `
    save=True `
    device=0 `
    workers=4 `
    seed=42 `
    plots=True
```

### 6.2 Target Success Metrics for Stage 3.1
- **Overall mAP@0.5:** $\ge 0.70$
- **Overall mAP@0.5:0.95:** $\ge 0.50$
- **Engine Bay (Class 0) mAP@0.5:** $\ge 0.85$ (distinct macro structure)
- **Battery (Class 1) mAP@0.5:** $\ge 0.75$ (well-represented component)
- **Radiator (Class 2) mAP@0.5:** $\ge 0.55$ (lower count, wide aspect ratio)
- **Inference Latency:** $\le 15\text{ ms}$ on standard GPU / $\le 80\text{ ms}$ on CPU (meets real-time diagnostics requirement).

---

## 7. Sign-off Matrix & Audit Conclusion

| Audit Item | Auditor / Script | Status | Notes |
|---|---|:---:|---|
| Dataset Directory Layout | `tests/test_three_class_integration.py::test_dataset_directory_structure_and_counts` | **APPROVED** | Verified 480 train / 146 val / 70 test |
| Label Coordinate Normalization | `tests/test_three_class_integration.py::test_normalized_coordinate_boundaries` | **APPROVED** | 100% within $[0.0, 1.0]$, 0 empty/NaN/Inf |
| Multi-Class Merging Integrity | `tests/test_three_class_integration.py::test_multi_class_and_multi_box_images` | **APPROVED** | 76 multi-class images correctly formatted |
| Cross-Split Leakage Freedom | `tests/test_three_class_integration.py::test_zero_cross_split_data_leakage` | **APPROVED** | 0 pairs with $D_H \le 4$ across 113,900 checks |
| YAML Ultralytics Compatibility | `tests/test_three_class_integration.py::test_data_yaml_correctness` | **APPROVED** | Matches Ultralytics YOLO schema |
| Source Immutability | `tests/test_three_class_integration.py::test_source_data_preservation` | **APPROVED** | Raw ZIPs, prototype, and cleaned sets untouched |

### Final Decision: **GO**
The dataset meets all technical, statistical, and architectural standards for production-grade object detection training. Stage 3.1 YOLOv8 Nano model training is authorized to commence immediately.
