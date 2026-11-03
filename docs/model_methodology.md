# AI & Model Methodology: Dual-Stage Multimodal Perception

## 1. Mathematical Formulation

The diagnostic problem is formulated as a conditional spatial detection pipeline.

Given an input RGB inspection image $I \in \mathbb{R}^{H \times W \times 3}$:

### Step 1: Subassembly Localization
A component detector parameterized by weights $\theta_C$ predicts a set of component instances:
$$\mathcal{C} = \{ (b_i^c, s_i^c, l_i^c) \}_{i=1}^{N_c}$$
Where:
- $b_i^c = (x_1, y_1, x_2, y_2)$ represents the bounding coordinates.
- $s_i^c \in [0, 1]$ represents component detection confidence.
- $l_i^c \in \mathcal{L}_{components}$ is the classified vehicle component label.

### Step 2: Fault & Defect Identification
A concurrent or conditional fault detector parameterized by $\theta_F$ predicts physical anomalies:
$$\mathcal{F} = \{ (b_j^f, s_j^f, l_j^f) \}_{j=1}^{N_f}$$
Where $l_j^f \in \mathcal{L}_{faults}$.

### Step 3: Spatial Correlation & Diagnosis Synthesis
For each detected fault $f_j$, its spatial intersection over component area (IoA) is evaluated:
$$\text{IoA}(f_j, c_i) = \frac{\text{Area}(b_j^f \cap b_i^c)}{\text{Area}(b_j^f)}$$

If $\text{IoA}(f_j, c_i) \ge \tau_{match}$ (default: $0.50$), fault $f_j$ is attributed directly to component $c_i$, formulating a structured diagnostic link:
$$\text{DiagnosticUnit}_k = \langle c_i, f_j, \text{Severity}(l_j^f) \rangle$$

---

## 2. Evaluation Metrics

To avoid arbitrary or subjective evaluation, all trained checkpoints must report standard metric benchmarks:

1. **Mean Average Precision (mAP@50 and mAP@50:95):**
   $$\text{AP} = \int_{0}^{1} p(r) \, dr$$
2. **Precision & Recall Curves:**
   $$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}, \quad \text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$
3. **Confusion Matrix:**
   Normalized across actual target classes to identify cross-class leakage (e.g., distinguishing oil leaks from coolant seepage).

---

## 3. Training & Validation Safeguards
- Zero mock inference: The pipeline refuses to emit predictions if weights are absent.
- Dataset integrity gate: Training scripts halt execution if:
  - The dataset YAML configuration is missing.
  - Fewer than a user-configured minimal number of images (default: 20 per split) are found.
  - Coordinate format violations exist in annotations (values outside $[0, 1]$).
