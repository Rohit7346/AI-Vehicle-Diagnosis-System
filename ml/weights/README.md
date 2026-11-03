# Model Weights Directory

This directory stores trained model checkpoints (`.pt`, `.onnx`, `.engine`) for:
1. **Component Detector**: `component_detector_best.pt`
2. **Fault Detector**: `fault_detector_best.pt`

## Checkpoint Rules
- Weights are strictly excluded from version control by `.gitignore` (large binary files).
- The inference engine checks for weights existence before attempting to run predictions.
- To produce weights:
  1. Prepare verified datasets in `data/processed/` and `data/annotations/`.
  2. Run `python -m ml.training.train_components` and `python -m ml.training.train_faults`.
  3. Copy best checkpoint from `runs/detect/.../weights/best.pt` to this folder.
