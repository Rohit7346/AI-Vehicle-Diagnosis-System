"""
Model Evaluation Harness.
Runs comprehensive benchmarking on held-out test splits.
Strictly requires genuine trained model weights and real annotated test sets.
"""

from pathlib import Path
from typing import Dict, Optional
import sys

from ml.models.base_model import ModelWeightsNotFoundError


class ModelEvaluator:
    """Evaluates component or fault detector checkpoints on benchmark datasets."""

    def __init__(self, weights_path: Path, data_yaml_path: Path):
        self.weights_path = Path(weights_path)
        self.data_yaml_path = Path(data_yaml_path)

    def evaluate(self, split: str = "test", device: str = "cpu") -> Dict[str, object]:
        """
        Executes validation on the given split using real YOLO metrics.
        Raises ModelWeightsNotFoundError if weights are not on disk.
        """
        if not self.weights_path.exists():
            raise ModelWeightsNotFoundError(
                f"Cannot evaluate model: Weights checkpoint not found at '{self.weights_path}'. "
                f"Train the model first or place legitimate weights in ml/weights/."
            )

        if not self.data_yaml_path.exists():
            raise FileNotFoundError(f"Dataset config not found at: {self.data_yaml_path}")

        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("Ultralytics is required for evaluation. Run: pip install ultralytics")

        print(f"[EVALUATION] Loading checkpoint: {self.weights_path}")
        model = YOLO(str(self.weights_path))

        print(f"[EVALUATION] Running validation on '{split}' split...")
        metrics = model.val(
            data=str(self.data_yaml_path),
            split=split,
            device=device,
            verbose=False,
        )

        results = {
            "weights_path": str(self.weights_path),
            "split": split,
            "mAP50": round(float(metrics.box.map50), 4),
            "mAP50_95": round(float(metrics.box.map), 4),
            "precision": round(float(metrics.box.mp), 4),
            "recall": round(float(metrics.box.mr), 4),
            "class_indices": metrics.box.ap_class_index.tolist() if hasattr(metrics.box, "ap_class_index") else [],
        }
        return results


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python -m ml.evaluation.evaluator <weights_path> <data_yaml_path> [split]")
        sys.exit(1)

    weights = Path(sys.argv[1])
    yaml_cfg = Path(sys.argv[2])
    test_split = sys.argv[3] if len(sys.argv) > 3 else "test"

    try:
        evaluator = ModelEvaluator(weights, yaml_cfg)
        metrics_out = evaluator.evaluate(test_split)
        print("\n=== Benchmark Results ===")
        for k, v in metrics_out.items():
            print(f"  {k}: {v}")
    except Exception as exc:
        print(f"[ERROR] Evaluation halted: {exc}", file=sys.stderr)
        sys.exit(1)
