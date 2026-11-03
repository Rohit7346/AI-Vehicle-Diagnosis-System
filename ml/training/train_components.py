"""
Component Detector Training Pipeline.
Enforces pre-flight dataset verification and annotation compliance before training.
Optimized for modular YOLO model selection and GPU acceleration.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional
import yaml

from ml.datasets.annotator_utils import AnnotationValidator


def verify_dataset_readiness(config_path: Path, min_samples: int = 10) -> bool:
    """
    Validates dataset existence, splits, and YOLO annotation integrity.
    Prevents launching training on empty or unverified directories.
    """
    print(f"[PRE-FLIGHT] Inspecting dataset configuration: {config_path}")
    if not config_path.exists():
        print(f"[ERROR] Config file not found: {config_path}", file=sys.stderr)
        return False

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    base_dir = config_path.parent.parent.parent  # Project root
    data_dir = (base_dir / "data").resolve()
    
    # Check both direct split folders and nested images/labels layouts
    train_images = data_dir / "processed" / "train"
    train_labels = data_dir / "annotations" / "train"
    val_images = data_dir / "processed" / "val"
    val_labels = data_dir / "annotations" / "val"

    expected_classes = len(cfg.get("names", {}))

    print(f"[PRE-FLIGHT] Verifying training split in: {train_images}")
    train_audit = AnnotationValidator.validate_split(train_images, train_labels, expected_classes)
    val_audit = AnnotationValidator.validate_split(val_images, val_labels, expected_classes)

    if not train_audit.get("valid"):
        print(f"[ERROR] Training dataset verification failed!", file=sys.stderr)
        print(f"        Total images found: {train_audit.get('total_images', 0)}", file=sys.stderr)
        print(f"        Total labels found: {train_audit.get('total_labels', 0)}", file=sys.stderr)
        if train_audit.get("missing_labels"):
            print(f"        Missing labels: {len(train_audit['missing_labels'])} files", file=sys.stderr)
        if train_audit.get("syntax_errors"):
            print(f"        Syntax errors: {train_audit['syntax_errors'][:5]}", file=sys.stderr)
        return False

    if train_audit.get("total_images", 0) < min_samples:
        print(
            f"[ABORT] Training split has only {train_audit.get('total_images')} images. "
            f"Minimum requirement is {min_samples} annotated samples. "
            f"Please populate 'data/raw/images/', run preprocessor, and annotate.",
            file=sys.stderr,
        )
        return False

    print(f"[SUCCESS] Dataset verified: {train_audit['total_images']} train samples, {val_audit.get('total_images', 0)} val samples.")
    return True


def run_training(
    config_path: Path,
    model_name: Optional[str] = None,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    device: Optional[str] = None,
    imgsz: int = 640,
):
    """Executes YOLO component detection training pipeline."""
    if not verify_dataset_readiness(config_path):
        print("[TERMINATED] Training aborted due to dataset validation failure.", file=sys.stderr)
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Resolve parameters: CLI args override YAML config, which overrides defaults
    target_model = model_name or cfg.get("model_name", "yolov8n.pt")
    target_epochs = epochs if epochs is not None else cfg.get("epochs", 100)
    target_batch = batch_size if batch_size is not None else cfg.get("batch_size", 8)
    target_device = device if device is not None else cfg.get("device", "0")

    print("[TRAINING] Initializing Ultralytics YOLO architecture...")
    print(f"           Model Architecture: {target_model}")
    print(f"           Epochs: {target_epochs}")
    print(f"           Batch Size: {target_batch}")
    print(f"           Compute Target: {target_device}")

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] Ultralytics not installed. Run: pip install ultralytics", file=sys.stderr)
        sys.exit(1)

    model = YOLO(target_model)
    model.train(
        data=str(config_path),
        epochs=target_epochs,
        batch=target_batch,
        imgsz=imgsz,
        device=target_device,
        project="runs/detect",
        name="component_detector",
        save=True,
    )
    print("[SUCCESS] Training pipeline completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Vehicle Component Detector")
    parser.add_argument("--config", type=Path, default=Path("ml/config/component_config.yaml"), help="Path to dataset config YAML")
    parser.add_argument("--model", type=str, default=None, help="YOLO model checkpoint or weights (e.g. yolov8n.pt, yolov8s.pt)")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size (recommended: 8 or 16 for 4GB RTX 3050)")
    parser.add_argument("--device", type=str, default=None, help="Compute device ('0' for GPU, 'cpu' for CPU)")
    parser.add_argument("--smoke-test", action="store_true", help="Use smoke-test configuration (3 epochs, yolov8n, batch 8)")
    args = parser.parse_args()

    cfg_file = Path("ml/config/smoke_test_components.yaml") if args.smoke_test else args.config
    run_training(
        config_path=cfg_file,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
    )
