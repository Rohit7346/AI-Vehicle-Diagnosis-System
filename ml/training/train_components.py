"""
Component Detector Training Pipeline.
Enforces pre-flight dataset verification and annotation compliance before training.
"""

import argparse
import sys
from pathlib import Path
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


def run_training(config_path: Path, epochs: int = 100, batch_size: int = 16, device: str = "cpu"):
    """Executes YOLO component detection training pipeline."""
    if not verify_dataset_readiness(config_path):
        print("[TERMINATED] Training aborted due to dataset validation failure.", file=sys.stderr)
        sys.exit(1)

    print("[TRAINING] Initializing Ultralytics YOLO architecture...")
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] Ultralytics not installed. Run: pip install ultralytics", file=sys.stderr)
        sys.exit(1)

    model = YOLO("yolov8m.pt")
    model.train(
        data=str(config_path),
        epochs=epochs,
        batch=batch_size,
        imgsz=640,
        device=device,
        project="runs/detect",
        name="component_detector",
        save=True,
    )
    print("[SUCCESS] Training pipeline completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Vehicle Component Detector")
    parser.add_argument("--config", type=Path, default=Path("ml/config/component_config.yaml"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    run_training(args.config, args.epochs, args.batch_size, args.device)
