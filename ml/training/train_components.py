"""
Component Detector Training Pipeline.
Enforces pre-flight dataset verification, Ultralytics YOLO dataset YAML passing,
hyperparameter isolation, and post-training checkpoint synchronization.
"""

import argparse
from pathlib import Path
import shutil
import sys
from typing import Dict, Optional
import yaml

from ml.datasets.annotator_utils import AnnotationValidator, format_summary_table


def verify_dataset_readiness(
    dataset_yaml_path: Path,
    min_samples_per_split: int = 10,
) -> bool:
    """
    Validates dataset existence, split separation, annotation bounds, and data leakage.
    Strictly halts if datasets are empty, corrupted, or unverified.
    """
    print("=" * 70)
    print(f"[PRE-FLIGHT] Verifying Dataset Specification: {dataset_yaml_path}")
    print("=" * 70)

    if not dataset_yaml_path.exists():
        print(f"[ERROR] Dataset configuration file not found at: {dataset_yaml_path}", file=sys.stderr)
        return False

    with open(dataset_yaml_path, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    # Resolve base data directory
    raw_path = data_cfg.get("path", "../../data")
    if Path(raw_path).is_absolute():
        base_data_dir = Path(raw_path)
    else:
        base_data_dir = (dataset_yaml_path.parent / raw_path).resolve()

    expected_classes = len(data_cfg.get("names", {}))
    audit = AnnotationValidator.audit_full_dataset(base_data_dir, expected_classes=expected_classes)

    # Print structured dataset audit table
    print(format_summary_table(audit))
    print("-" * 70)

    train_audit = audit["splits"].get("train", {})
    val_audit = audit["splits"].get("val", {})
    leakage = audit.get("leakage", {})

    # 1. Image count check
    if train_audit.get("total_images", 0) < min_samples_per_split:
        print(
            f"[ABORT] Training split has only {train_audit.get('total_images', 0)} images. "
            f"Minimum required is {min_samples_per_split} annotated samples. "
            f"Training cannot begin on empty or unpopulated splits.",
            file=sys.stderr,
        )
        return False

    if val_audit.get("total_images", 0) == 0:
        print("[ABORT] Validation split has 0 images. Validation set is strictly required.", file=sys.stderr)
        return False

    # 2. Syntax or coordinate boundary errors
    if not train_audit.get("valid") or not val_audit.get("valid"):
        print("[ABORT] Annotation syntax or bounding-box boundary errors detected.", file=sys.stderr)
        return False

    # 3. Data leakage check
    if leakage.get("has_leakage"):
        print("[ABORT] Critical data leakage detected between dataset splits!", file=sys.stderr)
        return False

    # 4. Class representation check
    if audit.get("missing_in_train"):
        print(
            f"[WARNING] Classes missing in training split: {audit['missing_in_train']}. "
            f"Every target class must have representative training samples.",
            file=sys.stderr,
        )
        return False

    print("[SUCCESS] Dataset passed all pre-flight verification checks.")
    return True


def run_training(
    dataset_yaml_path: Path,
    hyp_yaml_path: Path,
    model_override: Optional[str] = None,
    epochs_override: Optional[int] = None,
    batch_override: Optional[int] = None,
    device_override: Optional[str] = None,
    sync_to_inference: bool = True,
):
    """
    Executes YOLOv8 component detection training pipeline.
    Passes pure dataset YAML to Ultralytics and loads hyperparameters from separate config.
    """
    # 1. Enforce strict pre-flight gatekeeper
    if not verify_dataset_readiness(dataset_yaml_path):
        print("\n[TERMINATED] Training aborted by pre-flight dataset gatekeeper.", file=sys.stderr)
        print("             Please populate and validate datasets before launching training.", file=sys.stderr)
        sys.exit(1)

    # 2. Load hyperparameters from separate config
    if not hyp_yaml_path.exists():
        print(f"[ERROR] Hyperparameters configuration file not found at: {hyp_yaml_path}", file=sys.stderr)
        sys.exit(1)

    with open(hyp_yaml_path, "r", encoding="utf-8") as f:
        hyp = yaml.safe_load(f)

    # Apply optional CLI overrides
    target_model = model_override or hyp.get("model", "yolov8n.pt")
    target_epochs = epochs_override if epochs_override is not None else hyp.get("epochs", 100)
    target_batch = batch_override if batch_override is not None else hyp.get("batch", 16)
    target_device = device_override if device_override is not None else hyp.get("device", "0")
    imgsz = hyp.get("imgsz", 640)
    project = hyp.get("project", "runs/detect")
    name = hyp.get("name", "component_yolov8n")
    workers = hyp.get("workers", 2)
    amp = hyp.get("amp", True)
    save = hyp.get("save", True)

    print("\n" + "=" * 70)
    print(" Executing Stage 1 Training: Vehicle Component Detection")
    print("=" * 70)
    print(f" Dataset YAML      : {dataset_yaml_path.resolve()}")
    print(f" Hyperparameters   : {hyp_yaml_path.resolve()}")
    print(f" Model Architecture: {target_model}")
    print(f" Epochs            : {target_epochs}")
    print(f" Batch Size        : {target_batch}")
    print(f" Compute Device    : {target_device}")
    print(f" Automatic Mixed P.: {amp}")
    print(f" Output Location   : {project}/{name}/")
    print("=" * 70)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] Ultralytics is required. Install: pip install ultralytics", file=sys.stderr)
        sys.exit(1)

    # Initialize model from pretrained weights or architecture definition
    model = YOLO(target_model)

    # Execute training via Ultralytics API
    results = model.train(
        data=str(dataset_yaml_path.resolve()),
        epochs=target_epochs,
        batch=target_batch,
        imgsz=imgsz,
        device=target_device,
        workers=workers,
        amp=amp,
        project=project,
        name=name,
        save=save,
        plots=hyp.get("plots", True),
        optimizer=hyp.get("optimizer", "AdamW"),
        lr0=hyp.get("lr0", 0.001),
        patience=hyp.get("patience", 20),
    )

    print("\n[SUCCESS] Training loop completed.")

    # 3. Checkpoint verification & synchronization
    output_weights_dir = Path(project) / name / "weights"
    best_checkpoint = output_weights_dir / "best.pt"
    last_checkpoint = output_weights_dir / "last.pt"

    if best_checkpoint.exists():
        size_mb = round(best_checkpoint.stat().st_size / (1024 * 1024), 2)
        print(f"[CHECKPOINT] Best weights verified : {best_checkpoint} ({size_mb} MB)")
        print(f"[CHECKPOINT] Last weights verified : {last_checkpoint}")

        if sync_to_inference:
            target_dest = Path("ml/weights/component_detector_best.pt")
            target_dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(best_checkpoint, target_dest)
            print(f"[SYNC] Copied best weights to inference path: {target_dest.resolve()}")
            print("[SYNC] Backend API and React Dashboard will now report Model Status: READY.")
    else:
        print(f"[WARNING] Expected checkpoint not found at: {best_checkpoint}", file=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Vehicle Component Detector (YOLOv8)")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("ml/config/components_dataset.yaml"),
        help="Path to Ultralytics dataset specification YAML",
    )
    parser.add_argument(
        "--hyp",
        type=Path,
        default=Path("ml/config/hyperparameters_yolov8n.yaml"),
        help="Path to hyperparameters configuration YAML",
    )
    parser.add_argument("--model", type=str, default=None, help="Override YOLO model checkpoint (e.g. yolov8n.pt)")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs")
    parser.add_argument("--batch", type=int, default=None, help="Override batch size")
    parser.add_argument("--device", type=str, default=None, help="Override device ('0' for GPU, 'cpu' for CPU)")
    parser.add_argument("--smoke-test", action="store_true", help="Run 3-epoch smoke-test using smoke_test_hyperparameters.yaml")
    parser.add_argument("--no-sync", action="store_true", help="Do not copy best.pt to ml/weights/")
    args = parser.parse_args()

    hyp_file = Path("ml/config/smoke_test_hyperparameters.yaml") if args.smoke_test else args.hyp

    run_training(
        dataset_yaml_path=args.data,
        hyp_yaml_path=hyp_file,
        model_override=args.model,
        epochs_override=args.epochs,
        batch_override=args.batch,
        device_override=args.device,
        sync_to_inference=not args.no_sync,
    )
