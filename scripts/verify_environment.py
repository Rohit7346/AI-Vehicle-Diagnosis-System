"""
Environment & Monorepo Health Verification Script.
Checks python environment, dependencies, GPU availability, and directory contracts.
"""

import sys
from pathlib import Path

# Ensure UTF-8 console output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def check_environment():
    print("=" * 70)
    print(" AI Multimodal Vehicle Diagnosis System: Environment Audit")
    print("=" * 70)

    # 1. Python version check
    py_ver = sys.version.split()[0]
    print(f"[1] Python Runtime: {py_ver}")

    # 2. Key Library Verification
    libs = [
        ("FastAPI", "fastapi"),
        ("Uvicorn", "uvicorn"),
        ("Pydantic", "pydantic"),
        ("Pillow", "PIL"),
        ("NumPy", "numpy"),
        ("PyYAML", "yaml"),
        ("PyTorch", "torch"),
        ("Ultralytics YOLO", "ultralytics"),
        ("OpenCV", "cv2"),
    ]

    print("\n[2] Checking Dependency Installations:")
    for label, pkg in libs:
        try:
            mod = __import__(pkg)
            version = getattr(mod, "__version__", "installed")
            print(f"    [OK] {label:<20}: {version}")
        except ImportError:
            print(f"    [--] {label:<20}: Not installed (Optional or pending pip install)")

    # 3. Hardware acceleration
    print("\n[3] Hardware Acceleration:")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"    [OK] CUDA Acceleration Active: {torch.cuda.get_device_name(0)}")
        else:
            print("    [--] CUDA not available: Falling back to CPU compute.")
    except ImportError:
        print("    [--] PyTorch not loaded.")

    # 4. Directory Structure Check
    print("\n[4] Monorepo Directory Integrity:")
    dirs = [
        "data/raw/images",
        "data/raw/audio",
        "data/processed/train",
        "data/processed/val",
        "data/processed/test",
        "data/annotations/train",
        "data/annotations/val",
        "data/annotations/test",
        "docs",
        "ml/config",
        "ml/datasets",
        "ml/models",
        "ml/training",
        "ml/evaluation",
        "ml/inference",
        "ml/weights",
        "backend/app",
        "frontend/src",
    ]
    for d in dirs:
        p = Path(d)
        if p.exists() and p.is_dir():
            print(f"    [OK] {d}")
        else:
            print(f"    [!!] {d} (Missing!)")

    print("\n" + "=" * 70)
    print(" Audit Completed Successfully.")
    print("=" * 70)


if __name__ == "__main__":
    check_environment()
