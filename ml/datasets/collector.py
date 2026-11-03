"""
Dataset Collector & Ingestion Validator.
Enforces data collection protocols, camera quality standards, and file validation.
"""

from pathlib import Path
from typing import Dict, List, Tuple
from PIL import Image


class DatasetCollector:
    """Manages raw collection directories and enforces academic data collection criteria."""

    SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
    MIN_RESOLUTION = (800, 600)

    def __init__(self, base_data_dir: Path = Path("data")):
        self.base_data_dir = base_data_dir
        self.raw_dir = base_data_dir / "raw" / "images"
        self.processed_dir = base_data_dir / "processed"
        self.annotations_dir = base_data_dir / "annotations"

    def audit_raw_data(self) -> Dict[str, object]:
        """
        Scans raw image collections and validates file formats, dimensions, and readability.
        Does NOT download arbitrary datasets.
        """
        if not self.raw_dir.exists():
            return {
                "exists": False,
                "total_images": 0,
                "valid_images": 0,
                "corrupt_images": [],
                "substandard_resolution": [],
            }

        image_files = [
            f for f in self.raw_dir.iterdir()
            if f.suffix.lower() in self.SUPPORTED_EXTENSIONS
        ]

        valid_count = 0
        corrupt_files = []
        low_res_files = []

        for img_path in image_files:
            try:
                with Image.open(img_path) as img:
                    img.verify()
                # Re-open after verify() to inspect dimensions
                with Image.open(img_path) as img:
                    width, height = img.size
                    if width < self.MIN_RESOLUTION[0] or height < self.MIN_RESOLUTION[1]:
                        low_res_files.append((str(img_path.name), (width, height)))
                    else:
                        valid_count += 1
            except Exception as e:
                corrupt_files.append((str(img_path.name), str(e)))

        return {
            "exists": True,
            "raw_dir": str(self.raw_dir),
            "total_images": len(image_files),
            "valid_images": valid_count,
            "corrupt_images": corrupt_files,
            "substandard_resolution": low_res_files,
        }
