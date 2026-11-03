"""
Annotation format validation and conversion utilities for vehicle diagnostics.
"""

from pathlib import Path
from typing import Dict, List, Tuple


class AnnotationValidator:
    """Verifies label files against YOLO annotation contracts."""

    @staticmethod
    def validate_split(
        images_dir: Path,
        labels_dir: Path,
        expected_classes: int,
    ) -> Dict[str, object]:
        """
        Validates that all images in images_dir have matching valid annotations in labels_dir.
        """
        if not images_dir.exists():
            return {"valid": False, "error": f"Image directory does not exist: {images_dir}"}
        if not labels_dir.exists():
            return {"valid": False, "error": f"Labels directory does not exist: {labels_dir}"}

        image_extensions = {".jpg", ".jpeg", ".png", ".webp"}
        image_stems = {f.stem: f for f in images_dir.iterdir() if f.suffix.lower() in image_extensions}
        label_stems = {f.stem: f for f in labels_dir.iterdir() if f.suffix.lower() == ".txt"}

        missing_labels = list(image_stems.keys() - label_stems.keys())
        orphaned_labels = list(label_stems.keys() - image_stems.keys())
        syntax_errors = []
        out_of_bounds_errors = []

        for stem, label_file in label_stems.items():
            with open(label_file, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]
            for line_no, line in enumerate(lines, 1):
                parts = line.split()
                if len(parts) != 5:
                    syntax_errors.append(f"{label_file.name}:{line_no} Invalid column count ({len(parts)})")
                    continue
                try:
                    cls_id = int(parts[0])
                    x_c, y_c, w, h = map(float, parts[1:])
                    if not (0 <= cls_id < expected_classes):
                        out_of_bounds_errors.append(
                            f"{label_file.name}:{line_no} Class index {cls_id} exceeds range [0, {expected_classes - 1}]"
                        )
                    if not (0.0 <= x_c <= 1.0 and 0.0 <= y_c <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
                        out_of_bounds_errors.append(
                            f"{label_file.name}:{line_no} Coordinates not normalized [0.0, 1.0]: {parts[1:]}"
                        )
                except ValueError as e:
                    syntax_errors.append(f"{label_file.name}:{line_no} Value parse error: {str(e)}")

        is_valid = (
            len(missing_labels) == 0
            and len(syntax_errors) == 0
            and len(out_of_bounds_errors) == 0
            and len(image_stems) > 0
        )

        return {
            "valid": is_valid,
            "total_images": len(image_stems),
            "total_labels": len(label_stems),
            "missing_labels": missing_labels,
            "orphaned_labels": orphaned_labels,
            "syntax_errors": syntax_errors,
            "out_of_bounds_errors": out_of_bounds_errors,
        }
