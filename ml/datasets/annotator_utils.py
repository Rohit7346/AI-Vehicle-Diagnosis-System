"""
Annotation format validation, leakage detection, and dataset audit utilities.
Enforces strict Ultralytics YOLO compliance, coordinate boundaries, and class balance.
"""

import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import yaml

COMPONENT_ONTOLOGY = {
    0: "engine_bay",
    1: "battery",
    2: "alternator",
    3: "radiator",
    4: "brake_rotor",
    5: "brake_caliper",
    6: "tire_tread",
    7: "tire_sidewall",
    8: "suspension_strut",
    9: "exhaust_manifold",
}

SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


class AnnotationValidator:
    """Verifies label files against YOLO annotation contracts and validates dataset integrity."""

    @staticmethod
    def validate_split(
        images_dir: Path,
        labels_dir: Path,
        expected_classes: int = 10,
        class_names: Optional[Dict[int, str]] = None,
    ) -> Dict[str, object]:
        """
        Validates that all images in images_dir have matching valid annotations in labels_dir.
        Checks normalized coordinates, class ID boundaries, box dimensions, and class distribution.
        """
        names = class_names or COMPONENT_ONTOLOGY

        if not images_dir.exists():
            return {"valid": False, "error": f"Image directory does not exist: {images_dir}"}
        if not labels_dir.exists():
            return {"valid": False, "error": f"Labels directory does not exist: {labels_dir}"}

        image_files = {f.stem: f for f in images_dir.iterdir() if f.suffix.lower() in SUPPORTED_IMAGE_EXTS}
        label_files = {f.stem: f for f in labels_dir.iterdir() if f.suffix.lower() == ".txt"}

        missing_labels = list(image_files.keys() - label_files.keys())
        orphaned_labels = list(label_files.keys() - image_files.keys())
        syntax_errors = []
        out_of_bounds_errors = []
        degenerate_boxes = []

        class_counts = {cls_id: 0 for cls_id in range(expected_classes)}
        total_annotations = 0

        for stem, label_file in label_files.items():
            try:
                with open(label_file, "r", encoding="utf-8") as f:
                    lines = [line.strip() for line in f if line.strip()]
            except Exception as e:
                syntax_errors.append(f"{label_file.name}: Cannot read file ({str(e)})")
                continue

            for line_no, line in enumerate(lines, 1):
                parts = line.split()
                if len(parts) != 5:
                    syntax_errors.append(f"{label_file.name}:{line_no} Invalid column count ({len(parts)} expected 5)")
                    continue
                try:
                    cls_id = int(parts[0])
                    x_c, y_c, w, h = map(float, parts[1:])

                    # Class index boundary check
                    if not (0 <= cls_id < expected_classes):
                        out_of_bounds_errors.append(
                            f"{label_file.name}:{line_no} Class ID {cls_id} out of range [0, {expected_classes - 1}]"
                        )
                        continue

                    # Bounding box coordinate boundary checks (normalized [0, 1])
                    if not (0.0 <= x_c <= 1.0 and 0.0 <= y_c <= 1.0):
                        out_of_bounds_errors.append(
                            f"{label_file.name}:{line_no} Center coordinates ({x_c}, {y_c}) out of bounds [0.0, 1.0]"
                        )

                    if not (0.0 < w <= 1.0 and 0.0 < h <= 1.0):
                        degenerate_boxes.append(
                            f"{label_file.name}:{line_no} Degenerate dimensions (w={w}, h={h})"
                        )

                    # Check box corner bounds with small float tolerance
                    eps = 1e-3
                    x_min = x_c - w / 2.0
                    x_max = x_c + w / 2.0
                    y_min = y_c - h / 2.0
                    y_max = y_c + h / 2.0

                    if x_min < -eps or x_max > 1.0 + eps or y_min < -eps or y_max > 1.0 + eps:
                        out_of_bounds_errors.append(
                            f"{label_file.name}:{line_no} Box corners exceed image boundaries: "
                            f"[{round(x_min, 3)}, {round(y_min, 3)}, {round(x_max, 3)}, {round(y_max, 3)}]"
                        )

                    class_counts[cls_id] += 1
                    total_annotations += 1

                except ValueError as e:
                    syntax_errors.append(f"{label_file.name}:{line_no} Numeric parse error: {str(e)}")

        is_valid = (
            len(missing_labels) == 0
            and len(syntax_errors) == 0
            and len(out_of_bounds_errors) == 0
            and len(degenerate_boxes) == 0
            and len(image_files) > 0
        )

        return {
            "valid": is_valid,
            "images_dir": str(images_dir),
            "labels_dir": str(labels_dir),
            "total_images": len(image_files),
            "total_labels": len(label_files),
            "total_annotations": total_annotations,
            "class_distribution": {names.get(k, f"class_{k}"): v for k, v in class_counts.items()},
            "class_counts_by_id": class_counts,
            "missing_labels": missing_labels,
            "orphaned_labels": orphaned_labels,
            "syntax_errors": syntax_errors,
            "out_of_bounds_errors": out_of_bounds_errors,
            "degenerate_boxes": degenerate_boxes,
        }

    @staticmethod
    def detect_data_leakage(base_data_dir: Path) -> Dict[str, object]:
        """
        Verifies there is no data leakage between train, validation, and test splits.
        Detects both identical filenames and identical file content hashes (SHA-256).
        """
        splits = ["train", "val", "test"]
        split_hashes: Dict[str, Dict[str, Path]] = {"train": {}, "val": {}, "test": {}}
        split_stems: Dict[str, Set[str]] = {"train": set(), "val": set(), "test": set()}

        for split in splits:
            # Check canonical data/images/<split> or fallback to data/processed/<split>
            img_dir = base_data_dir / "images" / split
            if not img_dir.exists():
                img_dir = base_data_dir / "processed" / split
            if not img_dir.exists():
                continue

            for f in img_dir.iterdir():
                if f.suffix.lower() in SUPPORTED_IMAGE_EXTS:
                    split_stems[split].add(f.stem)
                    try:
                        with open(f, "rb") as fp:
                            file_hash = hashlib.sha256(fp.read()).hexdigest()
                            split_hashes[split][file_hash] = f
                    except Exception:
                        pass

        stem_leaks = []
        hash_leaks = []

        # Check train vs val
        t_v_stems = split_stems["train"] & split_stems["val"]
        if t_v_stems:
            stem_leaks.append(f"Train/Val Filename Overlap: {list(t_v_stems)[:5]}")
        t_v_hashes = set(split_hashes["train"].keys()) & set(split_hashes["val"].keys())
        if t_v_hashes:
            hash_leaks.append(f"Train/Val Exact Duplicate Images: {len(t_v_hashes)} files")

        # Check train vs test
        t_t_stems = split_stems["train"] & split_stems["test"]
        if t_t_stems:
            stem_leaks.append(f"Train/Test Filename Overlap: {list(t_t_stems)[:5]}")
        t_t_hashes = set(split_hashes["train"].keys()) & set(split_hashes["test"].keys())
        if t_t_hashes:
            hash_leaks.append(f"Train/Test Exact Duplicate Images: {len(t_t_hashes)} files")

        # Check val vs test
        v_t_stems = split_stems["val"] & split_stems["test"]
        if v_t_stems:
            stem_leaks.append(f"Val/Test Filename Overlap: {list(v_t_stems)[:5]}")
        v_t_hashes = set(split_hashes["val"].keys()) & set(split_hashes["test"].keys())
        if v_t_hashes:
            hash_leaks.append(f"Val/Test Exact Duplicate Images: {len(v_t_hashes)} files")

        has_leakage = len(stem_leaks) > 0 or len(hash_leaks) > 0
        return {
            "has_leakage": has_leakage,
            "filename_overlaps": stem_leaks,
            "content_duplicates": hash_leaks,
            "train_samples": len(split_stems["train"]),
            "val_samples": len(split_stems["val"]),
            "test_samples": len(split_stems["test"]),
        }

    @staticmethod
    def audit_full_dataset(base_data_dir: Path, expected_classes: int = 10) -> Dict[str, object]:
        """
        Executes a complete dataset audit across all splits and returns summary statistics.
        """
        splits_audit = {}
        for split in ["train", "val", "test"]:
            # Check canonical data/images/<split> first, then fallback to processed
            img_dir = base_data_dir / "images" / split
            lbl_dir = base_data_dir / "labels" / split
            if not img_dir.exists() or not any(img_dir.iterdir()):
                alt_img = base_data_dir / "processed" / split
                alt_lbl = base_data_dir / "annotations" / split
                if alt_img.exists() and any(alt_img.iterdir()):
                    img_dir = alt_img
                    lbl_dir = alt_lbl

            splits_audit[split] = AnnotationValidator.validate_split(img_dir, lbl_dir, expected_classes)

        leakage_audit = AnnotationValidator.detect_data_leakage(base_data_dir)

        # Check representation: every class should have examples in train
        train_counts = splits_audit["train"].get("class_counts_by_id", {})
        val_counts = splits_audit["val"].get("class_counts_by_id", {})
        missing_in_train = [COMPONENT_ONTOLOGY[c] for c, count in train_counts.items() if count == 0]
        missing_in_val = [COMPONENT_ONTOLOGY[c] for c, count in val_counts.items() if count == 0]

        return {
            "splits": splits_audit,
            "leakage": leakage_audit,
            "missing_in_train": missing_in_train,
            "missing_in_val": missing_in_val,
            "ready_for_training": (
                splits_audit["train"].get("valid", False)
                and splits_audit["val"].get("valid", False)
                and not leakage_audit.get("has_leakage", False)
                and len(missing_in_train) == 0
            ),
        }


def format_summary_table(audit_results: Dict[str, object]) -> str:
    """Formats audit results as a readable Markdown summary table."""
    splits = audit_results.get("splits", {})
    leakage = audit_results.get("leakage", {})

    lines = []
    lines.append("| Split | Images | Labels | Annotations | Status |")
    lines.append("| :--- | :---: | :---: | :---: | :--- |")

    for s in ["train", "val", "test"]:
        res = splits.get(s, {})
        valid_badge = "VALID" if res.get("valid") else "EMPTY / INVALID"
        lines.append(
            f"| `{s}` | {res.get('total_images', 0)} | {res.get('total_labels', 0)} | "
            f"{res.get('total_annotations', 0)} | {valid_badge} |"
        )

    lines.append("\n### Class Distribution:")
    lines.append("| ID | Class Name | Train Count | Val Count | Test Count |")
    lines.append("| :---: | :--- | :---: | :---: | :---: |")

    for cid in range(10):
        cname = COMPONENT_ONTOLOGY[cid]
        tr_c = splits.get("train", {}).get("class_counts_by_id", {}).get(cid, 0)
        va_c = splits.get("val", {}).get("class_counts_by_id", {}).get(cid, 0)
        te_c = splits.get("test", {}).get("class_counts_by_id", {}).get(cid, 0)
        lines.append(f"| `{cid}` | `{cname}` | {tr_c} | {va_c} | {te_c} |")

    lines.append("\n### Data Leakage Audit:")
    if leakage.get("has_leakage"):
        lines.append("- **WARNING: DATA LEAKAGE DETECTED!**")
        for err in leakage.get("filename_overlaps", []) + leakage.get("content_duplicates", []):
            lines.append(f"  - {err}")
    else:
        lines.append("- **PASSED**: Zero image overlap detected across train, val, and test splits.")

    return "\n".join(lines)


if __name__ == "__main__":
    base = Path("data")
    audit = AnnotationValidator.audit_full_dataset(base)
    print("=" * 70)
    print(" AI Vehicle Diagnosis System: Dataset Audit & Readiness Report")
    print("=" * 70)
    print(format_summary_table(audit))
    print("\n" + "=" * 70)
    if audit.get("ready_for_training"):
        print("[READY] Dataset satisfies all Ultralytics YOLO requirements.")
    else:
        print("[PENDING] Dataset not yet populated or validated. Awaiting sample ingestion.")
    print("=" * 70)
