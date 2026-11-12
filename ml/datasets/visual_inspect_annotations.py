"""
Visual Annotation Inspection Tool for Stage 2 Dataset Audit.
Renders YOLO bounding boxes on real prototype images to visually verify
ground truth quality, boundary tightness, and class consistency.

Selects:
- 10 representative images containing 'battery' (Class 1)
- 10 representative images containing 'radiator' (Class 3)
Saves rendered images with class tags into docs/dataset_review/.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
from PIL import Image, ImageDraw, ImageFont

PROTOTYPE_ONTOLOGY = {
    0: "engine_bay",
    1: "battery",
    2: "alternator",
    3: "radiator",
    4: "exhaust_manifold",
}

# Distinctive high-contrast color scheme for visualization
CLASS_COLORS = {
    0: (234, 179, 8),    # Amber
    1: (34, 197, 94),    # Emerald Green (Battery)
    2: (168, 85, 247),   # Purple (Alternator)
    3: (14, 165, 233),   # Sky Blue (Radiator)
    4: (239, 68, 68),    # Red (Exhaust Manifold)
}


def make_long_path(path: str | Path) -> str:
    """Ensure Windows long path compatibility."""
    p_str = str(Path(path).resolve())
    if os.name == "nt" and not p_str.startswith("\\\\?\\"):
        return f"\\\\?\\{p_str}"
    return p_str


def load_label_boxes(label_path: Path) -> List[Tuple[int, float, float, float, float]]:
    """Loads normalized YOLO bounding boxes from a label file."""
    boxes = []
    lbl_lp = make_long_path(label_path)
    if not os.path.exists(lbl_lp):
        return boxes

    with open(lbl_lp, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) == 5:
                try:
                    cls_id = int(parts[0])
                    x_c, y_c, w, h = map(float, parts[1:])
                    boxes.append((cls_id, x_c, y_c, w, h))
                except ValueError:
                    continue
    return boxes


def draw_bounding_boxes(
    img: Image.Image,
    boxes: List[Tuple[int, float, float, float, float]],
    line_width: int = 3,
) -> Image.Image:
    """Draws bounding boxes and class badges on a PIL Image."""
    annotated = img.convert("RGB")
    draw = ImageDraw.Draw(annotated)
    w_img, h_img = annotated.size

    for cls_id, x_c, y_c, w, h in boxes:
        color = CLASS_COLORS.get(cls_id, (255, 255, 255))
        cls_name = PROTOTYPE_ONTOLOGY.get(cls_id, f"class_{cls_id}")

        # Compute pixel coordinates
        x_min = max(0, int((x_c - w / 2.0) * w_img))
        y_min = max(0, int((y_c - h / 2.0) * h_img))
        x_max = min(w_img - 1, int((x_c + w / 2.0) * w_img))
        y_max = min(h_img - 1, int((y_c + h / 2.0) * h_img))

        # Draw main bounding rectangle
        draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=line_width)

        # Draw label header badge
        badge_text = f"{cls_name} ({cls_id})"
        # Simple text box placement
        badge_y_min = max(0, y_min - 16)
        badge_y_max = y_min
        badge_x_max = min(w_img, x_min + len(badge_text) * 8 + 8)

        draw.rectangle([x_min, badge_y_min, badge_x_max, badge_y_max], fill=color)
        draw.text((x_min + 4, badge_y_min + 1), badge_text, fill=(0, 0, 0))

    return annotated


def run_visual_inspection(
    dataset_dir: Path,
    output_dir: Path,
    num_samples_per_class: int = 10,
) -> Dict[str, List[Dict[str, object]]]:
    """Inspects and saves sample annotated images for battery and radiator classes."""
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    samples_collected = {1: [], 3: []}
    splits = ["train", "val", "test"]

    # Iterate splits to collect representative samples
    for split in splits:
        img_dir = dataset_dir / "images" / split
        lbl_dir = dataset_dir / "labels" / split
        if not img_dir.exists() or not lbl_dir.exists():
            continue

        img_files = sorted([f for f in img_dir.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"}], key=lambda x: x.name)

        for img_path in img_files:
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            boxes = load_label_boxes(lbl_path)
            if not boxes:
                continue

            classes_in_img = {b[0] for b in boxes}

            # Collect for Battery (ID 1)
            if 1 in classes_in_img and len(samples_collected[1]) < num_samples_per_class:
                samples_collected[1].append((split, img_path, lbl_path, boxes))

            # Collect for Radiator (ID 3)
            if 3 in classes_in_img and len(samples_collected[3]) < num_samples_per_class:
                samples_collected[3].append((split, img_path, lbl_path, boxes))

            if len(samples_collected[1]) >= num_samples_per_class and len(samples_collected[3]) >= num_samples_per_class:
                break

    inspection_log = {"battery": [], "radiator": []}

    # Render and save Battery samples
    for idx, (split, img_path, lbl_path, boxes) in enumerate(samples_collected[1], 1):
        img = Image.open(make_long_path(img_path))
        annotated = draw_bounding_boxes(img, boxes)
        out_fname = f"battery_sample_{idx:02d}.jpg"
        out_path = output_dir / out_fname
        annotated.save(out_path, quality=92)

        boxes_detail = [
            {"class_id": b[0], "class_name": PROTOTYPE_ONTOLOGY[b[0]], "x_c": round(b[1], 4), "y_c": round(b[2], 4), "w": round(b[3], 4), "h": round(b[4], 4)}
            for b in boxes
        ]
        inspection_log["battery"].append({
            "sample_index": idx,
            "filename": out_fname,
            "source_split": split,
            "original_image": img_path.name,
            "dimensions": f"{img.width}x{img.height}",
            "annotations": boxes_detail,
        })

    # Render and save Radiator samples
    for idx, (split, img_path, lbl_path, boxes) in enumerate(samples_collected[3], 1):
        img = Image.open(make_long_path(img_path))
        annotated = draw_bounding_boxes(img, boxes)
        out_fname = f"radiator_sample_{idx:02d}.jpg"
        out_path = output_dir / out_fname
        annotated.save(out_path, quality=92)

        boxes_detail = [
            {"class_id": b[0], "class_name": PROTOTYPE_ONTOLOGY[b[0]], "x_c": round(b[1], 4), "y_c": round(b[2], 4), "w": round(b[3], 4), "h": round(b[4], 4)}
            for b in boxes
        ]
        inspection_log["radiator"].append({
            "sample_index": idx,
            "filename": out_fname,
            "source_split": split,
            "original_image": img_path.name,
            "dimensions": f"{img.width}x{img.height}",
            "annotations": boxes_detail,
        })

    # Write JSON summary log
    log_path = output_dir / "inspection_summary.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(inspection_log, f, indent=2)

    return inspection_log


def main():
    parser = argparse.ArgumentParser(description="Visual inspection of prototype annotations.")
    parser.add_argument("--dataset", type=Path, default=Path("data/prototype_5class"), help="Path to prototype dataset")
    parser.add_argument("--output", type=Path, default=Path("docs/dataset_review"), help="Output directory for annotated images")
    parser.add_argument("--count", type=int, default=10, help="Number of samples per class")
    args = parser.parse_args()

    print(f"[INSPECT] Rendering {args.count} battery and {args.count} radiator images...")
    log = run_visual_inspection(args.dataset, args.output, num_samples_per_class=args.count)
    print(f"[SUCCESS] Saved {len(log['battery'])} battery images and {len(log['radiator'])} radiator images to {args.output}")


if __name__ == "__main__":
    main()
