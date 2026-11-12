"""
Reusable Image Quality & Duplicate Auditing Engine for Stage 2 Staging.
Performs automated multi-factor quality inspection on raw candidate images:
1. File integrity / corruption detection.
2. Minimum dimension validation (W >= 400, H >= 400).
3. Exact duplicate detection via SHA-256 cryptographic hashing.
4. Near-duplicate perceptual similarity via 64-bit Difference Hash (dHash).
5. Blurriness detection via high-frequency Laplacian variance.
6. Exposure extremes audit (excessively dark < 25 or overexposed > 240).
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
from PIL import Image, ImageOps

SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def make_long_path(path: str | Path) -> str:
    """Ensure Windows long path compatibility."""
    p_str = str(Path(path).resolve())
    if os.name == "nt" and not p_str.startswith("\\\\?\\"):
        return f"\\\\?\\{p_str}"
    return p_str


class ImageQualityAuditor:
    """Comprehensive image quality, integrity, and perceptual duplicate auditing tool."""

    def __init__(
        self,
        min_width: int = 400,
        min_height: int = 400,
        blur_threshold: float = 60.0,
        min_luminance: float = 25.0,
        max_luminance: float = 240.0,
        near_duplicate_hamming_dist: int = 4,
    ):
        self.min_width = min_width
        self.min_height = min_height
        self.blur_threshold = blur_threshold
        self.min_luminance = min_luminance
        self.max_luminance = max_luminance
        self.near_duplicate_dist = near_duplicate_hamming_dist

    @staticmethod
    def compute_dhash(img: Image.Image, hash_size: int = 8) -> int:
        """Computes a 64-bit Difference Hash (dHash) for perceptual similarity detection."""
        resized = img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.BILINEAR)
        arr = np.asarray(resized, dtype=np.int32)
        diff = arr[:, 1:] > arr[:, :-1]
        bits = diff.flatten()
        val = 0
        for b in bits:
            val = (val << 1) | int(b)
        return val

    @staticmethod
    def hamming_distance(h1: int, h2: int) -> int:
        """Calculates bitwise Hamming distance between two 64-bit hashes."""
        return bin(h1 ^ h2).count("1")

    @staticmethod
    def compute_blur_score(img: Image.Image) -> float:
        """Approximates image sharpness using second-order finite difference Laplacian variance."""
        arr = np.asarray(img.convert("L"), dtype=np.float32)
        if arr.shape[0] < 3 or arr.shape[1] < 3:
            return 0.0
        lap = (
            -4.0 * arr[1:-1, 1:-1]
            + arr[:-2, 1:-1]
            + arr[2:, 1:-1]
            + arr[1:-1, :-2]
            + arr[1:-1, 2:]
        )
        return float(lap.var())

    def inspect_single_image(self, img_path: Path) -> Dict[str, object]:
        """Runs the complete inspection suite on a single image file."""
        lp = make_long_path(img_path)
        result: Dict[str, object] = {
            "filename": img_path.name,
            "path": str(img_path),
            "file_size": 0,
            "width": 0,
            "height": 0,
            "sha256": None,
            "dhash": None,
            "blur_variance": 0.0,
            "mean_luminance": 0.0,
            "is_corrupted": False,
            "status": "ACCEPTED",
            "rejection_reasons": [],
        }

        if not os.path.exists(lp):
            result["is_corrupted"] = True
            result["status"] = "REJECTED"
            result["rejection_reasons"].append("File not found on disk")
            return result

        try:
            result["file_size"] = os.path.getsize(lp)
            with open(lp, "rb") as f:
                raw_bytes = f.read()
                result["sha256"] = hashlib.sha256(raw_bytes).hexdigest()

            # Attempt to decode image
            with Image.open(lp) as img:
                img.load()  # Force load pixel data to detect corruption
                w, h = img.size
                result["width"] = w
                result["height"] = h

                # Dimension check
                if w < self.min_width or h < self.min_height:
                    result["rejection_reasons"].append(
                        f"Undersized resolution ({w}x{h} < {self.min_width}x{self.min_height})"
                    )

                # Compute perceptual hash
                result["dhash"] = self.compute_dhash(img)

                # Blurriness check
                blur_var = self.compute_blur_score(img)
                result["blur_variance"] = round(blur_var, 2)
                if blur_var < self.blur_threshold:
                    result["rejection_reasons"].append(
                        f"Excessive blurriness (variance {blur_var:.1f} < {self.blur_threshold})"
                    )

                # Luminance / Exposure check
                gray_arr = np.asarray(img.convert("L"), dtype=np.float32)
                mean_lum = float(gray_arr.mean())
                result["mean_luminance"] = round(mean_lum, 2)
                if mean_lum < self.min_luminance:
                    result["rejection_reasons"].append(
                        f"Excessively underexposed/dark (mean luminance {mean_lum:.1f} < {self.min_luminance})"
                    )
                elif mean_lum > self.max_luminance:
                    result["rejection_reasons"].append(
                        f"Excessively overexposed/washed out (mean luminance {mean_lum:.1f} > {self.max_luminance})"
                    )

        except Exception as e:
            result["is_corrupted"] = True
            result["rejection_reasons"].append(f"Corrupted or invalid image stream: {str(e)}")

        if result["rejection_reasons"]:
            result["status"] = "REJECTED"

        return result

    def audit_collection(self, images_dir: Path) -> Dict[str, object]:
        """Audits an entire directory of images, detecting both exact and perceptual duplicates."""
        images_dir = Path(images_dir).resolve()
        audit_summary = {
            "directory": str(images_dir),
            "total_images": 0,
            "accepted_count": 0,
            "rejected_count": 0,
            "exact_duplicate_groups": [],
            "near_duplicate_pairs": [],
            "reasons_breakdown": {
                "corrupted": 0,
                "undersized": 0,
                "blurry": 0,
                "extreme_exposure": 0,
                "exact_duplicate": 0,
                "near_duplicate": 0,
            },
            "image_records": [],
        }

        if not images_dir.exists():
            return audit_summary

        image_files = sorted([
            f for f in images_dir.iterdir()
            if f.suffix.lower() in SUPPORTED_IMAGE_EXTS
        ], key=lambda x: x.name)

        audit_summary["total_images"] = len(image_files)
        sha_groups: Dict[str, List[str]] = {}
        dhashes: List[Tuple[str, int]] = []

        # Pass 1: Individual image quality inspection
        for img_path in image_files:
            record = self.inspect_single_image(img_path)
            audit_summary["image_records"].append(record)

            if record["is_corrupted"]:
                audit_summary["reasons_breakdown"]["corrupted"] += 1
            if any("Undersized" in r for r in record["rejection_reasons"]):
                audit_summary["reasons_breakdown"]["undersized"] += 1
            if any("blurriness" in r for r in record["rejection_reasons"]):
                audit_summary["reasons_breakdown"]["blurry"] += 1
            if any("underexposed" in r or "overexposed" in r for r in record["rejection_reasons"]):
                audit_summary["reasons_breakdown"]["extreme_exposure"] += 1

            if record["sha256"]:
                sha_groups.setdefault(record["sha256"], []).append(record["filename"])
            if record["dhash"] is not None:
                dhashes.append((record["filename"], record["dhash"]))

        # Pass 2: Exact duplicate analysis
        for sha, filenames in sha_groups.items():
            if len(filenames) > 1:
                audit_summary["exact_duplicate_groups"].append({
                    "sha256": sha,
                    "copies": filenames,
                    "count": len(filenames),
                })
                audit_summary["reasons_breakdown"]["exact_duplicate"] += (len(filenames) - 1)

        # Pass 3: Near-duplicate perceptual similarity analysis
        seen_pairs: Set[Tuple[str, str]] = set()
        for i in range(len(dhashes)):
            name_a, hash_a = dhashes[i]
            for j in range(i + 1, len(dhashes)):
                name_b, hash_b = dhashes[j]
                dist = self.hamming_distance(hash_a, hash_b)
                if dist <= self.near_duplicate_dist:
                    pair_key = (min(name_a, name_b), max(name_a, name_b))
                    if pair_key not in seen_pairs:
                        seen_pairs.add(pair_key)
                        audit_summary["near_duplicate_pairs"].append({
                            "image_a": name_a,
                            "image_b": name_b,
                            "hamming_distance": dist,
                        })

        audit_summary["reasons_breakdown"]["near_duplicate"] = len(audit_summary["near_duplicate_pairs"])

        # Determine final accepted count
        accepted = sum(1 for r in audit_summary["image_records"] if r["status"] == "ACCEPTED")
        audit_summary["accepted_count"] = accepted
        audit_summary["rejected_count"] = len(image_files) - accepted

        return audit_summary


def format_audit_markdown(audit: Dict[str, object], class_name: str) -> str:
    """Formats auditing output into a standardized GitHub-Flavored Markdown report."""
    md = []
    md.append(f"# Image Quality & Duplicate Audit: {class_name.upper()}\n\n")
    md.append(f"**Target Directory:** `{audit['directory']}`\n")
    md.append(f"**Total Images Inspected:** {audit['total_images']}\n")
    md.append(f"**Accepted Count:** {audit['accepted_count']}\n")
    md.append(f"**Rejected Count:** {audit['rejected_count']}\n\n")

    md.append("## 1. Quality & Integrity Metrics\n\n")
    md.append("| Check Metric | Threshold / Constraint | Detected Defect Count | Status |\n")
    md.append("|---|---|:---:|:---:|\n")
    md.append(f"| **Corrupted Images** | Zero tolerance (must decode) | {audit['reasons_breakdown']['corrupted']} | {'PASSED' if audit['reasons_breakdown']['corrupted'] == 0 else 'WARNING'} |\n")
    md.append(f"| **Undersized Resolution** | Minimum 400x400 pixels | {audit['reasons_breakdown']['undersized']} | {'PASSED' if audit['reasons_breakdown']['undersized'] == 0 else 'WARNING'} |\n")
    md.append(f"| **Excessive Blurriness** | Laplacian variance >= 60.0 | {audit['reasons_breakdown']['blurry']} | {'PASSED' if audit['reasons_breakdown']['blurry'] == 0 else 'WARNING'} |\n")
    md.append(f"| **Extreme Exposure** | Mean luminance in [25.0, 240.0] | {audit['reasons_breakdown']['extreme_exposure']} | {'PASSED' if audit['reasons_breakdown']['extreme_exposure'] == 0 else 'WARNING'} |\n")
    md.append(f"| **Exact SHA-256 Duplicates** | Zero duplicate content | {len(audit['exact_duplicate_groups'])} groups | {'PASSED' if len(audit['exact_duplicate_groups']) == 0 else 'DUPLICATES DETECTED'} |\n")
    md.append(f"| **Perceptual Near-Duplicates** | dHash Hamming distance <= 4 | {len(audit['near_duplicate_pairs'])} pairs | {'PASSED' if len(audit['near_duplicate_pairs']) == 0 else 'NEAR-DUPLICATES DETECTED'} |\n\n")

    if audit["exact_duplicate_groups"]:
        md.append("## 2. Exact Duplicates Detected\n\n")
        for g in audit["exact_duplicate_groups"]:
            md.append(f"- **SHA-256 `{g['sha256'][:16]}...`**: {', '.join(g['copies'])}\n")
        md.append("\n")

    if audit["near_duplicate_pairs"]:
        md.append("## 3. Near-Duplicate Perceptual Pairs Detected\n\n")
        for p in audit["near_duplicate_pairs"]:
            md.append(f"- `{p['image_a']}` <-> `{p['image_b']}` (Hamming Distance: {p['hamming_distance']})\n")
        md.append("\n")

    return "".join(md)


def main():
    parser = argparse.ArgumentParser(description="Audit staged image collection for quality, blur, and duplicates.")
    parser.add_argument("--input", type=Path, required=True, help="Path to images directory to audit")
    parser.add_argument("--output", type=Path, default=None, help="Path to write markdown audit report")
    parser.add_argument("--class-name", type=str, default="component", help="Name of component class")
    args = parser.parse_args()

    auditor = ImageQualityAuditor()
    audit_data = auditor.audit_collection(args.input)
    report_md = format_audit_markdown(audit_data, args.class_name)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(report_md)
        print(f"[SUCCESS] Audit report saved to: {args.output}")
    else:
        print(report_md)


if __name__ == "__main__":
    main()
