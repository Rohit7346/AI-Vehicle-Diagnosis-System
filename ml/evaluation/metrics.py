"""
Statistical evaluation metrics for vehicle component and defect detection.
Provides mathematical implementations of IoU, precision, recall, and F1.
"""

from typing import Dict, List, Tuple
import numpy as np


def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """
    Computes Intersection over Union (IoU) between two bounding boxes: [x1, y1, x2, y2].
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    intersection = inter_w * inter_h

    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0
    return float(intersection / union)


def compute_precision_recall_f1(tp: int, fp: int, fn: int) -> Dict[str, float]:
    """
    Computes exact precision, recall, and harmonic F1-score from raw counts.
    """
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
    }
