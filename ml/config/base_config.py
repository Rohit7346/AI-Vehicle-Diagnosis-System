"""
Configuration definitions for AI/ML training and inference.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
import yaml


@dataclass
class DetectionConfig:
    """Configuration schema for object detection models."""
    model_name: str
    weights_path: Path
    input_size: int = 640
    batch_size: int = 16
    epochs: int = 100
    learning_rate: float = 0.01
    device: str = "cpu"
    confidence_threshold: float = 0.35
    iou_threshold: float = 0.45
    classes: List[str] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "DetectionConfig":
        """Loads configuration from YAML file."""
        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file not found at: {yaml_path}")
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        
        return cls(
            model_name=data.get("model_name", "yolov8m"),
            weights_path=Path(data.get("weights_path", "ml/weights/best.pt")),
            input_size=data.get("input_size", 640),
            batch_size=data.get("batch_size", 16),
            epochs=data.get("epochs", 100),
            learning_rate=data.get("learning_rate", 0.01),
            device=data.get("device", "cpu"),
            confidence_threshold=data.get("confidence_threshold", 0.35),
            iou_threshold=data.get("iou_threshold", 0.45),
            classes=data.get("names", []),
        )


@dataclass
class PipelineConfig:
    """Holistic diagnostic pipeline orchestration configuration."""
    component_config: DetectionConfig
    fault_config: DetectionConfig
    save_annotated_images: bool = True
    output_dir: Path = Path("runs/inference")
