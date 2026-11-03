"""
Base detector interface and exceptions for vehicle diagnosis models.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np
from PIL import Image


class ModelWeightsNotFoundError(FileNotFoundError):
    """Raised when model inference is requested but trained weights are absent."""
    pass


class BaseVehicleDetector(ABC):
    """Abstract base class for vehicle detection models."""

    def __init__(
        self,
        weights_path: Optional[Union[str, Path]] = None,
        confidence_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        device: str = "cpu",
    ):
        self.weights_path = Path(weights_path) if weights_path else None
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.model = None
        self._class_names: Dict[int, str] = {}

        if self.weights_path and self.weights_path.exists():
            self._load_weights(self.weights_path)

    @property
    def is_loaded(self) -> bool:
        """Indicates whether genuine model weights are loaded and ready."""
        return self.model is not None

    @property
    def class_names(self) -> Dict[int, str]:
        """Returns the dictionary of class index to class names."""
        return self._class_names

    @abstractmethod
    def _load_weights(self, weights_path: Path) -> None:
        """Internal method to load weights into memory."""
        pass

    @abstractmethod
    def predict(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
        confidence: Optional[float] = None,
    ) -> List[Dict[str, object]]:
        """
        Runs genuine inference on the input image.
        Raises ModelWeightsNotFoundError if weights are not loaded.
        Never emits fake or placeholder predictions.
        """
        pass
