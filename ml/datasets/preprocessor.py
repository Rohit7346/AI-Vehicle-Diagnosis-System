"""
Image Preprocessing Module for Vehicle Fault Detection.
Provides deterministic normalization, CLAHE contrast enhancement,
aspect-ratio preserving letterboxing, and validation.
"""

from pathlib import Path
from typing import Tuple, Union
import numpy as np
from PIL import Image, ImageOps

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False


class ImagePreprocessor:
    """Preprocesses vehicle inspection imagery for inference and dataset preparation."""

    def __init__(
        self,
        target_size: Tuple[int, int] = (640, 640),
        apply_clahe: bool = True,
        normalize_tensor: bool = False,
    ):
        self.target_size = target_size
        self.apply_clahe = apply_clahe
        self.normalize_tensor = normalize_tensor

    def load_image(self, image_source: Union[str, Path, bytes, np.ndarray, Image.Image]) -> Image.Image:
        """Loads and validates an image from varied input sources."""
        if isinstance(image_source, Image.Image):
            return image_source.convert("RGB")
        
        if isinstance(image_source, (str, Path)):
            path = Path(image_source)
            if not path.exists():
                raise FileNotFoundError(f"Inspection image not found: {path}")
            img = Image.open(path)
            return ImageOps.exif_transpose(img).convert("RGB")
        
        if isinstance(image_source, bytes):
            import io
            img = Image.open(io.BytesIO(image_source))
            return ImageOps.exif_transpose(img).convert("RGB")

        if isinstance(image_source, np.ndarray):
            if image_source.ndim == 2:
                # Grayscale to RGB
                return Image.fromarray(image_source).convert("RGB")
            elif image_source.ndim == 3:
                return Image.fromarray(image_source).convert("RGB")
            raise ValueError(f"Unsupported numpy array shape: {image_source.shape}")

        raise TypeError(f"Unsupported image source type: {type(image_source)}")

    def enhance_contrast(self, image: Image.Image) -> Image.Image:
        """
        Enhances local contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
        to make mechanical cracks, oil seepage, and rust flakes stand out under dim workshop lighting.
        """
        if not self.apply_clahe:
            return image

        np_img = np.array(image)
        if _HAS_CV2:
            # Convert to LAB, apply CLAHE on L-channel
            lab = cv2.cvtColor(np_img, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            merged = cv2.merge((cl, a, b))
            enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
            return Image.fromarray(enhanced)
        else:
            # High-fidelity fallback using PIL ImageOps autocontrast
            return ImageOps.autocontrast(image, cutoff=1)

    def letterbox_resize(
        self,
        image: Image.Image,
        target_size: Tuple[int, int] = (640, 640),
        fill_color: Tuple[int, int, int] = (114, 114, 114),
    ) -> Tuple[Image.Image, float, Tuple[int, int]]:
        """
        Resizes an image preserving aspect ratio with padding (letterbox).
        Returns:
            (letterboxed_image, scale_ratio, (pad_w, pad_h))
        """
        orig_w, orig_h = image.size
        target_w, target_h = target_size

        scale = min(target_w / orig_w, target_h / orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)

        resized = image.resize((new_w, new_h), Image.Resampling.BILINEAR)

        letterboxed = Image.new("RGB", (target_w, target_h), fill_color)
        pad_x = (target_w - new_w) // 2
        pad_y = (target_h - new_h) // 2
        letterboxed.paste(resized, (pad_x, pad_y))

        return letterboxed, scale, (pad_x, pad_y)

    def process(
        self,
        image_source: Union[str, Path, bytes, np.ndarray, Image.Image]
    ) -> Tuple[np.ndarray, dict]:
        """
        Full preprocessing pipeline.
        Returns:
            (preprocessed_ndarray, metadata_dict)
        """
        raw_image = self.load_image(image_source)
        orig_width, orig_height = raw_image.size

        enhanced = self.enhance_contrast(raw_image)
        letterboxed, scale, (pad_x, pad_y) = self.letterbox_resize(enhanced, self.target_size)

        arr = np.array(letterboxed, dtype=np.float32)
        if self.normalize_tensor:
            arr = arr / 255.0  # Normalize to [0, 1]

        metadata = {
            "original_width": orig_width,
            "original_height": orig_height,
            "target_size": self.target_size,
            "scale": scale,
            "padding": (pad_x, pad_y),
            "clahe_applied": self.apply_clahe,
        }
        return arr, metadata
