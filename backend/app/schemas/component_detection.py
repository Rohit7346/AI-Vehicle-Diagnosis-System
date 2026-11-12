"""
Pydantic schemas for Component Localization API requests and responses.
Defines strict schemas for vehicle subsystem detections, bounding boxes,
model metadata, and operational safeguards.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class BoundingBoxPixels(BaseModel):
    """Absolute pixel coordinates in original image space."""

    x_min: int = Field(..., description="Top-left X coordinate in pixels", ge=0)
    y_min: int = Field(..., description="Top-left Y coordinate in pixels", ge=0)
    x_max: int = Field(..., description="Bottom-right X coordinate in pixels", ge=0)
    y_max: int = Field(..., description="Bottom-right Y coordinate in pixels", ge=0)


class BoundingBoxNormalized(BaseModel):
    """Normalized coordinates scaled to [0.0, 1.0]."""

    x_min: float = Field(..., description="Normalized top-left X coordinate", ge=0.0, le=1.0)
    y_min: float = Field(..., description="Normalized top-left Y coordinate", ge=0.0, le=1.0)
    x_max: float = Field(..., description="Normalized bottom-right X coordinate", ge=0.0, le=1.0)
    y_max: float = Field(..., description="Normalized bottom-right Y coordinate", ge=0.0, le=1.0)


class DetectedComponent(BaseModel):
    """Individual vehicle component detected in the image."""

    component_id: str = Field(..., description="Unique detection identifier")
    class_id: int = Field(..., description="Zero-indexed class ID (0: engine_bay, 1: battery, 2: radiator)", ge=0, le=2)
    class_name: str = Field(..., description="Standardized component class name")
    confidence: float = Field(..., description="Model detection confidence score", ge=0.0, le=1.0)
    box_pixels: BoundingBoxPixels = Field(..., description="Bounding box in absolute integer pixels")
    box_normalized: BoundingBoxNormalized = Field(..., description="Bounding box in normalized float coordinates")
    status: str = Field(..., description="Operational status: 'enabled' or 'experimental'")
    is_experimental: bool = Field(..., description="Whether detection comes from an experimental class")
    warning: Optional[str] = Field(None, description="Operational safety warning or caveat for this component")


class ImageDimensions(BaseModel):
    """Metadata regarding the processed image dimensions."""

    width: int = Field(..., description="Original image width in pixels", gt=0)
    height: int = Field(..., description="Original image height in pixels", gt=0)
    format: Optional[str] = Field(None, description="Image format (e.g. JPEG, PNG)")


class ComponentDetectionSummary(BaseModel):
    """High-level summary of all localized components."""

    total_detections: int = Field(..., description="Total count of components identified", ge=0)
    detected_classes: List[str] = Field(..., description="Unique component class names detected")
    class_counts: Dict[str, int] = Field(..., description="Per-class detection tally")


class ComponentDetectionResponse(BaseModel):
    """Complete structured response for component detection endpoint."""

    success: bool = Field(True, description="Whether inference completed successfully")
    model_name: str = Field(..., description="Component detection model identifier")
    model_version: str = Field(..., description="Model checkpoint semantic version")
    device: str = Field(..., description="Inference compute device used (e.g. 'cuda:0', 'cpu')")
    inference_latency_ms: float = Field(..., description="Model forward-pass latency in milliseconds", ge=0.0)
    image: ImageDimensions = Field(..., description="Image dimensional metadata")
    detections: List[DetectedComponent] = Field(default_factory=list, description="List of localized components")
    summary: ComponentDetectionSummary = Field(..., description="Component tally and class summaries")
    disclaimer: str = Field(..., description="Mandatory disclaimer separating component detection from fault diagnosis")
    safeguards: Dict[str, str] = Field(..., description="Policy safeguards and operational caveats per class")
