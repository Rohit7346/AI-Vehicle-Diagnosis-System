"""
Data contracts and Pydantic schemas for vehicle fault inference.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional, Tuple
from pydantic import BaseModel, Field


class FaultSeverity(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MODERATE = "MODERATE"
    LOW = "LOW"
    UNKNOWN = "UNKNOWN"


class PipelineStatus(str, Enum):
    READY = "READY"
    WEIGHTS_MISSING = "WEIGHTS_MISSING"
    UNINITIALIZED = "UNINITIALIZED"
    INFERENCE_ERROR = "INFERENCE_ERROR"


class BoundingBox(BaseModel):
    x1: float = Field(..., description="Top-left X coordinate")
    y1: float = Field(..., description="Top-left Y coordinate")
    x2: float = Field(..., description="Bottom-right X coordinate")
    y2: float = Field(..., description="Bottom-right Y coordinate")


class ComponentDetection(BaseModel):
    component_id: str
    label: str
    confidence: float
    bbox: List[float]


class FaultDetection(BaseModel):
    fault_id: str
    label: str
    confidence: float
    severity: FaultSeverity
    recommendation: str
    bbox: List[float]
    associated_component: Optional[str] = None


class ImageMetadata(BaseModel):
    original_width: int
    original_height: int
    format: Optional[str] = None
    clahe_applied: bool = False


class DiagnosticSummary(BaseModel):
    total_components_detected: int
    total_faults_detected: int
    critical_faults_count: int
    high_faults_count: int
    overall_health_score: float = Field(
        ..., description="Calculated vehicle component health index from 0.0 (unsafe) to 100.0 (pristine)"
    )
    status_summary: str


class DiagnosticReport(BaseModel):
    inference_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    pipeline_status: PipelineStatus
    image_metadata: ImageMetadata
    components: List[ComponentDetection] = Field(default_factory=list)
    faults: List[FaultDetection] = Field(default_factory=list)
    summary: DiagnosticSummary
