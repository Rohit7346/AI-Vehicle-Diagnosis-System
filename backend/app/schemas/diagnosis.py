"""
Pydantic Schemas for API Requests and Responses.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from ml.inference.schemas import DiagnosticReport


class SystemMetrics(BaseModel):
    python_version: str
    cuda_available: bool
    device: str
    upload_dir_writable: bool


class ModelWeightStatus(BaseModel):
    loaded: bool
    weights_path: str
    weights_exist_on_disk: bool


class HealthCheckResponse(BaseModel):
    status: str = Field(..., json_schema_extra={"example": "healthy"})
    version: str = Field(..., json_schema_extra={"example": "1.0.0"})
    environment: str = Field(..., json_schema_extra={"example": "development"})
    system: SystemMetrics
    models: Dict[str, Any]


class DiagnosisErrorResponse(BaseModel):
    success: bool = False
    error: str
    details: Optional[str] = None
    action_required: Optional[str] = None
