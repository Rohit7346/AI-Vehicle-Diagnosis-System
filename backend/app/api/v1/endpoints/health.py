"""
Health Check and System Readiness Endpoint.
"""

from fastapi import APIRouter, status
from backend.app.schemas.diagnosis import HealthCheckResponse
from backend.app.services.diagnosis_service import diagnosis_service

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthCheckResponse,
    status_code=status.HTTP_200_OK,
    summary="System and Model Health Readiness",
    description="Returns server status, hardware acceleration information, and ML model checkpoint readiness.",
)
async def get_health():
    """Health check endpoint."""
    return diagnosis_service.get_system_health()
