"""
Component Detection API Endpoints.
Provides high-performance localization for vehicle components (engine bay, battery, radiator).
"""

from typing import Any, Dict, Optional
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from backend.app.core.config import settings
from backend.app.schemas.component_detection import ComponentDetectionResponse
from backend.app.services.component_service import component_service
from ml.models.base_model import ModelWeightsNotFoundError

router = APIRouter()

ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}


@router.post(
    "/detect/components",
    response_model=ComponentDetectionResponse,
    status_code=status.HTTP_200_OK,
    summary="Detect vehicle components in an uploaded image",
    description=(
        "Localizes automotive subsystems (engine_bay, battery, radiator) using the trained "
        "YOLOv8 Nano detector. Returns absolute pixel coordinates, normalized coordinates, "
        "model metadata, operational status flags, and safety disclaimers."
    ),
    responses={
        200: {"description": "Component detection executed successfully."},
        400: {"description": "Invalid file MIME type, empty upload, corrupt image, or out-of-range confidence."},
        413: {"description": "Image payload exceeds maximum configured file size."},
        503: {"description": "Component detector model checkpoint is missing or uninitialized."},
    },
)
async def detect_vehicle_components(
    file: UploadFile = File(..., description="High-resolution vehicle subsystem image (JPEG, PNG, WEBP)"),
    confidence_threshold: Optional[float] = Form(
        None, description="Optional per-request detection confidence threshold override (0.01 to 1.0)"
    ),
) -> ComponentDetectionResponse:
    """Processes uploaded image for vehicle components and returns standardized detections."""
    # 1. Validate MIME type
    if file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{file.content_type}'. Supported: JPEG, PNG, WEBP.",
        )

    # 2. Read file contents and enforce size limit
    contents = await file.read()
    max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    if len(contents) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Uploaded image ({round(len(contents) / (1024 * 1024), 2)}MB) exceeds limit of {settings.MAX_UPLOAD_SIZE_MB}MB.",
        )

    if len(contents) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty file uploaded.",
        )

    # 3. Validate confidence threshold if overridden
    if confidence_threshold is not None:
        if not (0.01 <= confidence_threshold <= 1.0):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Confidence threshold must be between 0.01 and 1.0.",
            )

    # 4. Execute Component Detection
    try:
        response = component_service.detect_components(
            image_bytes=contents,
            confidence_override=confidence_threshold,
        )
        return response
    except ModelWeightsNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "Component detector weights uninitialized or not found on disk.",
                "message": str(exc),
                "action_required": "Ensure trained weights exist at 'ml/weights/component_detector_best.pt'.",
            },
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Corrupt or unreadable image file: {str(exc)}",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Component detection inference failure: {str(exc)}",
        )


@router.get(
    "/detect/status",
    status_code=status.HTTP_200_OK,
    summary="Component detector status and class policy metadata",
    description="Returns model weights readiness, compute device, active classes, and operational policies.",
)
async def get_component_detector_status() -> Dict[str, Any]:
    """Returns component detector readiness and class policy metadata."""
    return component_service.get_status()
