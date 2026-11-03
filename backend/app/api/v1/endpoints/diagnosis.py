"""
Diagnostic Inference Endpoints for image analysis.
"""

from typing import Optional
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from pydantic import ValidationError

from backend.app.core.config import settings
from backend.app.services.diagnosis_service import diagnosis_service
from ml.inference.schemas import DiagnosticReport
from ml.models.base_model import ModelWeightsNotFoundError

router = APIRouter()

ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}


@router.post(
    "/diagnose/image",
    response_model=DiagnosticReport,
    status_code=status.HTTP_200_OK,
    summary="Upload vehicle inspection image for diagnosis",
    description="Processes inspection image through component localization and visible fault identification pipelines.",
    responses={
        200: {"description": "Diagnostic evaluation successfully executed."},
        400: {"description": "Invalid file format or corrupt image payload."},
        413: {"description": "Image payload exceeds maximum file size."},
        503: {"description": "Model weights are missing or not yet trained."},
    },
)
async def diagnose_vehicle_image(
    file: UploadFile = File(..., description="High-resolution vehicle subsystem image"),
    confidence_threshold: Optional[float] = Form(
        None, description="Optional detection confidence threshold (0.05 to 1.0)"
    ),
):
    """Processes uploaded image for vehicle components and visible defects."""
    # 1. Validate MIME type
    if file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{file.content_type}'. Supported: JPEG, PNG, WEBP.",
        )

    # 2. Read and enforce size limit
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

    # 3. Validate confidence threshold
    if confidence_threshold is not None:
        if not (0.01 <= confidence_threshold <= 1.0):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Confidence threshold must be between 0.01 and 1.0.",
            )

    # 4. Execute ML Inference Pipeline
    try:
        report = diagnosis_service.process_image_diagnosis(
            image_bytes=contents,
            confidence_override=confidence_threshold,
        )
        return report
    except ModelWeightsNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "Model weights uninitialized or not found on disk.",
                "message": str(exc),
                "action_required": "Train the models using ml.training scripts or deposit weights in ml/weights/.",
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
            detail=f"Inference execution failure: {str(exc)}",
        )
