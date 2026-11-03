"""
Aggregator router for API version 1.
"""

from fastapi import APIRouter
from backend.app.api.v1.endpoints import health_router, diagnosis_router

api_v1_router = APIRouter()

api_v1_router.include_router(health_router, tags=["Health & System"])
api_v1_router.include_router(diagnosis_router, tags=["Multimodal Diagnosis"])
