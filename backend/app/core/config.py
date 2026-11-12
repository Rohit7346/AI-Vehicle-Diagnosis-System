"""
Core application settings and environment variable configuration.
"""

from pathlib import Path
from typing import List, Union
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """System-wide backend configuration."""

    PROJECT_NAME: str = "AI Multimodal Vehicle Diagnosis System"
    VERSION: str = "1.0.0"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    API_V1_PREFIX: str = "/api/v1"

    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # CORS Configuration
    CORS_ORIGINS: Union[List[str], str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
    ]

    @field_validator("CORS_ORIGINS", mode="before")
    def parse_cors_origins(cls, v):
        if isinstance(v, str) and not v.startswith("["):
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v

    # Machine Learning Checkpoint Paths & Settings
    COMPONENT_MODEL_WEIGHTS: Path = Path("ml/weights/component_detector_best.pt")
    FAULT_MODEL_WEIGHTS: Path = Path("ml/weights/fault_detector_best.pt")
    COMPONENT_MODEL_NAME: str = "YOLOv8n-VehicleComponents-3Class"
    COMPONENT_MODEL_VERSION: str = "3.1.0"
    COMPONENT_CONFIDENCE_THRESHOLD: float = 0.25
    DEVICE: str = "auto"
    CONFIDENCE_THRESHOLD: float = 0.35
    IOU_THRESHOLD: float = 0.45

    # Storage & Upload Limits
    UPLOAD_DIR: Path = Path("temp/uploads")
    MAX_UPLOAD_SIZE_MB: int = 25

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
