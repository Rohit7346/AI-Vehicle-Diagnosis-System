from .health import router as health_router
from .diagnosis import router as diagnosis_router
from .component_detection import router as component_router

__all__ = ["health_router", "diagnosis_router", "component_router"]
