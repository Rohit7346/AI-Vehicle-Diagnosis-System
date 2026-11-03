"""
Local development server launcher.
Usage:
    python backend/run.py
    # or
    uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
"""

import uvicorn
from backend.app.core.config import settings

if __name__ == "__main__":
    uvicorn.run(
        "backend.app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info",
    )
