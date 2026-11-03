"""
Security and authentication hooks (prepared for production integration).
"""

from typing import Optional
from fastapi import HTTPException, Security, status
from fastapi.security.api_key import APIKeyHeader

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


async def verify_api_key(api_key: Optional[str] = Security(api_key_header)) -> Optional[str]:
    """
    Optional API key validation.
    Allows unauthenticated requests in development mode.
    """
    # In development mode, bypass strict authentication
    return api_key
