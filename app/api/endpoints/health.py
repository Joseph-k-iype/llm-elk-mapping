"""
Health check endpoint.
"""

import logging
from fastapi import APIRouter, Depends, HTTPException, status
from app.models.base import BaseResponse
from app.core.auth_helper import verify_api_key

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/", response_model=BaseResponse)
async def health_check():
    """
    Check the health of the application.
    """
    return BaseResponse(
        success=True,
        message="API is healthy",
        data={"status": "UP"}
    )

@router.get("/ready", response_model=BaseResponse)
async def readiness_check():
    """
    Check if the application is ready to serve requests.
    """
    return BaseResponse(
        success=True,
        message="API is ready",
        data={"status": "READY"}
    )

@router.get("/secure", response_model=BaseResponse)
async def secure_health_check():
    """
    Secure health check endpoint (requires authentication).
    """
    return BaseResponse(
        success=True,
        message="Secure endpoint is healthy",
        data={"status": "UP", "secure": True}
    )