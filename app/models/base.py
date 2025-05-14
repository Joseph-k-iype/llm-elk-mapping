"""
Base models for the application.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class BaseResponse(BaseModel):
    """Base response model for API endpoints."""
    success: bool
    message: str = ""
    data: Optional[Any] = None