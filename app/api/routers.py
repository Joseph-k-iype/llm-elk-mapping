"""
API Router configuration.
"""

from fastapi import APIRouter
from app.api.endpoints import health, mapping

api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(health.router, prefix="/health", tags=["health"])
api_router.include_router(mapping.router, prefix="/mapping", tags=["mapping"])