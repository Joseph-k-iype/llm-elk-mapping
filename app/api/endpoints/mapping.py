"""
Mapping endpoints.
"""

import logging
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from typing import List, Dict, Any
from app.models.mapping import MappingRequest, MappingResponse
from app.models.base import BaseResponse
from app.services.mapping_service import MappingService
from app.core.auth_helper import verify_api_key
from app.core.environment import get_os_env

logger = logging.getLogger(__name__)
router = APIRouter()

# In-memory mapping service cache
_mapping_service_instance = None

async def get_mapping_service() -> MappingService:
    """
    Get or create a mapping service instance.
    
    Returns:
        MappingService: The mapping service instance
    """
    global _mapping_service_instance
    
    if _mapping_service_instance is None:
        # Import here to avoid circular imports
        from app.services.azure_openai import AzureOpenAIService
        from app.services.elasticsearch_service import ElasticsearchService
        from app.services.vector_service import VectorService
        
        # Create service instances
        azure_service = AzureOpenAIService()
        es_service = ElasticsearchService()
        
        # Connect to Elasticsearch
        await es_service.connect()
        
        # Create mapping service
        vector_service = VectorService(azure_service, es_service)
        _mapping_service_instance = MappingService(azure_service, es_service, vector_service)
    
    return _mapping_service_instance

@router.post("/term", response_model=MappingResponse)
async def map_business_term(
    request: MappingRequest,
    mapping_service: MappingService = Depends(get_mapping_service)
):
    """
    Map a business term to existing terms.
    
    Args:
        request: Mapping request
        mapping_service: Mapping service instance
        
    Returns:
        Mapping results
    """
    try:
        logger.info(f"Processing mapping request: {request.name}")
        response = await mapping_service.map_business_term(request)
        return response
    except Exception as e:
        logger.error(f"Error processing mapping request: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing mapping request: {str(e)}"
        )

@router.get("/terms", response_model=BaseResponse)
async def get_all_terms(
    mapping_service: MappingService = Depends(get_mapping_service)
):
    """
    Get all business terms.
    
    Args:
        mapping_service: Mapping service instance
        
    Returns:
        List of business terms
    """
    try:
        logger.info("Retrieving all business terms")
        terms = await mapping_service.get_all_business_terms()
        return BaseResponse(
            success=True,
            message=f"Retrieved {len(terms)} business terms",
            data=terms
        )
    except Exception as e:
        logger.error(f"Error retrieving business terms: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving business terms: {str(e)}"
        )

@router.post("/refresh-token", response_model=BaseResponse)
async def refresh_azure_token(
    background_tasks: BackgroundTasks,
    mapping_service: MappingService = Depends(get_mapping_service)
):
    """
    Refresh the Azure token.
    
    Args:
        background_tasks: FastAPI background tasks
        mapping_service: Mapping service instance
        
    Returns:
        Success message
    """
    try:
        logger.info("Refreshing Azure token")
        # Run token refresh in background
        background_tasks.add_task(mapping_service.azure_service.refresh_tokens)
        return BaseResponse(
            success=True,
            message="Token refresh initiated",
            data=None
        )
    except Exception as e:
        logger.error(f"Error refreshing Azure token: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error refreshing Azure token: {str(e)}"
        )