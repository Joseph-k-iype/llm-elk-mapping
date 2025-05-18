"""
Enhanced mapping endpoints with asynchronous processing support and required ID fields.
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional, Union
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks, Query, Path
from fastapi.responses import JSONResponse

from app.models.mapping import MappingRequest, MappingResponse, MappingResult
from app.models.job import MappingJob, JobStatus, MappingJobResponse, MappingJobsResponse, MappingJobRequest
from app.models.base import BaseResponse
from app.services.mapping_service import MappingService
from app.services.azure_openai import AzureOpenAIService
from app.services.elasticsearch_service import ElasticsearchService
from app.services.vector_service import VectorService
from app.services.job_tracking_service import JobTrackingService
from app.core.auth_helper import verify_api_key
from app.core.environment import get_os_env
from datetime import datetime

# Setup logger
logger = logging.getLogger(__name__)

# Define the router at the module level
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
        try:
            logger.info("Initializing mapping service...")
            
            # Create service instances
            azure_service = AzureOpenAIService()
            es_service = ElasticsearchService()
            
            # Connect to Elasticsearch
            await es_service.connect()
            
            # Create vector service
            vector_service = VectorService(azure_service, es_service)
            
            # Create mapping service
            _mapping_service_instance = MappingService(azure_service, es_service, vector_service)
            
            logger.info("Mapping service initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing mapping service: {e}")
            raise
    
    return _mapping_service_instance

@router.post("/term", response_model=MappingResponse, 
           summary="Map business term synchronously",
           description="Map a business term to existing terms using AI agent evaluation")
async def map_business_term(
    request: MappingRequest,
    mapping_service: MappingService = Depends(get_mapping_service)
):
    """
    Map a business term to existing terms (synchronous).
    
    Args:
        request: Mapping request containing the term to map
        mapping_service: Mapping service instance
        
    Returns:
        Mapping results with AI agent evaluation
    """
    try:
        logger.info(f"Processing synchronous mapping request: {request.name}")
        response = await mapping_service.map_business_term(request)
        return response
    except Exception as e:
        logger.error(f"Error processing synchronous mapping request: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing mapping request: {str(e)}"
        )

@router.post("/term/async", response_model=MappingJobResponse,
           summary="Map business term asynchronously",
           description="Create a background job to map a business term with required job ID and process ID")
async def map_business_term_async(
    request: MappingJobRequest,
    mapping_service: MappingService = Depends(get_mapping_service)
):
    """
    Map a business term to existing terms asynchronously.
    
    Creates a background job for mapping and returns a job ID that can be used
    to check on the status and results.
    
    Args:
        request: Mapping job request (including required ID fields)
        mapping_service: Mapping service instance
        
    Returns:
        Job ID and status
    """
    try:
        logger.info(f"Creating asynchronous mapping job with ID {request.id}, process ID {request.process_id} for: {request.name}")
        
        # Validate required fields
        if not request.id or not request.process_id:
            logger.error("Missing required ID fields")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Both 'id' and 'process_id' are required fields"
            )
        
        # Convert to MappingRequest
        mapping_request = request.to_mapping_request()
        
        # Create job with specified IDs
        job = await mapping_service.create_mapping_job(request.id, request.process_id, mapping_request)
        
        return MappingJobResponse(
            success=True,
            message="Mapping job created successfully",
            job=job
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating asynchronous mapping job: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating mapping job: {str(e)}"
        )

@router.get("/jobs/{job_id}", response_model=MappingJobResponse,
          summary="Get job by ID",
          description="Retrieve a mapping job by its job ID")
async def get_mapping_job(
    job_id: str = Path(..., description="ID of the mapping job"),
    mapping_service: MappingService = Depends(get_mapping_service)
):
    """
    Get a mapping job by ID.
    
    Args:
        job_id: ID of the job to get
        mapping_service: Mapping service instance
        
    Returns:
        Job details including status and results if completed
    """
    try:
        logger.info(f"Getting mapping job: {job_id}")
        job = await mapping_service.get_mapping_job(job_id)
        
        if not job:
            logger.warning(f"Mapping job {job_id} not found")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Mapping job {job_id} not found"
            )
        
        return MappingJobResponse(
            success=True,
            message="Mapping job retrieved successfully",
            job=job
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting mapping job: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting mapping job: {str(e)}"
        )

@router.get("/jobs/process/{process_id}", response_model=MappingJobResponse,
          summary="Get job by process ID",
          description="Retrieve a mapping job by its process ID")
async def get_mapping_job_by_process_id(
    process_id: str = Path(..., description="Process ID of the mapping job"),
    mapping_service: MappingService = Depends(get_mapping_service)
):
    """
    Get a mapping job by process ID.
    
    Args:
        process_id: Process ID of the job to get
        mapping_service: Mapping service instance
        
    Returns:
        Job details including status and results if completed
    """
    try:
        logger.info(f"Getting mapping job by process ID: {process_id}")
        job = await mapping_service.get_mapping_job_by_process_id(process_id)
        
        if not job:
            logger.warning(f"Mapping job with process ID {process_id} not found")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Mapping job with process ID {process_id} not found"
            )
        
        return MappingJobResponse(
            success=True,
            message="Mapping job retrieved successfully",
            job=job
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting mapping job by process ID: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting mapping job by process ID: {str(e)}"
        )

@router.get("/jobs", response_model=MappingJobsResponse,
          summary="List mapping jobs",
          description="List mapping jobs with optional filtering by status and pagination")
async def get_mapping_jobs(
    status: Optional[JobStatus] = Query(None, description="Filter by job status"),
    page: int = Query(1, ge=1, description="Page number (starting from 1)"),
    page_size: int = Query(10, ge=1, le=100, description="Number of items per page"),
    mapping_service: MappingService = Depends(get_mapping_service)
):
    """
    Get mapping jobs with optional filtering.
    
    Args:
        status: Filter by job status (optional)
        page: Page number (1-based)
        page_size: Number of results per page
        mapping_service: Mapping service instance
        
    Returns:
        List of jobs with pagination information
    """
    try:
        logger.info(f"Getting mapping jobs (status={status}, page={page}, page_size={page_size})")
        jobs, total = await mapping_service.get_mapping_jobs(status, page, page_size)
        
        return MappingJobsResponse(
            success=True,
            message=f"Retrieved {len(jobs)} mapping jobs",
            jobs=jobs,
            total=total,
            page=page,
            page_size=page_size
        )
    except Exception as e:
        logger.error(f"Error getting mapping jobs: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting mapping jobs: {str(e)}"
        )

@router.get("/terms", response_model=BaseResponse,
          summary="Get all business terms",
          description="Retrieve all available business terms from the index")
async def get_all_terms(
    mapping_service: MappingService = Depends(get_mapping_service)
):
    """
    Get all business terms.
    
    Args:
        mapping_service: Mapping service instance
        
    Returns:
        List of all indexed business terms
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

@router.delete("/jobs/{job_id}", response_model=BaseResponse,
             summary="Delete a mapping job",
             description="Delete a mapping job by its ID")
async def delete_mapping_job(
    job_id: str = Path(..., description="ID of the mapping job to delete"),
    mapping_service: MappingService = Depends(get_mapping_service)
):
    """
    Delete a mapping job.
    
    Args:
        job_id: ID of the job to delete
        mapping_service: Mapping service instance
        
    Returns:
        Success message
    """
    try:
        logger.info(f"Deleting mapping job: {job_id}")
        # Use the job tracking service to delete the job
        await mapping_service.job_tracking_service.connect(mapping_service.es_service.client)
        deleted = await mapping_service.job_tracking_service.delete_job(job_id)
        
        if not deleted:
            logger.warning(f"Mapping job {job_id} not found for deletion")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Mapping job {job_id} not found"
            )
        
        return BaseResponse(
            success=True,
            message=f"Mapping job {job_id} deleted successfully",
            data=None
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting mapping job: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting mapping job: {str(e)}"
        )

@router.get("/health", response_model=BaseResponse,
          summary="Check mapping service health",
          description="Check if the mapping service is healthy and connected to dependencies")
async def check_mapping_health(
    mapping_service: MappingService = Depends(get_mapping_service)
):
    """
    Check if the mapping service is healthy.
    
    Args:
        mapping_service: Mapping service instance
        
    Returns:
        Health status information
    """
    try:
        # Check if we can connect to Elasticsearch
        es_connected = mapping_service.es_service._connected
        if not es_connected:
            try:
                await mapping_service.es_service.connect()
                es_connected = True
            except Exception:
                es_connected = False
        
        # Check job tracking index
        job_index_exists = False
        if es_connected:
            try:
                await mapping_service.job_tracking_service.connect(mapping_service.es_service.client)
                job_index_exists = await mapping_service.job_tracking_service.client.indices.exists(
                    index=mapping_service.job_tracking_service.index_name
                )
            except Exception:
                job_index_exists = False
        
        # Return health status
        return BaseResponse(
            success=True,
            message="Mapping service health check completed",
            data={
                "elasticsearch_connected": es_connected,
                "job_index_exists": job_index_exists,
                "timestamp": datetime.now().isoformat()
            }
        )
    except Exception as e:
        logger.error(f"Error checking mapping service health: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error checking mapping service health: {str(e)}"
        )

@router.post("/refresh-token", response_model=BaseResponse,
           summary="Refresh Azure tokens",
           description="Refresh Azure AD tokens used for service authentication")
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