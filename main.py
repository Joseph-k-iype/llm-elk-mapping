"""
Main application entry point.
"""

import os
import logging
import uvicorn
import contextlib
import asyncio
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, List, Any
from app.api.routers import api_router
from app.config.settings import get_settings
from app.core.logging_config import configure_logging, log_request
from app.core.auth_helper import start_token_refresh_service

logger = logging.getLogger(__name__)

def create_app() -> FastAPI:
    """
    Create the FastAPI application.
    
    Returns:
        FastAPI: The FastAPI application
    """
    # Get settings
    settings = get_settings()
    
    # Configure logging
    configure_logging(settings.logging.model_dump())
    
    # Create FastAPI app
    app = FastAPI(
        title=settings.app_name,
        version=settings.version,
        debug=settings.debug,
        openapi_url="/api/openapi.json",
        docs_url="/api/docs",
        redoc_url="/api/redoc"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_hosts,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Add logging middleware
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start_time = asyncio.get_event_loop().time()
        
        # Process the request
        response = await call_next(request)
        
        # Calculate duration
        duration_ms = (asyncio.get_event_loop().time() - start_time) * 1000
        
        # Log the request
        log_request(request.method, request.url.path, response.status_code, duration_ms)
        
        return response
    
    # Add error handling middleware
    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        logger.exception(f"Unhandled exception: {exc}")
        return JSONResponse(
            status_code=500,
            content={"detail": f"Internal server error: {str(exc)}"}
        )
    
    # Include API router
    app.include_router(api_router, prefix="/api")
    
    # Start token refresh service
    if settings.environment != "test":
        token_refresh_thread = start_token_refresh_service()
        logger.info("Azure token refresh service started")
    
    logger.info(f"{settings.app_name} v{settings.version} started in {settings.environment} mode")
    return app

app = create_app()

if __name__ == "__main__":
    # Run the application using Uvicorn
    settings = get_settings()
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug
    )