"""
Service for mapping business terms.
"""

import logging
from typing import List, Dict, Any, Optional
from app.models.mapping import MappingRequest, MappingResponse, MappingResult
from app.services.azure_openai import AzureOpenAIService
from app.services.elasticsearch_service import ElasticsearchService
from app.services.vector_service import VectorService
from datetime import datetime

logger = logging.getLogger(__name__)

class MappingService:
    """Service for mapping business terms."""
    
    def __init__(self, 
                azure_service: AzureOpenAIService,
                es_service: ElasticsearchService,
                vector_service: VectorService):
        """
        Initialize the Mapping Service.
        
        Args:
            azure_service: Azure OpenAI service
            es_service: Elasticsearch service
            vector_service: Vector service
        """
        self.azure_service = azure_service
        self.es_service = es_service
        self.vector_service = vector_service
    
    async def map_business_term(self, request: MappingRequest) -> MappingResponse:
        """
        Map a business term to existing terms.
        
        Args:
            request: Mapping request
            
        Returns:
            Mapping response with results
        """
        try:
            logger.info(f"Processing mapping request for '{request.name}'")
            
            # Use the LangGraph workflow for hybrid search
            results = await self.vector_service.create_langgraph_workflow(request)
            
            # Create response
            response = MappingResponse(
                success=True,
                message="Mapping completed successfully",
                results=results,
                query=request,
                timestamp=datetime.now()
            )
            
            logger.info(f"Mapping completed with {len(results)} results")
            return response
        except Exception as e:
            logger.error(f"Error mapping business term: {e}")
            
            # Return error response
            return MappingResponse(
                success=False,
                message=f"Error mapping business term: {str(e)}",
                results=[],
                query=request,
                timestamp=datetime.now()
            )
    
    async def get_all_business_terms(self) -> List[Dict[str, Any]]:
        """
        Get all indexed business terms.
        
        Returns:
            List of business terms
        """
        try:
            logger.info("Retrieving all business terms")
            terms = await self.es_service.get_all_documents()
            logger.info(f"Retrieved {len(terms)} business terms")
            return terms
        except Exception as e:
            logger.error(f"Error retrieving business terms: {e}")
            raise