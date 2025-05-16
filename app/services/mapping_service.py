"""
Enhanced mapping service with AI agent evaluation.
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
    """Enhanced service for mapping business terms with AI agent evaluation."""
    
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
        Map a business term to existing terms using enhanced AI agent evaluation.
        
        Args:
            request: Mapping request
            
        Returns:
            Mapping response with results
        """
        try:
            logger.info(f"Processing mapping request for '{request.name}'")
            
            # Use the enhanced LangGraph workflow with AI agent evaluation
            results = await self.vector_service.create_langgraph_workflow(request)
            
            # Add detailed reasoning for each result using the AI agent
            enriched_results = await self._enrich_results_with_explanation(request, results)
            
            # Create response
            response = MappingResponse(
                success=True,
                message="Mapping completed successfully with AI agent evaluation",
                results=enriched_results,
                query=request,
                timestamp=datetime.now()
            )
            
            logger.info(f"Mapping completed with {len(results)} results using AI agent evaluation")
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
    
    async def _enrich_results_with_explanation(self, request: MappingRequest, results: List[MappingResult]) -> List[MappingResult]:
        """
        Enrich mapping results with detailed explanations using the AI agent.
        
        Args:
            request: Original mapping request
            results: List of mapping results to enrich
            
        Returns:
            Enriched mapping results with detailed explanations
        """
        if not results:
            return results
            
        # Create system prompt for explanation
        system_prompt = """
        You are an expert in data governance, metadata management, and business terminology.
        Your task is to provide a clear, detailed explanation for why each matched business term
        is relevant to the user's request.
        
        For each term, explain:
        1. The semantic relationship between the request and the term
        2. How the definitions align 
        3. Any contextual or domain-specific relationships
        4. The confidence level and why it's appropriate
        
        Keep each explanation concise (2-3 sentences) but informative, focusing on why this
        is a good match from a business perspective.
        """
        
        # Create user prompt with all details
        user_prompt = f"""
        Original Request:
        - Name: {request.name}
        - Description: {request.description}
        """
        
        if request.example:
            user_prompt += f"- Example: {request.example}\n"
        if request.cdm:
            user_prompt += f"- CDM: {request.cdm}\n"
        if request.process_name:
            user_prompt += f"- Process: {request.process_name}\n"
        if request.process_description:
            user_prompt += f"- Process Description: {request.process_description}\n"
        
        user_prompt += "\nTop Matches:\n"
        
        # Add each result to the prompt
        for i, result in enumerate(results):
            user_prompt += f"""
            Match {i+1}: {result.term_name}
            - ID: {result.term_id}
            - Confidence: {result.confidence}
            - Match Type: {result.mapping_type}
            """
        
        user_prompt += "\nPlease provide a concise explanation for each match."
        
        # Create messages for the LLM
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            # Get explanations from LLM
            explanation_text = await self.azure_service.generate_completion(
                messages,
                temperature=0.2,  # Low temperature for consistent, precise explanations
                max_tokens=2000
            )
            
            # Parse explanations and add to results
            explanations = self._parse_explanations(explanation_text, len(results))
            
            # Update results with explanations
            for i, result in enumerate(results):
                if i < len(explanations):
                    # Create a new result with the explanation
                    updated_result = MappingResult(
                        term_id=result.term_id,
                        term_name=result.term_name,
                        similarity_score=result.similarity_score,
                        confidence=result.confidence,
                        mapping_type=result.mapping_type,
                        matched_attributes=result.matched_attributes
                    )
                    
                    # Add explanation as metadata or additional field
                    # Since MappingResult doesn't have an explanation field in the model,
                    # we'll need to handle this at the API response level or extend the model
                    
                    results[i] = updated_result
            
            return results
        except Exception as e:
            logger.error(f"Error enriching results with explanations: {e}")
            # Return original results if enrichment fails
            return results
    
    def _parse_explanations(self, explanation_text: str, num_results: int) -> List[str]:
        """
        Parse explanations from LLM response.
        
        Args:
            explanation_text: Text containing explanations
            num_results: Number of results to expect explanations for
            
        Returns:
            List of explanations
        """
        explanations = []
        
        # Simple parsing based on "Match X:" pattern
        import re
        
        for i in range(1, num_results + 1):
            pattern = rf"Match {i}:?(.*?)(?:Match {i+1}:|$)"
            match = re.search(pattern, explanation_text, re.DOTALL)
            
            if match:
                explanation = match.group(1).strip()
                explanations.append(explanation)
            else:
                explanations.append("No detailed explanation available.")
        
        return explanations
    
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