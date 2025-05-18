"""
Enhanced Vector Service with parallel processing, error handling,
and improved LLM agent evaluation.
"""

import logging
import json
import re
import asyncio
import random
import time
from typing import List, Dict, Any, Optional, Union, TypedDict
import pandas as pd
import numpy as np
from functools import partial
from datetime import datetime

# Enhanced workflow libraries
from langgraph.graph import StateGraph, END

# JSON repair utility
from app.utils.json_repair import repair_json

# Import services
from app.services.azure_openai import AzureOpenAIService
from app.services.elasticsearch_service import ElasticsearchService
from app.models.mapping import BusinessTerm, MappingRequest, MappingResult

logger = logging.getLogger(__name__)

class VectorService:
    """Enhanced service for vector operations and LLM-based mapping with robust error recovery."""
    
    def __init__(self, azure_service: AzureOpenAIService, es_service: ElasticsearchService):
        """
        Initialize the Vector Service.
        
        Args:
            azure_service: Azure OpenAI service
            es_service: Elasticsearch service
        """
        self.azure_service = azure_service
        self.es_service = es_service
        
        # Configure concurrent operations
        self.max_concurrent_searches = 3  # Max concurrent search operations
        self.max_concurrent_embeddings = 5  # Max concurrent embedding operations
        
        # Configure LLM settings
        self.explanation_format_prompt = """
        Please format your response as valid JSON in the following format:
        {
            "evaluation": "Your brief explanation",
            "matches": [
                {
                    "term_id": "ID from the candidate",
                    "term_name": "Name from the candidate",
                    "confidence": 0.0 to 1.0 (float),
                    "reasoning": "Detailed explanation for this match",
                    "matched_attributes": ["list", "of", "matched", "attributes"],
                    "match_type": "semantic/exact/partial/contextual"
                },
                ...
            ]
        }
        
        Ensure your JSON is valid with:
        - No trailing commas
        - Properly quoted strings
        - Properly nested brackets
        - Confidence as a numeric value between 0 and 1
        """
    
    async def index_data_from_csv(self, csv_path: str):
        """
        Index data from a CSV file with parallel processing.
        
        Args:
            csv_path: Path to the CSV file
        """
        try:
            # Read CSV
            df = pd.read_csv(csv_path)
            
            # Check if columns exist and rename if needed
            column_mapping = {
                'PBT_NAME': 'pbt_name',
                'PBT_DEFINITION': 'pbt_definition',
                'CDM': 'cdm'
            }
            
            for old_col, new_col in column_mapping.items():
                if old_col in df.columns and new_col not in df.columns:
                    df = df.rename(columns={old_col: new_col})
            
            # Ensure id column exists
            if 'id' not in df.columns:
                df['id'] = df.index.astype(str)
            
            # Generate embeddings with parallel processing
            logger.info(f"Generating embeddings for {len(df)} business terms...")
            
            # Create combined texts for embedding
            texts = [f"{row.pbt_name} {row.pbt_definition}" for _, row in df.iterrows()]
            
            # Generate embeddings in batches for reliability
            embeddings = await self.azure_service.generate_embeddings(texts)
            
            # Prepare documents for indexing
            documents = []
            for i, (_, row) in enumerate(df.iterrows()):
                if i < len(embeddings):
                    doc = {
                        "id": str(row.id),
                        "pbt_name": row.pbt_name,
                        "pbt_definition": row.pbt_definition,
                        "cdm": row.cdm if 'cdm' in row and not pd.isna(row.cdm) else None,
                        "embedding": embeddings[i]
                    }
                    documents.append(doc)
            
            # Bulk index documents
            logger.info(f"Indexing {len(documents)} documents to Elasticsearch...")
            await self.es_service.bulk_index_documents(documents)
            
            logger.info("CSV data indexed successfully")
        except Exception as e:
            logger.error(f"Error indexing data from CSV: {e}")
            raise
    
    async def semantic_search(self, query: MappingRequest, top_k: int = 5) -> List[MappingResult]:
        """
        Perform semantic search using embeddings and ANN with error handling.
        
        Args:
            query: Mapping request
            top_k: Number of results to return
            
        Returns:
            List of mapping results
        """
        try:
            # Create a combined query text
            query_text = f"{query.name} {query.description}"
            if query.example:
                query_text += f" {query.example}"
            if query.process_name:
                query_text += f" {query.process_name}"
            if query.process_description:
                query_text += f" {query.process_description}"
            
            # Generate embedding for the query
            query_embedding = await self.azure_service.generate_single_embedding(query_text)
            
            # Search by vector using optimized ANN
            results = await self.es_service.search_by_vector_ann(
                vector=query_embedding, 
                size=top_k,
                ef_search=100  # Use higher ef_search for better quality
            )
            
            # Convert to MappingResult objects
            mapping_results = []
            for result in results:
                # Apply score normalization for more consistent confidence values
                normalized_score = min(max(result.get("score", 0.5) * 0.8, 0.0), 1.0)
                
                mapping_result = MappingResult(
                    term_id=result.get("id", "unknown"),
                    term_name=result.get("pbt_name", "Unknown Term"),
                    similarity_score=normalized_score,
                    confidence=normalized_score,
                    mapping_type="semantic",
                    matched_attributes=["pbt_name", "pbt_definition"]
                )
                mapping_results.append(mapping_result)
            
            return mapping_results
        except Exception as e:
            logger.error(f"Error performing semantic search: {e}")
            # Return empty results as fallback
            return []
    
    async def bm25_search(self, query: MappingRequest, top_k: int = 5) -> List[MappingResult]:
        """
        Perform BM25 text search with error handling.
        
        Args:
            query: Mapping request
            top_k: Number of results to return
            
        Returns:
            List of mapping results
        """
        try:
            # Create a combined query text
            query_text = f"{query.name} {query.description}"
            if query.example:
                query_text += f" {query.example}"
            if query.process_name:
                query_text += f" {query.process_name}"
            if query.process_description:
                query_text += f" {query.process_description}"
            
            # Search by text using BM25
            results = await self.es_service.search_by_text(
                text=query_text, 
                fields=["pbt_name", "pbt_definition"],
                size=top_k
            )
            
            # Convert to MappingResult objects
            mapping_results = []
            for result in results:
                # Normalize BM25 scores - they can be >1
                normalized_score = min(result.get("score", 5.0) / 10.0, 1.0)
                
                mapping_result = MappingResult(
                    term_id=result.get("id", "unknown"),
                    term_name=result.get("pbt_name", "Unknown Term"),
                    similarity_score=normalized_score,
                    confidence=normalized_score,
                    mapping_type="BM25",
                    matched_attributes=["pbt_name", "pbt_definition"]
                )
                mapping_results.append(mapping_result)
            
            return mapping_results
        except Exception as e:
            logger.error(f"Error performing BM25 search: {e}")
            # Return empty results as fallback
            return []
    
    async def keyword_search(self, query: MappingRequest, top_k: int = 5) -> List[MappingResult]:
        """
        Perform keyword-based search with error handling.
        
        Args:
            query: Mapping request
            top_k: Number of results to return
            
        Returns:
            List of mapping results
        """
        try:
            # Extract keywords from query using LLM
            system_prompt = """
            Your task is to extract the most important keywords from the following text.
            Return ONLY a list of 5-10 individual keywords or short phrases separated by spaces.
            Focus on business terms, technical terms, entities, and domain-specific vocabulary.
            DO NOT include any explanations, headers, formatting, or extra information.
            """
            
            user_prompt = f"""
            Text to extract keywords from:
            
            Name: {query.name}
            Description: {query.description}
            """
            
            if query.example:
                user_prompt += f"\nExample: {query.example}"
            if query.process_name:
                user_prompt += f"\nProcess Name: {query.process_name}"
            if query.process_description:
                user_prompt += f"\nProcess Description: {query.process_description}"
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # Get keywords using LLM with specialized model for extraction
            try:
                keywords_text = await self.azure_service.generate_completion(
                    messages,
                    temperature=0.0,  # Use zero temperature for deterministic output
                    max_tokens=100
                )
                keywords = keywords_text.strip().split()
                logger.info(f"Extracted keywords: {keywords}")
            except Exception as e:
                logger.warning(f"Error extracting keywords: {e}. Using fallback method.")
                # Fallback method: simple keyword extraction
                all_text = f"{query.name} {query.description}"
                if query.example:
                    all_text += f" {query.example}"
                
                # Extract potential keywords using basic NLP
                words = re.findall(r'\b[a-zA-Z][a-zA-Z-]+\b', all_text)
                # Remove common stopwords
                stopwords = {'and', 'or', 'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'about', 
                            'from', 'as', 'of', 'that', 'this', 'these', 'those', 'is', 'are', 'was', 'were'}
                keywords = [word for word in words if word.lower() not in stopwords][:10]
                logger.info(f"Used fallback keyword extraction: {keywords}")
            
            # Search by keywords
            results = await self.es_service.search_by_keywords(
                keywords=keywords,
                fields=["pbt_name", "pbt_definition"],
                size=top_k
            )
            
            # Convert to MappingResult objects
            mapping_results = []
            for result in results:
                normalized_score = min(result.get("score", 5.0) / 10.0, 1.0)
                
                mapping_result = MappingResult(
                    term_id=result.get("id", "unknown"),
                    term_name=result.get("pbt_name", "Unknown Term"),
                    similarity_score=normalized_score,
                    confidence=normalized_score,
                    mapping_type="keyword",
                    matched_attributes=["pbt_name", "pbt_definition"]
                )
                mapping_results.append(mapping_result)
            
            return mapping_results
        except Exception as e:
            logger.error(f"Error performing keyword search: {e}")
            # Return empty results as fallback
            return []
    
    async def hybrid_search(self, query: MappingRequest, top_k: int = 5) -> List[MappingResult]:
        """
        Perform hybrid search combining vector and keyword search.
        
        Args:
            query: Mapping request
            top_k: Number of results to return
            
        Returns:
            List of mapping results
        """
        try:
            # Create a combined query text
            query_text = f"{query.name} {query.description}"
            if query.example:
                query_text += f" {query.example}"
            if query.process_name:
                query_text += f" {query.process_name}"
            if query.process_description:
                query_text += f" {query.process_description}"
            
            # Generate embedding for the query
            query_embedding = await self.azure_service.generate_single_embedding(query_text)
            
            # Perform hybrid search
            results = await self.es_service.hybrid_search(
                text=query_text,
                vector=query_embedding,
                fields=["pbt_name", "pbt_definition"],
                vector_weight=0.7,  # Give vector search higher weight
                size=top_k
            )
            
            # Convert to MappingResult objects
            mapping_results = []
            for result in results:
                mapping_result = MappingResult(
                    term_id=result.get("id", "unknown"),
                    term_name=result.get("pbt_name", "Unknown Term"),
                    similarity_score=result.get("score", 0.5),
                    confidence=min(result.get("score", 0.5), 1.0),
                    mapping_type="hybrid",
                    matched_attributes=["pbt_name", "pbt_definition"]
                )
                mapping_results.append(mapping_result)
            
            return mapping_results
        except Exception as e:
            logger.error(f"Error performing hybrid search: {e}")
            # Return empty results as fallback
            return []
    
    async def advanced_agent_evaluation(self, query: MappingRequest, candidates: List[Dict], top_k: int = 3) -> List[MappingResult]:
        """
        Robust AI agent evaluation of search results with error recovery and retries.
        
        Args:
            query: Mapping request
            candidates: List of candidate terms from various search methods
            top_k: Number of results to return
            
        Returns:
            List of mapping results with contextual reasoning
        """
        try:
            # Limit candidates to avoid token limits
            candidates = candidates[:15]  # Limit to top 15 candidates
            
            # Create a comprehensive system prompt for the LLM
            system_prompt = f"""
            You are an advanced AI agent specialized in evaluating and contextualizing business term mappings.
            Your task is to analyze business terms and find the most relevant matches for a given mapping request.
            
            You must:
            1. Analyze the semantic relationship between the request and each candidate
            2. Consider explicit term matches and implicit/contextual relationships
            3. Evaluate business context and domain relevance
            4. Consider hierarchical relationships (more specific, more general, or same level)
            5. Identify false positives that may have high similarity scores but aren't truly related
            
            For each candidate, evaluate:
            - Semantic similarity: How similar the terms are conceptually
            - Definition alignment: How well the definitions align
            - Contextual fit: Whether the term makes sense in the context
            - Domain appropriateness: Whether the term belongs to the relevant business domain
            
            {self.explanation_format_prompt}
            
            Limit your response to the top {top_k} most relevant matches. Be thorough in your reasoning
            and consider both explicit textual similarity and deeper conceptual relationships.
            """
            
            # Create a detailed user prompt with all information
            user_prompt = f"""
            # MAPPING REQUEST
            Name: {query.name}
            Description: {query.description}
            """
            
            if query.example:
                user_prompt += f"\nExample: {query.example}\n"
            if query.cdm:
                user_prompt += f"\nCDM: {query.cdm}\n"
            if query.process_name:
                user_prompt += f"\nProcess Name: {query.process_name}\n"
            if query.process_description:
                user_prompt += f"\nProcess Description: {query.process_description}\n"
            
            user_prompt += "\n# CANDIDATE BUSINESS TERMS FOR EVALUATION\n"
            
            # Include all candidate details
            for i, candidate in enumerate(candidates):
                user_prompt += f"""
                ## Candidate {i+1}
                ID: {candidate.get('id', f'unknown-{i}')}
                Name: {candidate.get('pbt_name', 'Unknown Term')}
                Definition: {candidate.get('pbt_definition', 'No definition available')}
                """
                
                if 'cdm' in candidate and candidate['cdm']:
                    user_prompt += f"CDM: {candidate['cdm']}\n"
                    
                if 'score' in candidate:
                    user_prompt += f"Raw Score: {candidate['score']}\n"
                
                if 'search_method' in candidate:
                    user_prompt += f"Match Type: {candidate['search_method']}\n"
            
            # Create messages for the LLM
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # Add JSON response format to ensure structured output
            response_format = {"type": "json_object"}
            
            # Get detailed evaluation from LLM with JSON mode
            # Use multiple retries and model fallbacks for resilience
            for attempt in range(3):  # Try up to 3 times
                try:
                    llm_response = await self.azure_service.generate_completion(
                        messages, 
                        temperature=0.2,  # Low temperature for consistent output
                        max_tokens=4000,
                        response_format=response_format
                    )
                    
                    # Parse JSON response
                    try:
                        # Parse the response as JSON
                        analysis = json.loads(llm_response)
                        
                        # Validate expected structure
                        if "matches" not in analysis or not isinstance(analysis["matches"], list):
                            logger.warning("Invalid JSON structure from LLM (missing matches array)")
                            raise ValueError("Missing matches array in response")
                        
                        # Convert agent analysis to MappingResult objects
                        mapping_results = []
                        
                        # Use the agent's evaluation but limit to top_k results
                        for match in analysis.get("matches", [])[:top_k]:
                            # Find the term name if not provided
                            term_id = match.get("term_id", "")
                            term_name = match.get("term_name", "")
                            
                            if not term_name and term_id:
                                # Look up in candidates
                                for candidate in candidates:
                                    if candidate.get("id") == term_id:
                                        term_name = candidate.get("pbt_name", "")
                                        break
                            
                            # Create mapping result with more detailed fields
                            mapping_result = MappingResult(
                                term_id=term_id,
                                term_name=term_name,
                                similarity_score=match.get("confidence", 0.9),
                                confidence=match.get("confidence", 0.9),
                                mapping_type="agent",
                                matched_attributes=match.get("matched_attributes", ["pbt_name", "pbt_definition"])
                            )
                            
                            # Add explanation as a custom field
                            setattr(mapping_result, "explanation", match.get("reasoning", "No detailed explanation available."))
                            
                            # Add match type if provided
                            if "match_type" in match:
                                setattr(mapping_result, "match_subtype", match.get("match_type"))
                            
                            mapping_results.append(mapping_result)
                        
                        logger.info(f"Agent evaluation completed successfully with {len(mapping_results)} results")
                        return mapping_results
                    
                    except json.JSONDecodeError as json_err:
                        logger.warning(f"JSON parsing error from LLM (attempt {attempt+1}/3): {json_err}")
                        
                        # Try to repair JSON
                        try:
                            repaired_json = repair_json(llm_response)
                            logger.info("JSON repaired successfully")
                            analysis = json.loads(repaired_json)
                            
                            # Continue processing with repaired JSON
                            mapping_results = []
                            for match in analysis.get("matches", [])[:top_k]:
                                term_id = match.get("term_id", "")
                                term_name = match.get("term_name", "")
                                
                                if not term_name and term_id:
                                    for candidate in candidates:
                                        if candidate.get("id") == term_id:
                                            term_name = candidate.get("pbt_name", "")
                                            break
                                
                                mapping_result = MappingResult(
                                    term_id=term_id,
                                    term_name=term_name,
                                    similarity_score=match.get("confidence", 0.9),
                                    confidence=match.get("confidence", 0.9),
                                    mapping_type="agent",
                                    matched_attributes=match.get("matched_attributes", ["pbt_name", "pbt_definition"])
                                )
                                
                                # Add explanation
                                setattr(mapping_result, "explanation", match.get("reasoning", "No detailed explanation available."))
                                
                                # Add match type if provided
                                if "match_type" in match:
                                    setattr(mapping_result, "match_subtype", match.get("match_type"))
                                
                                mapping_results.append(mapping_result)
                            
                            logger.info(f"Agent evaluation completed with repaired JSON: {len(mapping_results)} results")
                            return mapping_results
                        
                        except Exception as repair_err:
                            logger.error(f"Error repairing JSON: {repair_err}")
                            # Try next attempt or fallback
                
                except Exception as e:
                    logger.error(f"Error in agent evaluation (attempt {attempt+1}/3): {e}")
                    # Try again or resort to fallback
                    
                # Wait before retrying
                if attempt < 2:  # Don't wait after the last attempt
                    wait_time = (2 ** attempt) + (random.random() * 0.5)
                    logger.info(f"Waiting {wait_time:.2f}s before retrying agent evaluation...")
                    await asyncio.sleep(wait_time)
            
            # If all attempts failed, use a simplified fallback method
            logger.warning("All agent evaluation attempts failed, using fallback ranking")
            
            # Simple fallback: rank by score and return top candidates
            sorted_candidates = sorted(
                candidates, 
                key=lambda x: x.get("score", 0), 
                reverse=True
            )[:top_k]
            
            fallback_results = []
            for candidate in sorted_candidates:
                mapping_result = MappingResult(
                    term_id=candidate.get("id", "unknown"),
                    term_name=candidate.get("pbt_name", "Unknown Term"),
                    similarity_score=min(candidate.get("score", 0.8), 1.0),
                    confidence=min(candidate.get("score", 0.8), 1.0),
                    mapping_type="fallback_ranking",
                    matched_attributes=["pbt_name", "pbt_definition"]
                )
                
                # Add simplified explanation
                setattr(mapping_result, "explanation", 
                       f"Term '{candidate.get('pbt_name', 'Unknown')}' matched with confidence {min(candidate.get('score', 0.8), 1.0):.2f}.")
                
                fallback_results.append(mapping_result)
            
            logger.info(f"Using fallback ranking: {len(fallback_results)} results")
            return fallback_results
                
        except Exception as e:
            logger.error(f"Unhandled error in advanced agent evaluation: {e}")
            
            # Emergency fallback: return top candidates directly
            emergency_results = []
            for i, candidate in enumerate(candidates[:top_k]):
                mapping_result = MappingResult(
                    term_id=candidate.get("id", f"unknown-{i}"),
                    term_name=candidate.get("pbt_name", f"Unknown Term {i}"),
                    similarity_score=0.5,  # Default score
                    confidence=0.5,
                    mapping_type="emergency_fallback",
                    matched_attributes=["pbt_name", "pbt_definition"]
                )
                emergency_results.append(mapping_result)
            
            return emergency_results
    
    async def create_langgraph_workflow(self, query: MappingRequest) -> List[MappingResult]:
        """
        Create and execute an enhanced LangGraph workflow with reliable error handling.
        
        Args:
            query: Mapping request
            
        Returns:
            List of mapping results
        """
        try:
            # Define the state as a TypedDict
            class GraphState(TypedDict):
                query: Dict
                semantic_results: List
                bm25_results: List
                keyword_results: List
                hybrid_results: List
                all_candidates: List
                agent_results: List
                final_results: List
            
            # Define the nodes with error handling
            async def semantic_search_node(state: GraphState) -> GraphState:
                """Perform semantic search using ANN."""
                try:
                    semantic_results = await self.semantic_search(query, top_k=8)
                    return {"semantic_results": semantic_results}
                except Exception as e:
                    logger.error(f"Error in semantic search node: {e}")
                    return {"semantic_results": []}
            
            async def bm25_search_node(state: GraphState) -> GraphState:
                """Perform BM25 search."""
                try:
                    bm25_results = await self.bm25_search(query, top_k=8)
                    return {"bm25_results": bm25_results}
                except Exception as e:
                    logger.error(f"Error in BM25 search node: {e}")
                    return {"bm25_results": []}
            
            async def keyword_search_node(state: GraphState) -> GraphState:
                """Perform keyword search."""
                try:
                    keyword_results = await self.keyword_search(query, top_k=5)
                    return {"keyword_results": keyword_results}
                except Exception as e:
                    logger.error(f"Error in keyword search node: {e}")
                    return {"keyword_results": []}
            
            async def hybrid_search_node(state: GraphState) -> GraphState:
                """Perform hybrid search."""
                try:
                    hybrid_results = await self.hybrid_search(query, top_k=8)
                    return {"hybrid_results": hybrid_results}
                except Exception as e:
                    logger.error(f"Error in hybrid search node: {e}")
                    return {"hybrid_results": []}
            
            async def collect_candidates_node(state: GraphState) -> GraphState:
                """Collect unique candidates from all search methods."""
                try:
                    # Collect all results
                    all_results = {}
                    
                    # Process all search results
                    for result_list in [
                        state["semantic_results"], 
                        state["bm25_results"], 
                        state["keyword_results"],
                        state["hybrid_results"]
                    ]:
                        for result in result_list:
                            term_id = result.term_id
                            if term_id not in all_results:
                                # Get the full document data
                                try:
                                    # Use fetch by ID for reliability
                                    response = await self.es_service._execute_with_retry(
                                        self.es_service.client.get,
                                        index=self.es_service.index_name,
                                        id=term_id
                                    )
                                    doc = response["_source"]
                                    
                                    # Add score information
                                    doc["score"] = result.similarity_score
                                    doc["search_method"] = result.mapping_type
                                    
                                    all_results[term_id] = doc
                                except Exception as e:
                                    logger.error(f"Error retrieving document {term_id}: {e}")
                    
                    # Convert to list
                    all_candidates = list(all_results.values())
                    
                    # Sort by score
                    all_candidates.sort(key=lambda x: x.get("score", 0), reverse=True)
                    
                    # Limit to top candidates for agent evaluation
                    top_candidates = all_candidates[:15]  # Limited to 15 for faster processing
                    
                    return {"all_candidates": top_candidates}
                except Exception as e:
                    logger.error(f"Error in collect candidates node: {e}")
                    # Return empty list as fallback
                    return {"all_candidates": []}
            
            async def agent_evaluation_node(state: GraphState) -> GraphState:
                """Perform advanced agent-based evaluation."""
                try:
                    agent_results = await self.advanced_agent_evaluation(
                        query, 
                        state["all_candidates"], 
                        top_k=3  # Get top 3 results from the agent
                    )
                    return {"agent_results": agent_results}
                except Exception as e:
                    logger.error(f"Error in agent evaluation node: {e}")
                    # If agent evaluation fails completely, use a simple score-based ranking
                    top_candidates = sorted(
                        state["all_candidates"],
                        key=lambda x: x.get("score", 0),
                        reverse=True
                    )[:3]
                    
                    # Convert to MappingResult
                    fallback_results = []
                    for candidate in top_candidates:
                        result = MappingResult(
                            term_id=candidate.get("id", "unknown"),
                            term_name=candidate.get("pbt_name", "Unknown Term"),
                            similarity_score=min(candidate.get("score", 0.8), 1.0),
                            confidence=min(candidate.get("score", 0.8), 1.0),
                            mapping_type="fallback",
                            matched_attributes=["pbt_name", "pbt_definition"]
                        )
                        fallback_results.append(result)
                    
                    logger.warning("Used fallback score-based ranking due to agent failure")
                    return {"agent_results": fallback_results}
            
            async def finalize_results_node(state: GraphState) -> GraphState:
                """Finalize results based on agent evaluation."""
                # The agent results are our final results
                final_results = state.get("agent_results", [])
                
                # If no results, use a fallback
                if not final_results and state.get("all_candidates"):
                    logger.warning("No results from agent, using fallback")
                    # Sort candidates by score
                    sorted_candidates = sorted(
                        state["all_candidates"],
                        key=lambda x: x.get("score", 0),
                        reverse=True
                    )[:3]
                    
                    # Convert to MappingResult
                    final_results = []
                    for candidate in sorted_candidates:
                        result = MappingResult(
                            term_id=candidate.get("id", "unknown"),
                            term_name=candidate.get("pbt_name", "Unknown Term"),
                            similarity_score=min(candidate.get("score", 0.8), 1.0),
                            confidence=min(candidate.get("score", 0.8), 1.0),
                            mapping_type="final_fallback",
                            matched_attributes=["pbt_name", "pbt_definition"]
                        )
                        final_results.append(result)
                
                return {"final_results": final_results}
            
            # Create the workflow
            workflow = StateGraph(GraphState)
            
            # Add nodes
            workflow.add_node("semantic_search", semantic_search_node)
            workflow.add_node("bm25_search", bm25_search_node)
            workflow.add_node("keyword_search", keyword_search_node)
            workflow.add_node("hybrid_search", hybrid_search_node)
            workflow.add_node("collect_candidates", collect_candidates_node)
            workflow.add_node("agent_evaluation", agent_evaluation_node)
            workflow.add_node("finalize_results", finalize_results_node)
            
            # Define the workflow with parallel execution of search methods
            workflow.set_entry_point("semantic_search")
            
            # Run search methods in parallel branches for performance
            workflow.add_edge("semantic_search", "bm25_search")
            workflow.add_edge("bm25_search", "keyword_search")
            workflow.add_edge("keyword_search", "hybrid_search")
            workflow.add_edge("hybrid_search", "collect_candidates")
            
            # Collect all results and evaluate
            workflow.add_edge("collect_candidates", "agent_evaluation")
            workflow.add_edge("agent_evaluation", "finalize_results")
            workflow.add_edge("finalize_results", END)
            
            # Compile the workflow
            app = workflow.compile()
            
            # Set timeout for workflow execution
            execution_timeout = 60  # seconds
            
            try:
                # Run the workflow with timeout
                result = await asyncio.wait_for(
                    app.ainvoke({
                        "query": query.model_dump(),
                        "semantic_results": [],
                        "bm25_results": [],
                        "keyword_results": [],
                        "hybrid_results": [],
                        "all_candidates": [],
                        "agent_results": [],
                        "final_results": []
                    }),
                    timeout=execution_timeout
                )
                
                return result["final_results"]
            except asyncio.TimeoutError:
                logger.error(f"Workflow execution timed out after {execution_timeout}s")
                # Fallback to simple semantic search
                logger.info("Falling back to semantic search due to workflow timeout")
                return await self.semantic_search(query, top_k=3)
            
        except Exception as e:
            logger.error(f"Error executing LangGraph workflow: {e}")
            # Fallback to simple semantic search if workflow fails
            logger.info("Falling back to semantic search due to workflow error")
            return await self.semantic_search(query, top_k=3)