"""
Enhanced vector service with AI agent for contextual evaluation.
"""

import logging
import json
from typing import List, Dict, Any, Optional, Union, TypedDict
import pandas as pd
import numpy as np
from langgraph.graph import StateGraph, END
from app.services.azure_openai import AzureOpenAIService
from app.services.elasticsearch_service import ElasticsearchService
from app.models.mapping import BusinessTerm, MappingRequest, MappingResult

logger = logging.getLogger(__name__)

class VectorService:
    """Service for vector operations and LLM-based mapping with enhanced AI agent capabilities."""
    
    def __init__(self, azure_service: AzureOpenAIService, es_service: ElasticsearchService):
        """
        Initialize the Vector Service.
        
        Args:
            azure_service: Azure OpenAI service
            es_service: Elasticsearch service
        """
        self.azure_service = azure_service
        self.es_service = es_service
    
    async def index_data_from_csv(self, csv_path: str):
        """
        Index data from a CSV file.
        
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
            
            # Generate embeddings
            logger.info("Generating embeddings for business terms...")
            texts = [f"{row.pbt_name} {row.pbt_definition}" for _, row in df.iterrows()]
            embeddings = await self.azure_service.generate_embeddings(texts)
            
            # Prepare documents for indexing
            documents = []
            for i, (_, row) in enumerate(df.iterrows()):
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
        Perform semantic search using embeddings and ANN.
        
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
            
            # Generate embedding for the query
            query_embedding = await self.azure_service.generate_single_embedding(query_text)
            
            # Search by vector using optimized ANN
            results = await self.es_service.search_by_vector_ann(vector=query_embedding, size=top_k)
            
            # Convert to MappingResult objects
            mapping_results = []
            for result in results:
                mapping_result = MappingResult(
                    term_id=result["id"],
                    term_name=result["pbt_name"],
                    similarity_score=result["score"],
                    confidence=min(result["score"] * 1.5, 1.0),  # Scale and cap confidence
                    mapping_type="semantic",
                    matched_attributes=["pbt_name", "pbt_definition"]
                )
                mapping_results.append(mapping_result)
            
            return mapping_results
        except Exception as e:
            logger.error(f"Error performing semantic search: {e}")
            raise
    
    async def bm25_search(self, query: MappingRequest, top_k: int = 5) -> List[MappingResult]:
        """
        Perform BM25 text search.
        
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
            
            # Search by text using BM25
            results = await self.es_service.search_by_text(
                text=query_text, 
                fields=["pbt_name", "pbt_definition"],
                size=top_k
            )
            
            # Convert to MappingResult objects
            mapping_results = []
            for result in results:
                # Normalize BM25 scores (they can be > 1)
                normalized_score = min(result["score"] / 10.0, 1.0)
                
                mapping_result = MappingResult(
                    term_id=result["id"],
                    term_name=result["pbt_name"],
                    similarity_score=normalized_score,
                    confidence=normalized_score,
                    mapping_type="BM25",
                    matched_attributes=["pbt_name", "pbt_definition"]
                )
                mapping_results.append(mapping_result)
            
            return mapping_results
        except Exception as e:
            logger.error(f"Error performing BM25 search: {e}")
            raise
    
    async def keyword_search(self, query: MappingRequest, top_k: int = 5) -> List[MappingResult]:
        """
        Perform keyword-based search.
        
        Args:
            query: Mapping request
            top_k: Number of results to return
            
        Returns:
            List of mapping results
        """
        try:
            # Extract keywords from query
            system_prompt = """
            Extract the most important keywords from the following text. 
            Return only a list of keywords, separated by spaces.
            Focus on business and technical terms, entities, and domain-specific vocabulary.
            """
            
            user_prompt = f"""
            Text to extract keywords from:
            
            Name: {query.name}
            Description: {query.description}
            """
            
            if query.example:
                user_prompt += f"\nExample: {query.example}"
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # Get keywords using LLM
            keywords_text = await self.azure_service.generate_completion(messages)
            keywords = keywords_text.strip().split()
            
            # Search by keywords
            results = await self.es_service.search_by_keywords(
                keywords=keywords,
                fields=["pbt_name", "pbt_definition"],
                size=top_k
            )
            
            # Convert to MappingResult objects
            mapping_results = []
            for result in results:
                normalized_score = min(result["score"] / 10.0, 1.0)
                
                mapping_result = MappingResult(
                    term_id=result["id"],
                    term_name=result["pbt_name"],
                    similarity_score=normalized_score,
                    confidence=normalized_score,
                    mapping_type="keyword",
                    matched_attributes=["pbt_name", "pbt_definition"]
                )
                mapping_results.append(mapping_result)
            
            return mapping_results
        except Exception as e:
            logger.error(f"Error performing keyword search: {e}")
            raise
    
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
            
            # Generate embedding for the query
            query_embedding = await self.azure_service.generate_single_embedding(query_text)
            
            # Perform hybrid search
            results = await self.es_service.hybrid_search(
                text=query_text,
                vector=query_embedding,
                fields=["pbt_name", "pbt_definition"],
                vector_weight=0.7,
                size=top_k
            )
            
            # Convert to MappingResult objects
            mapping_results = []
            for result in results:
                mapping_result = MappingResult(
                    term_id=result["id"],
                    term_name=result["pbt_name"],
                    similarity_score=result["score"],
                    confidence=min(result["score"], 1.0),
                    mapping_type="hybrid",
                    matched_attributes=["pbt_name", "pbt_definition"]
                )
                mapping_results.append(mapping_result)
            
            return mapping_results
        except Exception as e:
            logger.error(f"Error performing hybrid search: {e}")
            raise
    
    async def advanced_agent_evaluation(self, query: MappingRequest, candidates: List[Dict], top_k: int = 3) -> List[MappingResult]:
        """
        Enhanced AI agent for contextual evaluation of search results.
        This agent evaluates candidates based on semantic relevance and business context,
        and returns the top matches with detailed reasoning.
        
        Args:
            query: Mapping request
            candidates: List of candidate terms from various search methods
            top_k: Number of results to return (default: 3)
            
        Returns:
            List of mapping results with contextual reasoning
        """
        try:
            # Create a comprehensive system prompt for the LLM that acts as an AI agent
            system_prompt = """
            You are an advanced AI agent specialized in evaluating and contextualizing business term mappings.
            Your task is to analyze business terms and their definitions to find the most relevant matches
            for a given mapping request.
            
            You must:
            1. Deeply analyze the semantic relationship between the request and each candidate
            2. Consider both explicit term matches and implicit/contextual relationships
            3. Evaluate the business context and domain relevance
            4. Consider hierarchical relationships (is the term more specific, more general, or at the same level?)
            5. Identify false positives that may have high similarity scores but aren't truly related
            
            For each candidate, evaluate:
            - Semantic similarity: How similar the terms are conceptually
            - Definition alignment: How well the definitions align
            - Contextual fit: Whether the term makes sense in the provided context
            - Domain appropriateness: Whether the term belongs to the relevant business domain
            
            Return your analysis in the following JSON format:
            
            {
                "evaluation": "A brief explanation of how you approached the evaluation",
                "matches": [
                    {
                        "term_id": "id of the term",
                        "term_name": "name of the term",
                        "confidence": 0.0-1.0,
                        "reasoning": "A detailed explanation of why this term is a good match",
                        "matched_attributes": ["list of attributes that matched"],
                        "match_type": "exact/partial/contextual/hierarchical"
                    }
                ]
            }
            
            Limit your response to the top 3 most relevant matches. Be thorough in your reasoning
            and ensure you consider both explicit textual similarity and deeper conceptual relationships.
            """
            
            # Create a detailed user prompt with all the information
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
                ID: {candidate['id']}
                Name: {candidate['pbt_name']}
                Definition: {candidate['pbt_definition']}
                """
                
                if 'cdm' in candidate and candidate['cdm']:
                    user_prompt += f"CDM: {candidate['cdm']}\n"
                    
                if 'score' in candidate:
                    user_prompt += f"Raw Score: {candidate['score']}\n"
            
            # Create messages for the LLM
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # Get detailed evaluation from LLM with increased temperature for more nuanced reasoning
            # Use more tokens for detailed analysis
            llm_response = await self.azure_service.generate_completion(
                messages, 
                temperature=0.2,  # Low but not zero to allow some creativity in reasoning
                max_tokens=4000
            )
            
            # Parse the LLM response to extract JSON
            try:
                # Find JSON in the response
                import re
                json_match = re.search(r'({[\s\S]*})', llm_response)
                
                if json_match:
                    json_str = json_match.group(1)
                    analysis = json.loads(json_str)
                else:
                    # Fallback if JSON is not properly formatted
                    logger.warning("JSON not found in LLM response, using fallback parsing")
                    # Try to parse structured content directly
                    analysis = {"matches": []}
                    
                    # Extract matches section
                    matches_section = re.search(r'"matches"\s*:\s*\[([\s\S]*?)\]', llm_response)
                    if matches_section:
                        matches_text = matches_section.group(1)
                        match_items = re.findall(r'{([\s\S]*?)}', matches_text)
                        
                        for match_text in match_items:
                            match_dict = {}
                            
                            # Extract each field
                            term_id = re.search(r'"term_id"\s*:\s*"([^"]*)"', match_text)
                            if term_id:
                                match_dict["term_id"] = term_id.group(1)
                                
                            term_name = re.search(r'"term_name"\s*:\s*"([^"]*)"', match_text)
                            if term_name:
                                match_dict["term_name"] = term_name.group(1)
                                
                            confidence = re.search(r'"confidence"\s*:\s*([\d.]+)', match_text)
                            if confidence:
                                match_dict["confidence"] = float(confidence.group(1))
                                
                            reasoning = re.search(r'"reasoning"\s*:\s*"([^"]*)"', match_text)
                            if reasoning:
                                match_dict["reasoning"] = reasoning.group(1)
                                
                            match_type = re.search(r'"match_type"\s*:\s*"([^"]*)"', match_text)
                            if match_type:
                                match_dict["match_type"] = match_type.group(1)
                                
                            # Add to matches if we have the minimum required fields
                            if "term_id" in match_dict and "confidence" in match_dict:
                                analysis["matches"].append(match_dict)
                
                # Convert agent analysis to MappingResult objects
                mapping_results = []
                
                # Use the agent's evaluation but limit to top_k results
                for match in analysis.get("matches", [])[:top_k]:
                    # Find the term name if not provided by the agent
                    term_name = match.get("term_name", "")
                    if not term_name:
                        for candidate in candidates:
                            if candidate["id"] == match["term_id"]:
                                term_name = candidate["pbt_name"]
                                break
                    
                    # Create mapping result
                    mapping_result = MappingResult(
                        term_id=match["term_id"],
                        term_name=term_name,
                        similarity_score=match.get("confidence", 0.9),  # Default if not provided
                        confidence=match.get("confidence", 0.9),
                        mapping_type="agent",
                        matched_attributes=match.get("matched_attributes", ["pbt_name", "pbt_definition"])
                    )
                    mapping_results.append(mapping_result)
                
                return mapping_results
                
            except Exception as parse_error:
                logger.error(f"Error parsing LLM response: {parse_error}")
                logger.error(f"Raw LLM response: {llm_response[:200]}...")
                
                # Fallback to a simple ranking of candidates if parsing fails
                logger.info("Using fallback ranking due to parsing error")
                mapping_results = []
                
                # Sort candidates by score if available
                sorted_candidates = sorted(
                    candidates, 
                    key=lambda x: x.get("score", 0), 
                    reverse=True
                )[:top_k]
                
                for candidate in sorted_candidates:
                    mapping_result = MappingResult(
                        term_id=candidate["id"],
                        term_name=candidate["pbt_name"],
                        similarity_score=min(candidate.get("score", 0.8), 1.0),
                        confidence=min(candidate.get("score", 0.8), 1.0),
                        mapping_type="agent_fallback",
                        matched_attributes=["pbt_name", "pbt_definition"]
                    )
                    mapping_results.append(mapping_result)
                
                return mapping_results
                
        except Exception as e:
            logger.error(f"Error in advanced agent evaluation: {e}")
            raise
    
    async def create_langgraph_workflow(self, query: MappingRequest) -> List[MappingResult]:
        """
        Create and execute an enhanced LangGraph workflow with AI agent evaluation.
        
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
            
            # Define the nodes
            async def semantic_search_node(state: GraphState) -> GraphState:
                """Perform semantic search using ANN."""
                semantic_results = await self.semantic_search(query, top_k=8)
                return {"semantic_results": semantic_results}
            
            async def bm25_search_node(state: GraphState) -> GraphState:
                """Perform BM25 search."""
                bm25_results = await self.bm25_search(query, top_k=8)
                return {"bm25_results": bm25_results}
            
            async def keyword_search_node(state: GraphState) -> GraphState:
                """Perform keyword search."""
                keyword_results = await self.keyword_search(query, top_k=5)
                return {"keyword_results": keyword_results}
            
            async def hybrid_search_node(state: GraphState) -> GraphState:
                """Perform hybrid search."""
                hybrid_results = await self.hybrid_search(query, top_k=8)
                return {"hybrid_results": hybrid_results}
            
            async def collect_candidates_node(state: GraphState) -> GraphState:
                """Collect unique candidates from all search methods."""
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
                                response = await self.es_service.client.get(
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
                
                # Limit to top 10 candidates for agent evaluation
                top_candidates = all_candidates[:10]
                
                return {"all_candidates": top_candidates}
            
            async def agent_evaluation_node(state: GraphState) -> GraphState:
                """Perform advanced agent-based evaluation."""
                agent_results = await self.advanced_agent_evaluation(
                    query, 
                    state["all_candidates"], 
                    top_k=3  # Get top 3 results from the agent
                )
                return {"agent_results": agent_results}
            
            async def finalize_results_node(state: GraphState) -> GraphState:
                """Finalize results based on agent evaluation."""
                # The agent results are our final results
                return {"final_results": state["agent_results"]}
            
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
            
            # Run search methods in parallel branches
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
            
            # Run the workflow with the initial state
            result = await app.ainvoke(
                {
                    "query": query.model_dump(),
                    "semantic_results": [],
                    "bm25_results": [],
                    "keyword_results": [],
                    "hybrid_results": [],
                    "all_candidates": [],
                    "agent_results": [],
                    "final_results": []
                }
            )
            
            return result["final_results"]
        except Exception as e:
            logger.error(f"Error executing LangGraph workflow: {e}")
            # Fallback to simple semantic search if workflow fails
            logger.info("Falling back to semantic search due to workflow error")
            return await self.semantic_search(query, top_k=3)