"""
Service for vector operations and LLM-based mapping.
"""

import logging
from typing import List, Dict, Any, Optional, Union, TypedDict
import pandas as pd
import numpy as np
from langgraph.graph import StateGraph, END
from app.services.azure_openai import AzureOpenAIService
from app.services.elasticsearch_service import ElasticsearchService
from app.models.mapping import BusinessTerm, MappingRequest, MappingResult

logger = logging.getLogger(__name__)

class VectorService:
    """Service for vector operations and LLM-based mapping."""
    
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
        Perform semantic search using embeddings.
        
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
            
            # Search by vector
            results = await self.es_service.search_by_vector(vector=query_embedding, size=top_k)
            
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
    
    async def agent_based_mapping(self, query: MappingRequest, candidates: List[Dict], top_k: int = 3) -> List[MappingResult]:
        """
        Perform agent-based mapping using LLM.
        
        Args:
            query: Mapping request
            candidates: List of candidate terms
            top_k: Number of results to return
            
        Returns:
            List of mapping results
        """
        try:
            # Create a prompt for the LLM
            system_prompt = """
            You are an expert in data governance and business terminology mapping. 
            Your task is to evaluate the semantic similarity and relevance between a user's mapping request and a set of business terms.
            
            For each candidate business term, assign a confidence score (0.0 to 1.0) based on how well it matches the request.
            Provide clear reasoning for each match, identifying the specific aspects of the term that align with the request.
            
            Return your analysis in the following JSON format:
            
            {
                "matches": [
                    {
                        "term_id": "id of the term",
                        "score": 0.0-1.0,
                        "reasoning": "Brief explanation of why this term matches",
                        "matched_attributes": ["list of attributes that matched"]
                    }
                ]
            }
            """
            
            user_prompt = f"""
            MAPPING REQUEST:
            - Name: {query.name}
            - Description: {query.description}
            """
            
            if query.example:
                user_prompt += f"- Example: {query.example}\n"
            if query.cdm:
                user_prompt += f"- CDM: {query.cdm}\n"
            if query.process_name:
                user_prompt += f"- Process Name: {query.process_name}\n"
            if query.process_description:
                user_prompt += f"- Process Description: {query.process_description}\n"
            
            user_prompt += "\nCANDIDATE BUSINESS TERMS:\n"
            
            for i, candidate in enumerate(candidates[:10]):  # Limit to 10 candidates
                user_prompt += f"""
                {i+1}. ID: {candidate['id']}
                   Name: {candidate['pbt_name']}
                   Definition: {candidate['pbt_definition']}
                """
                if candidate.get('cdm'):
                    user_prompt += f"   CDM: {candidate['cdm']}\n"
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # Get analysis from LLM
            llm_response = await self.azure_service.generate_completion(messages)
            
            # Parse response
            try:
                # Extract JSON from the response
                import re
                import json
                
                json_match = re.search(r'({.*})', llm_response.replace('\n', ' '), re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                    analysis = json.loads(json_str)
                else:
                    raise ValueError("No JSON found in LLM response")
                
                # Convert to MappingResult objects
                mapping_results = []
                for match in analysis.get("matches", [])[:top_k]:
                    mapping_result = MappingResult(
                        term_id=match["term_id"],
                        term_name=next((c["pbt_name"] for c in candidates if c["id"] == match["term_id"]), ""),
                        similarity_score=match["score"],
                        confidence=match["score"],
                        mapping_type="agent",
                        matched_attributes=match.get("matched_attributes", [])
                    )
                    mapping_results.append(mapping_result)
                
                return mapping_results
            except Exception as e:
                logger.error(f"Error parsing LLM response: {e}")
                # Fallback to a default mapping if parsing fails
                mapping_results = []
                for i, candidate in enumerate(candidates[:top_k]):
                    mapping_result = MappingResult(
                        term_id=candidate["id"],
                        term_name=candidate["pbt_name"],
                        similarity_score=0.7 - (i * 0.1),  # Decreasing scores
                        confidence=0.7 - (i * 0.1),
                        mapping_type="agent_fallback",
                        matched_attributes=["pbt_name", "pbt_definition"]
                    )
                    mapping_results.append(mapping_result)
                return mapping_results
        except Exception as e:
            logger.error(f"Error performing agent-based mapping: {e}")
            raise
    
    async def create_langgraph_workflow(self, query: MappingRequest) -> List[MappingResult]:
        """
        Create and execute a LangGraph workflow for hybrid search.
        
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
                agent_results: List
                final_results: List
            
            # Define the nodes
            async def semantic_search_node(state: GraphState) -> GraphState:
                """Perform semantic search."""
                semantic_results = await self.semantic_search(query)
                return {"semantic_results": semantic_results}
            
            async def bm25_search_node(state: GraphState) -> GraphState:
                """Perform BM25 search."""
                bm25_results = await self.bm25_search(query)
                return {"bm25_results": bm25_results}
            
            async def keyword_search_node(state: GraphState) -> GraphState:
                """Perform keyword search."""
                keyword_results = await self.keyword_search(query)
                return {"keyword_results": keyword_results}
            
            async def agent_search_node(state: GraphState) -> GraphState:
                """Perform agent-based mapping."""
                # Collect candidates from previous search methods
                candidates = {}
                for result in state["semantic_results"] + state["bm25_results"] + state["keyword_results"]:
                    if result.term_id not in candidates:
                        # Get the full document
                        response = await self.es_service.client.get(
                            index=self.es_service.index_name,
                            id=result.term_id
                        )
                        candidates[result.term_id] = response["_source"]
                
                agent_results = await self.agent_based_mapping(query, list(candidates.values()))
                return {"agent_results": agent_results}
            
            async def combine_results_node(state: GraphState) -> GraphState:
                """Combine and rank results."""
                # Collect all results
                all_results = {}
                
                # Define weights for each method
                weights = {
                    "semantic": 0.35,
                    "BM25": 0.25,
                    "keyword": 0.15,
                    "agent": 0.25
                }
                
                # Process semantic results
                for result in state["semantic_results"]:
                    term_id = result.term_id
                    if term_id not in all_results:
                        all_results[term_id] = {
                            "term_id": term_id,
                            "term_name": result.term_name,
                            "scores": {},
                            "confidence": 0.0,
                            "matched_attributes": set(),
                            "methods": []
                        }
                    all_results[term_id]["scores"]["semantic"] = result.similarity_score
                    all_results[term_id]["methods"].append("semantic")
                    all_results[term_id]["matched_attributes"].update(result.matched_attributes)
                
                # Process BM25 results
                for result in state["bm25_results"]:
                    term_id = result.term_id
                    if term_id not in all_results:
                        all_results[term_id] = {
                            "term_id": term_id,
                            "term_name": result.term_name,
                            "scores": {},
                            "confidence": 0.0,
                            "matched_attributes": set(),
                            "methods": []
                        }
                    all_results[term_id]["scores"]["BM25"] = result.similarity_score
                    all_results[term_id]["methods"].append("BM25")
                    all_results[term_id]["matched_attributes"].update(result.matched_attributes)
                
                # Process keyword results
                for result in state["keyword_results"]:
                    term_id = result.term_id
                    if term_id not in all_results:
                        all_results[term_id] = {
                            "term_id": term_id,
                            "term_name": result.term_name,
                            "scores": {},
                            "confidence": 0.0,
                            "matched_attributes": set(),
                            "methods": []
                        }
                    all_results[term_id]["scores"]["keyword"] = result.similarity_score
                    all_results[term_id]["methods"].append("keyword")
                    all_results[term_id]["matched_attributes"].update(result.matched_attributes)
                
                # Process agent results
                for result in state["agent_results"]:
                    term_id = result.term_id
                    if term_id not in all_results:
                        all_results[term_id] = {
                            "term_id": term_id,
                            "term_name": result.term_name,
                            "scores": {},
                            "confidence": 0.0,
                            "matched_attributes": set(),
                            "methods": []
                        }
                    all_results[term_id]["scores"]["agent"] = result.similarity_score
                    all_results[term_id]["methods"].append("agent")
                    all_results[term_id]["matched_attributes"].update(result.matched_attributes)
                
                # Calculate weighted scores
                for term_id, result in all_results.items():
                    weighted_score = 0.0
                    total_weight = 0.0
                    
                    for method, score in result["scores"].items():
                        weight = weights.get(method, 0.0)
                        weighted_score += score * weight
                        total_weight += weight
                    
                    if total_weight > 0:
                        result["confidence"] = weighted_score / total_weight
                    else:
                        result["confidence"] = 0.0
                
                # Convert to list and sort by confidence
                final_results = []
                for term_id, result in all_results.items():
                    # Get the primary mapping type (highest weighted method)
                    primary_method = max(
                        result["methods"], 
                        key=lambda m: weights.get(m, 0) * result["scores"].get(m, 0)
                    ) if result["methods"] else "hybrid"
                    
                    mapping_result = MappingResult(
                        term_id=term_id,
                        term_name=result["term_name"],
                        similarity_score=result["confidence"],
                        confidence=result["confidence"],
                        mapping_type=primary_method,
                        matched_attributes=list(result["matched_attributes"])
                    )
                    final_results.append(mapping_result)
                
                # Sort by confidence (descending)
                final_results.sort(key=lambda x: x.confidence, reverse=True)
                
                return {"final_results": final_results[:10]}  # Return top 10
            
            # Create the workflow
            workflow = StateGraph(GraphState)
            
            # Add nodes
            workflow.add_node("semantic_search", semantic_search_node)
            workflow.add_node("bm25_search", bm25_search_node)
            workflow.add_node("keyword_search", keyword_search_node)
            workflow.add_node("agent_search", agent_search_node)
            workflow.add_node("combine_results", combine_results_node)
            
            # Define the workflow
            workflow.set_entry_point("semantic_search")
            workflow.add_edge("semantic_search", "bm25_search")
            workflow.add_edge("bm25_search", "keyword_search")
            workflow.add_edge("keyword_search", "agent_search")
            workflow.add_edge("agent_search", "combine_results")
            workflow.add_edge("combine_results", END)
            
            # Compile the workflow
            app = workflow.compile()
            
            # Run the workflow with the initial state
            result = await app.ainvoke(
                {
                    "query": query.model_dump(),
                    "semantic_results": [],
                    "bm25_results": [],
                    "keyword_results": [],
                    "agent_results": [],
                    "final_results": []
                }
            )
            
            return result["final_results"]
        except Exception as e:
            logger.error(f"Error creating LangGraph workflow: {e}")
            # Fallback to simple semantic search
            return await self.semantic_search(query)