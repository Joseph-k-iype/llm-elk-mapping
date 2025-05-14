"""
Service for interacting with Elasticsearch.
"""

import logging
from typing import List, Dict, Any, Optional, Union, Tuple
from elasticsearch import AsyncElasticsearch
from elasticsearch.exceptions import NotFoundError, RequestError
from app.config.settings import get_settings
from app.models.mapping import BusinessTerm

logger = logging.getLogger(__name__)

class ElasticsearchService:
    """Service for interacting with Elasticsearch."""
    
    def __init__(self):
        """Initialize the Elasticsearch service."""
        settings = get_settings()
        self.hosts = settings.elasticsearch.hosts
        self.index_name = settings.elasticsearch.index_name
        self.username = settings.elasticsearch.username
        self.password = settings.elasticsearch.password
        
        self.client = None
    
    async def connect(self):
        """Connect to Elasticsearch."""
        try:
            auth = None
            if self.username and self.password:
                auth = (self.username, self.password)
            
            self.client = AsyncElasticsearch(
                hosts=self.hosts,
                basic_auth=auth,
                verify_certs=False,
                request_timeout=30
            )
            
            logger.info(f"Connected to Elasticsearch at {', '.join(self.hosts)}")
            return self.client
        except Exception as e:
            logger.error(f"Error connecting to Elasticsearch: {e}")
            raise
    
    async def close(self):
        """Close the Elasticsearch connection."""
        if self.client:
            await self.client.close()
            logger.info("Elasticsearch connection closed")
    
    async def create_index(self, force: bool = False):
        """
        Create the Elasticsearch index.
        
        Args:
            force: Whether to force create the index (delete if exists)
        """
        try:
            if force:
                try:
                    await self.client.indices.delete(index=self.index_name)
                    logger.info(f"Existing index '{self.index_name}' deleted")
                except NotFoundError:
                    pass
            
            # Check if index exists
            exists = await self.client.indices.exists(index=self.index_name)
            if exists:
                logger.info(f"Index '{self.index_name}' already exists")
                return
            
            # Index settings with BM25 similarity and dense vector support
            settings = {
                "settings": {
                    "number_of_shards": 1,
                    "number_of_replicas": 0,
                    "analysis": {
                        "analyzer": {
                            "default": {
                                "type": "standard"
                            }
                        }
                    }
                },
                "mappings": {
                    "properties": {
                        "id": {"type": "keyword"},
                        "pbt_name": {"type": "text", "analyzer": "standard"},
                        "pbt_definition": {"type": "text", "analyzer": "standard"},
                        "cdm": {"type": "text", "analyzer": "standard"},
                        "embedding": {
                            "type": "dense_vector",
                            "dims": 3072,  # Dimension size for text-embedding-3-large
                            "index": True,
                            "similarity": "cosine"
                        }
                    }
                }
            }
            
            await self.client.indices.create(index=self.index_name, body=settings)
            logger.info(f"Index '{self.index_name}' created successfully")
        except Exception as e:
            logger.error(f"Error creating index: {e}")
            raise
    
    async def index_document(self, term: Dict[str, Any]):
        """
        Index a document in Elasticsearch.
        
        Args:
            term: Business term to index
        """
        try:
            await self.client.index(
                index=self.index_name,
                document=term,
                id=term["id"],
                refresh=True
            )
            logger.debug(f"Document indexed with ID: {term['id']}")
        except Exception as e:
            logger.error(f"Error indexing document: {e}")
            raise
    
    async def bulk_index_documents(self, terms: List[Dict[str, Any]]):
        """
        Bulk index documents in Elasticsearch.
        
        Args:
            terms: List of business terms to index
        """
        try:
            operations = []
            for term in terms:
                operations.append({"index": {"_index": self.index_name, "_id": term["id"]}})
                operations.append(term)
            
            if operations:
                await self.client.bulk(operations=operations, refresh=True)
                logger.info(f"Bulk indexed {len(terms)} documents")
        except Exception as e:
            logger.error(f"Error bulk indexing documents: {e}")
            raise
    
    async def search_by_vector(self, 
                              vector: List[float], 
                              filter_query: Optional[Dict] = None,
                              size: int = 10) -> List[Dict]:
        """
        Search documents by vector similarity.
        
        Args:
            vector: Embedding vector to search
            filter_query: Optional filter query
            size: Number of results to return
            
        Returns:
            List of matching documents
        """
        try:
            # Construct kNN query
            knn_query = {
                "field": "embedding",
                "query_vector": vector,
                "k": size,
                "num_candidates": size * 2
            }
            
            if filter_query:
                knn_query["filter"] = filter_query
            
            # Execute search
            response = await self.client.search(
                index=self.index_name,
                knn=knn_query,
                size=size
            )
            
            # Process results
            results = []
            for hit in response["hits"]["hits"]:
                doc = hit["_source"]
                doc["score"] = hit["_score"]
                results.append(doc)
            
            return results
        except Exception as e:
            logger.error(f"Error searching by vector: {e}")
            raise
    
    async def search_by_text(self, 
                            text: str, 
                            fields: List[str] = ["pbt_name", "pbt_definition"],
                            size: int = 10) -> List[Dict]:
        """
        Search documents by text using BM25.
        
        Args:
            text: Text to search
            fields: Fields to search in
            size: Number of results to return
            
        Returns:
            List of matching documents
        """
        try:
            query = {
                "multi_match": {
                    "query": text,
                    "fields": fields,
                    "type": "best_fields",
                    "operator": "or"
                }
            }
            
            response = await self.client.search(
                index=self.index_name,
                query=query,
                size=size
            )
            
            results = []
            for hit in response["hits"]["hits"]:
                doc = hit["_source"]
                doc["score"] = hit["_score"]
                results.append(doc)
            
            return results
        except Exception as e:
            logger.error(f"Error searching by text: {e}")
            raise
    
    async def search_by_keywords(self, 
                               keywords: List[str], 
                               fields: List[str] = ["pbt_name", "pbt_definition"],
                               size: int = 10) -> List[Dict]:
        """
        Search documents by keywords.
        
        Args:
            keywords: Keywords to search
            fields: Fields to search in
            size: Number of results to return
            
        Returns:
            List of matching documents
        """
        try:
            should_clauses = []
            
            for field in fields:
                should_clauses.append({
                    "match": {
                        field: {
                            "query": " ".join(keywords),
                            "operator": "or"
                        }
                    }
                })
            
            query = {
                "bool": {
                    "should": should_clauses,
                    "minimum_should_match": 1
                }
            }
            
            response = await self.client.search(
                index=self.index_name,
                query=query,
                size=size
            )
            
            results = []
            for hit in response["hits"]["hits"]:
                doc = hit["_source"]
                doc["score"] = hit["_score"]
                results.append(doc)
            
            return results
        except Exception as e:
            logger.error(f"Error searching by keywords: {e}")
            raise
    
    async def hybrid_search(self,
                          text: str,
                          vector: List[float],
                          fields: List[str] = ["pbt_name", "pbt_definition"],
                          vector_weight: float = 0.7,
                          size: int = 10) -> List[Dict]:
        """
        Hybrid search combining vector and text search.
        
        Args:
            text: Text to search
            vector: Embedding vector
            fields: Fields to search in
            vector_weight: Weight for vector search (0-1)
            size: Number of results to return
            
        Returns:
            List of matching documents
        """
        try:
            # Create combined query with function score
            query = {
                "function_score": {
                    "query": {
                        "multi_match": {
                            "query": text,
                            "fields": fields,
                            "type": "best_fields",
                            "operator": "or"
                        }
                    },
                    "functions": [
                        {
                            "script_score": {
                                "script": {
                                    "source": "cosineSimilarity(params.vector, 'embedding') + 1.0",
                                    "params": {
                                        "vector": vector
                                    }
                                }
                            },
                            "weight": vector_weight
                        }
                    ],
                    "boost_mode": "multiply",
                    "score_mode": "sum"
                }
            }
            
            response = await self.client.search(
                index=self.index_name,
                query=query,
                size=size
            )
            
            results = []
            for hit in response["hits"]["hits"]:
                doc = hit["_source"]
                doc["score"] = hit["_score"]
                results.append(doc)
            
            return results
        except RequestError as e:
            logger.error(f"Error in hybrid search: {e}")
            # Fallback to vector search if script_score fails
            logger.info("Falling back to vector search")
            return await self.search_by_vector(vector, size=size)
        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            raise
    
    async def get_all_documents(self, size: int = 1000) -> List[Dict]:
        """
        Get all documents from the index.
        
        Args:
            size: Maximum number of documents to retrieve
            
        Returns:
            List of documents
        """
        try:
            response = await self.client.search(
                index=self.index_name,
                query={"match_all": {}},
                size=size
            )
            
            results = []
            for hit in response["hits"]["hits"]:
                results.append(hit["_source"])
            
            return results
        except Exception as e:
            logger.error(f"Error getting all documents: {e}")
            raise