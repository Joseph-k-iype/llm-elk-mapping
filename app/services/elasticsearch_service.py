
"""
Enhanced Elasticsearch service with advanced vector search capabilities including KNN, ANN, and HNSW.
"""

import logging
import time
import asyncio
from typing import List, Dict, Any, Optional, Union, Tuple
from elasticsearch import AsyncElasticsearch, ConnectionError
from elasticsearch.exceptions import NotFoundError, RequestError, ConnectionTimeout, TransportError
from app.core.environment import get_os_env

logger = logging.getLogger(__name__)

class ElasticsearchService:
    """Service for interacting with Elasticsearch with enhanced vector search capabilities."""
    
    def __init__(self):
        """Initialize the Elasticsearch service with environment-based configuration."""
        # Get environment
        env = get_os_env()
        
        # Get Elasticsearch configuration
        self.hosts = self._parse_hosts(env.get("ELASTICSEARCH_HOSTS", '["http://localhost:9200"]'))
        self.index_name = env.get("ELASTICSEARCH_INDEX_NAME", "business_terms")
        self.username = env.get("ELASTICSEARCH_USERNAME", None)
        self.password = env.get("ELASTICSEARCH_PASSWORD", None)
        self.timeout = int(env.get("ELASTICSEARCH_TIMEOUT", "30"))
        self.max_retries = int(env.get("ELASTICSEARCH_MAX_RETRIES", "3"))
        self.retry_on_timeout = env.get("ELASTICSEARCH_RETRY_ON_TIMEOUT", "True").lower() in ('true', 't', 'yes', 'y', '1')
        
        # Advanced vector search configuration
        self.vector_dimensions = int(env.get("ELASTICSEARCH_VECTOR_DIMENSIONS", "3072"))
        self.vector_similarity = env.get("ELASTICSEARCH_VECTOR_SIMILARITY", "cosine")
        self.hnsw_m = int(env.get("ELASTICSEARCH_HNSW_M", "16"))  # Graph connections per node
        self.hnsw_ef_construction = int(env.get("ELASTICSEARCH_HNSW_EF_CONSTRUCTION", "100"))  # Higher = better quality, slower indexing
        self.hnsw_ef_search = int(env.get("ELASTICSEARCH_HNSW_EF_SEARCH", "100"))  # Higher = better search quality, slower search
        
        # Log configuration
        logger.info(f"Elasticsearch configuration:")
        logger.info(f"  - Hosts: {self.hosts}")
        logger.info(f"  - Index: {self.index_name}")
        logger.info(f"  - Username: {'set' if self.username else 'not set'}")
        logger.info(f"  - Password: {'set' if self.password else 'not set'}")
        logger.info(f"  - Timeout: {self.timeout}s")
        logger.info(f"  - Max Retries: {self.max_retries}")
        logger.info(f"  - Retry On Timeout: {self.retry_on_timeout}")
        
        # Log vector search configuration
        logger.info(f"Vector search configuration:")
        logger.info(f"  - Vector Dimensions: {self.vector_dimensions}")
        logger.info(f"  - Vector Similarity: {self.vector_similarity}")
        logger.info(f"  - HNSW M: {self.hnsw_m}")
        logger.info(f"  - HNSW EF Construction: {self.hnsw_ef_construction}")
        logger.info(f"  - HNSW EF Search: {self.hnsw_ef_search}")
        
        self.client = None
        self._connected = False
    
    def _parse_hosts(self, hosts_str: str) -> List[str]:
        """
        Parse hosts string into list of hosts.
        
        Args:
            hosts_str: String representation of hosts (JSON array or single string)
            
        Returns:
            List of host URLs
        """
        try:
            if hosts_str.startswith('[') and hosts_str.endswith(']'):
                # Looks like a JSON array
                import json
                hosts = json.loads(hosts_str)
            else:
                # Single host string
                hosts = [hosts_str]
            
            # Clean up hosts (remove quotes, ensure https://)
            clean_hosts = []
            for host in hosts:
                if isinstance(host, str):
                    # Remove quotes if they exist
                    host = host.strip('\'"')
                    
                    # Ensure https:// protocol if not specified
                    if not host.startswith(('http://', 'https://')):
                        host = f"https://{host}"
                    
                    clean_hosts.append(host)
            
            return clean_hosts
        except Exception as e:
            logger.warning(f"Error parsing hosts: {e}, falling back to default")
            return ["http://localhost:9200"]
    
    async def connect(self, max_retries: int = 3) -> Optional[AsyncElasticsearch]:
        """
        Connect to Elasticsearch with retries.
        
        Args:
            max_retries: Maximum number of connection attempts
            
        Returns:
            AsyncElasticsearch client if successful, None otherwise
        """
        if self._connected and self.client:
            logger.debug("Already connected to Elasticsearch")
            return self.client
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Connecting to Elasticsearch (attempt {attempt+1}/{max_retries})...")
                
                # Prepare connection parameters
                conn_params = {
                    "timeout": self.timeout,
                    "max_retries": self.max_retries,
                    "retry_on_timeout": self.retry_on_timeout,
                    "verify_certs": False,  # For development ease
                    "ssl_show_warn": False  # Suppress SSL warnings
                }
                
                # Add authentication if provided
                if self.username and self.password:
                    conn_params["basic_auth"] = (self.username, self.password)
                    logger.info(f"Using basic authentication with username: {self.username}")
                
                # Create client
                self.client = AsyncElasticsearch(
                    self.hosts,
                    **conn_params
                )
                
                # Test connection
                info = await self.client.info()
                es_version = info["version"]["number"]
                cluster_name = info["cluster_name"]
                logger.info(f"Connected to Elasticsearch version {es_version} on cluster '{cluster_name}'")
                
                self._connected = True
                return self.client
                
            except ConnectionError as e:
                logger.error(f"Failed to connect to Elasticsearch (attempt {attempt+1}): {e}")
                if attempt + 1 < max_retries:
                    # Wait before retrying (exponential backoff)
                    sleep_time = 2 ** attempt
                    logger.info(f"Retrying in {sleep_time} seconds...")
                    await asyncio.sleep(sleep_time)
            except Exception as e:
                logger.error(f"Unexpected error connecting to Elasticsearch: {e}")
                if attempt + 1 < max_retries:
                    await asyncio.sleep(2 ** attempt)
                else:
                    raise
        
        logger.error(f"Failed to connect to Elasticsearch after {max_retries} attempts")
        return None
    
    async def close(self):
        """Close the Elasticsearch connection."""
        if self.client:
            try:
                await self.client.close()
                logger.info("Elasticsearch connection closed")
                self._connected = False
            except Exception as e:
                logger.error(f"Error closing Elasticsearch connection: {e}")
    
    async def create_index(self, force: bool = False):
        """
        Create the Elasticsearch index with advanced vector search capabilities.
        
        Args:
            force: Whether to force create the index (delete if exists)
        """
        try:
            # Ensure client is connected
            if not self.client:
                await self.connect()
            
            if force:
                try:
                    logger.info(f"Forcibly deleting index '{self.index_name}' if it exists")
                    await self.client.indices.delete(index=self.index_name)
                    logger.info(f"Existing index '{self.index_name}' deleted")
                except NotFoundError:
                    logger.info(f"Index '{self.index_name}' does not exist, nothing to delete")
                except Exception as e:
                    logger.warning(f"Error deleting index: {e}")
            
            # Check if index exists
            exists = await self.client.indices.exists(index=self.index_name)
            if exists:
                logger.info(f"Index '{self.index_name}' already exists")
                return
            
            # Index settings with enhanced vector search capabilities
            settings = {
                "settings": {
                    "number_of_shards": 1,
                    "number_of_replicas": 0,
                    "index.knn": True,  # Enable KNN capabilities
                    "analysis": {
                        "analyzer": {
                            "default": {
                                "type": "standard"
                            },
                            "ngram_analyzer": {
                                "tokenizer": "standard",
                                "filter": ["lowercase", "ngram_filter"]
                            }
                        },
                        "filter": {
                            "ngram_filter": {
                                "type": "ngram",
                                "min_gram": 3,
                                "max_gram": 4
                            }
                        }
                    }
                },
                "mappings": {
                    "properties": {
                        "id": {"type": "keyword"},
                        "pbt_name": {
                            "type": "text", 
                            "analyzer": "standard",
                            "fields": {
                                "keyword": {"type": "keyword"},
                                "ngram": {"type": "text", "analyzer": "ngram_analyzer"}
                            }
                        },
                        "pbt_definition": {
                            "type": "text", 
                            "analyzer": "standard"
                        },
                        "cdm": {
                            "type": "text", 
                            "analyzer": "standard",
                            "fields": {
                                "keyword": {"type": "keyword"}
                            }
                        },
                        "embedding": {
                            "type": "dense_vector",
                            "dims": self.vector_dimensions,
                            "index": True,
                            "similarity": self.vector_similarity,
                            "index_options": {
                                "type": "hnsw",
                                "m": self.hnsw_m,
                                "ef_construction": self.hnsw_ef_construction
                            }
                        }
                    }
                }
            }
            
            logger.info(f"Creating index '{self.index_name}' with {self.vector_dimensions} dimensions and HNSW algorithm")
            await self.client.indices.create(index=self.index_name, body=settings)
            logger.info(f"Index '{self.index_name}' created successfully with advanced vector search settings")
            
        except Exception as e:
            logger.error(f"Error creating index: {e}")
            raise
    
    async def index_document(self, term: Dict[str, Any], max_retries: int = 3) -> bool:
        """
        Index a document in Elasticsearch with retries.
        
        Args:
            term: Business term to index
            max_retries: Maximum number of retry attempts
            
        Returns:
            True if successful, False otherwise
        """
        # Ensure client is connected
        if not self.client:
            await self.connect()
        
        for attempt in range(max_retries):
            try:
                await self.client.index(
                    index=self.index_name,
                    document=term,
                    id=term["id"],
                    refresh=True
                )
                logger.debug(f"Document indexed with ID: {term['id']}")
                return True
            except ConnectionTimeout as e:
                logger.warning(f"Connection timeout indexing document (attempt {attempt+1}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
            except Exception as e:
                logger.error(f"Error indexing document (attempt {attempt+1}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    return False
        
        return False
    
    async def bulk_index_documents(self, terms: List[Dict[str, Any]], batch_size: int = 50, max_retries: int = 3) -> int:
        """
        Bulk index documents in Elasticsearch with retries and batching.
        
        Args:
            terms: List of business terms to index
            batch_size: Size of batches for bulk operations
            max_retries: Maximum number of retry attempts
            
        Returns:
            Number of successfully indexed documents
        """
        if not terms:
            return 0
        
        # Ensure client is connected
        if not self.client:
            await self.connect()
        
        success_count = 0
        
        # Process in batches
        for i in range(0, len(terms), batch_size):
            batch = terms[i:i+batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(terms) + batch_size - 1) // batch_size
            
            logger.info(f"Processing bulk index batch {batch_num}/{total_batches} ({len(batch)} documents)")
            
            # Prepare bulk operations
            operations = []
            for term in batch:
                operations.append({"index": {"_index": self.index_name, "_id": term["id"]}})
                operations.append(term)
            
            # Try to index the batch with retries
            for attempt in range(max_retries):
                try:
                    result = await self.client.bulk(operations=operations, refresh=True)
                    
                    # Check for errors
                    if not result.get("errors", False):
                        success_count += len(batch)
                        logger.info(f"Successfully indexed batch {batch_num}/{total_batches}")
                        break
                    else:
                        # Count successful operations
                        successful_ops = sum(1 for item in result.get("items", []) 
                                          if item.get("index", {}).get("status", 500) < 400)
                        success_count += successful_ops
                        
                        # Log errors
                        error_count = len(batch) - successful_ops
                        logger.warning(f"Batch {batch_num} completed with {error_count} errors")
                        
                        if attempt < max_retries - 1:
                            logger.info(f"Retrying failed operations (attempt {attempt+2}/{max_retries})")
                            
                            # Prepare operations for retry (only failed ones)
                            retry_operations = []
                            for i, item in enumerate(result.get("items", [])):
                                status = item.get("index", {}).get("status", 500)
                                if status >= 400:
                                    doc_idx = i // 2
                                    if doc_idx < len(batch):
                                        retry_operations.append({"index": {"_index": self.index_name, "_id": batch[doc_idx]["id"]}})
                                        retry_operations.append(batch[doc_idx])
                            
                            # Update operations for retry
                            operations = retry_operations
                        else:
                            logger.error(f"Failed to index some documents in batch {batch_num} after {max_retries} attempts")
                    
                except ConnectionTimeout as e:
                    logger.warning(f"Connection timeout in bulk index (attempt {attempt+1}): {e}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2 ** attempt)
                    else:
                        logger.error(f"Failed to index batch {batch_num} after {max_retries} attempts")
                except Exception as e:
                    logger.error(f"Error in bulk index (attempt {attempt+1}): {e}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2 ** attempt)
                    else:
                        logger.error(f"Failed to index batch {batch_num} after {max_retries} attempts")
            
            # Add delay between batches to avoid overloading ES
            if i + batch_size < len(terms):
                await asyncio.sleep(1)
        
        logger.info(f"Bulk indexing completed: {success_count}/{len(terms)} documents indexed successfully")
        return success_count
    
    async def search_by_vector_ann(self, 
                               vector: List[float], 
                               filter_query: Optional[Dict] = None,
                               size: int = 10,
                               ef_search: Optional[int] = None,
                               max_retries: int = 2) -> List[Dict]:
        """
        Search documents by vector similarity using ANN with HNSW.
        This is the most efficient approach for large datasets.
        
        Args:
            vector: Embedding vector to search
            filter_query: Optional filter query
            size: Number of results to return
            ef_search: Runtime search parameter (higher = better quality but slower)
            max_retries: Maximum number of retry attempts
            
        Returns:
            List of matching documents
        """
        # Ensure client is connected
        if not self.client:
            await self.connect()
        
        for attempt in range(max_retries):
            try:
                # Build kNN query
                knn_query = {
                    "field": "embedding",
                    "query_vector": vector,
                    "k": size,
                    "num_candidates": size * 4  # Increase for better quality
                }
                
                # Add runtime parameters if provided
                if ef_search:
                    knn_query["ef_search"] = ef_search
                    
                # Add filter if provided
                if filter_query:
                    knn_query["filter"] = filter_query
                
                # Execute kNN search
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
                
                logger.info(f"ANN vector search returned {len(results)} results")
                return results
                
            except ConnectionTimeout as e:
                logger.warning(f"Connection timeout in ANN search (attempt {attempt+1}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
            except Exception as e:
                logger.error(f"Error in ANN search (attempt {attempt+1}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    logger.error(f"ANN search failed after {max_retries} attempts")
                    return []
        
        return []
    
    async def search_by_vector_exact(self, 
                                  vector: List[float], 
                                  filter_query: Optional[Dict] = None,
                                  size: int = 10,
                                  max_retries: int = 2) -> List[Dict]:
        """
        Search documents by vector similarity using exact KNN (slower but accurate).
        
        Args:
            vector: Embedding vector to search
            filter_query: Optional filter query
            size: Number of results to return
            max_retries: Maximum number of retry attempts
            
        Returns:
            List of matching documents
        """
        # Ensure client is connected
        if not self.client:
            await self.connect()
        
        for attempt in range(max_retries):
            try:
                # Create script score query based on similarity function
                if self.vector_similarity == "cosine":
                    script = {
                        "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                        "params": {
                            "query_vector": vector
                        }
                    }
                elif self.vector_similarity == "dot_product":
                    script = {
                        "source": "dotProduct(params.query_vector, 'embedding')",
                        "params": {
                            "query_vector": vector
                        }
                    }
                else:  # l2_norm (euclidean)
                    script = {
                        "source": "1 / (1 + l2norm(params.query_vector, 'embedding'))",
                        "params": {
                            "query_vector": vector
                        }
                    }
                
                # Create query
                query = {
                    "script_score": {
                        "query": filter_query if filter_query else {"match_all": {}},
                        "script": script
                    }
                }
                
                # Execute search
                response = await self.client.search(
                    index=self.index_name,
                    query=query,
                    size=size
                )
                
                # Process results
                results = []
                for hit in response["hits"]["hits"]:
                    doc = hit["_source"]
                    doc["score"] = hit["_score"]
                    results.append(doc)
                
                logger.info(f"Exact vector search returned {len(results)} results")
                return results
                
            except ConnectionTimeout as e:
                logger.warning(f"Connection timeout in exact vector search (attempt {attempt+1}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
            except Exception as e:
                logger.error(f"Error in exact vector search (attempt {attempt+1}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    logger.error(f"Exact vector search failed after {max_retries} attempts")
                    return []
        
        return []
    
    async def search_by_text(self, 
                           text: str, 
                           fields: List[str] = ["pbt_name", "pbt_definition"],
                           size: int = 10,
                           max_retries: int = 2) -> List[Dict]:
        """
        Search documents by text using BM25.
        
        Args:
            text: Text to search
            fields: Fields to search in
            size: Number of results to return
            max_retries: Maximum number of retry attempts
            
        Returns:
            List of matching documents
        """
        # Ensure client is connected
        if not self.client:
            await self.connect()
        
        for attempt in range(max_retries):
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
                
                logger.info(f"Text search returned {len(results)} results")
                return results
                
            except ConnectionTimeout as e:
                logger.warning(f"Connection timeout in text search (attempt {attempt+1}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
            except Exception as e:
                logger.error(f"Error in text search (attempt {attempt+1}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    logger.error(f"Text search failed after {max_retries} attempts")
                    return []
        
        return []
    
    async def search_by_keywords(self, 
                              keywords: List[str], 
                              fields: List[str] = ["pbt_name", "pbt_definition"],
                              size: int = 10,
                              max_retries: int = 2) -> List[Dict]:
        """
        Search documents by keywords.
        
        Args:
            keywords: Keywords to search
            fields: Fields to search in
            size: Number of results to return
            max_retries: Maximum number of retry attempts
            
        Returns:
            List of matching documents
        """
        # Ensure client is connected
        if not self.client:
            await self.connect()
        
        for attempt in range(max_retries):
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
                
                logger.info(f"Keyword search returned {len(results)} results")
                return results
                
            except ConnectionTimeout as e:
                logger.warning(f"Connection timeout in keyword search (attempt {attempt+1}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
            except Exception as e:
                logger.error(f"Error in keyword search (attempt {attempt+1}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    logger.error(f"Keyword search failed after {max_retries} attempts")
                    return []
        
        return []
    
    async def hybrid_search(self,
                         text: str,
                         vector: List[float],
                         fields: List[str] = ["pbt_name", "pbt_definition"],
                         vector_weight: float = 0.7,
                         size: int = 10,
                         max_retries: int = 2) -> List[Dict]:
        """
        Hybrid search combining vector and text search.
        
        Args:
            text: Text to search
            vector: Embedding vector
            fields: Fields to search in
            vector_weight: Weight for vector search (0-1)
            size: Number of results to return
            max_retries: Maximum number of retry attempts
            
        Returns:
            List of matching documents
        """
        # Ensure client is connected
        if not self.client:
            await self.connect()
        
        # Define the script based on similarity function
        if self.vector_similarity == "cosine":
            score_script = "cosineSimilarity(params.vector, 'embedding') + 1.0"
        elif self.vector_similarity == "dot_product":
            score_script = "dotProduct(params.vector, 'embedding')"
        else:  # l2_norm
            score_script = "1 / (1 + l2norm(params.vector, 'embedding'))"
        
        for attempt in range(max_retries):
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
                                        "source": score_script,
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
                
                logger.info(f"Hybrid search returned {len(results)} results")
                return results
                
            except RequestError as e:
                logger.warning(f"Request error in hybrid search (attempt {attempt+1}): {e}")
                # Fall back to ANN search
                logger.warning("Falling back to ANN search")
                return await self.search_by_vector_ann(vector, size=size)
            except ConnectionTimeout as e:
                logger.warning(f"Connection timeout in hybrid search (attempt {attempt+1}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
            except Exception as e:
                logger.error(f"Error in hybrid search (attempt {attempt+1}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    logger.error(f"Hybrid search failed after {max_retries} attempts, falling back to ANN search")
                    return await self.search_by_vector_ann(vector, size=size)
        
        # If all retries fail, try ANN search as last resort
        logger.warning("All hybrid search attempts failed, falling back to ANN search")
        return await self.search_by_vector_ann(vector, size=size)
    
    async def get_all_documents(self, size: int = 1000, max_retries: int = 2) -> List[Dict]:
        """
        Get all documents from the index.
        
        Args:
            size: Maximum number of documents to retrieve
            max_retries: Maximum number of retry attempts
            
        Returns:
            List of documents
        """
        # Ensure client is connected
        if not self.client:
            await self.connect()
        
        for attempt in range(max_retries):
            try:
                response = await self.client.search(
                    index=self.index_name,
                    query={"match_all": {}},
                    size=size
                )
                
                results = []
                for hit in response["hits"]["hits"]:
                    results.append(hit["_source"])
                
                logger.info(f"Retrieved {len(results)} documents from index")
                return results
                
            except ConnectionTimeout as e:
                logger.warning(f"Connection timeout getting all documents (attempt {attempt+1}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
            except Exception as e:
                logger.error(f"Error getting all documents (attempt {attempt+1}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    logger.error(f"Failed to get all documents after {max_retries} attempts")
                    return []
        
        return []