"""
Enhanced Elasticsearch service with production-grade connection handling, resilient queries,
and advanced vector search capabilities.
"""

import logging
import time
import asyncio
import json
from typing import List, Dict, Any, Optional, Union, Tuple
from elasticsearch import AsyncElasticsearch, ConnectionError, NotFoundError, RequestError, TransportError
from elasticsearch.exceptions import ConnectionTimeout, AuthorizationException, AuthenticationException
from app.core.environment import get_os_env

logger = logging.getLogger(__name__)

class ElasticsearchService:
    """
    Production-grade service for interacting with Elasticsearch with enhanced connection pooling,
    automatic retry mechanisms, and advanced vector search capabilities.
    """
    
    def __init__(self):
        """Initialize the Elasticsearch service with environment-based configuration."""
        # Get environment
        env = get_os_env()
        
        # Get Elasticsearch configuration
        self.hosts = self._parse_hosts(env.get("ELASTICSEARCH_HOSTS", '["http://localhost:9200"]'))
        self.index_name = env.get("ELASTICSEARCH_INDEX_NAME", "business_terms")
        self.username = env.get("ELASTICSEARCH_USERNAME", None)
        self.password = env.get("ELASTICSEARCH_PASSWORD", None)
        
        # Connection settings
        self.request_timeout = int(env.get("ELASTICSEARCH_REQUEST_TIMEOUT", "30"))
        self.connect_timeout = int(env.get("ELASTICSEARCH_CONNECT_TIMEOUT", "10"))
        self.max_retries = int(env.get("ELASTICSEARCH_MAX_RETRIES", "5"))
        self.retry_on_timeout = env.get("ELASTICSEARCH_RETRY_ON_TIMEOUT", "True").lower() in ('true', 't', 'yes', 'y', '1')
        self.retry_on_status = [429, 500, 502, 503, 504]  # Retry on these status codes
        
        # Connection pool settings
        self.maxsize = int(env.get("ELASTICSEARCH_CONN_MAXSIZE", "25"))  # Max connections per node
        self.max_keepalive_time = int(env.get("ELASTICSEARCH_MAX_KEEPALIVE", "600"))  # In seconds
        
        # Circuit breaker settings
        self.circuit_breaker_enabled = env.get("ELASTICSEARCH_CIRCUIT_BREAKER", "True").lower() in ('true', 't', 'yes', 'y', '1')
        self.circuit_breaker_threshold = int(env.get("ELASTICSEARCH_CB_THRESHOLD", "5"))
        self.circuit_breaker_timeout = int(env.get("ELASTICSEARCH_CB_TIMEOUT", "60"))
        
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
        logger.info(f"  - Request timeout: {self.request_timeout}s")
        logger.info(f"  - Connect timeout: {self.connect_timeout}s")
        logger.info(f"  - Max Retries: {self.max_retries}")
        logger.info(f"  - Retry on status: {self.retry_on_status}")
        logger.info(f"  - Connection pool size: {self.maxsize}")
        logger.info(f"  - Circuit breaker: {'enabled' if self.circuit_breaker_enabled else 'disabled'}")
        
        # Circuit breaker state
        self._circuit_open = False
        self._failure_count = 0
        self._last_failure_time = 0
        
        # Create client with None to ensure connect() is called before use
        self.client = None
        self._connected = False
    
    def _parse_hosts(self, hosts_str: str) -> List[str]:
        """
        Parse hosts string into list of hosts with error handling.
        
        Args:
            hosts_str: String representation of hosts (JSON array or single string)
            
        Returns:
            List of host URLs
        """
        try:
            # Try to parse as JSON array
            if isinstance(hosts_str, str) and hosts_str.startswith('[') and hosts_str.endswith(']'):
                hosts = json.loads(hosts_str.replace("'", '"'))
            elif isinstance(hosts_str, list):
                hosts = hosts_str
            else:
                # Single host string
                hosts = [hosts_str]
            
            # Clean up hosts (remove quotes, ensure protocol)
            clean_hosts = []
            for host in hosts:
                if isinstance(host, str):
                    # Remove quotes if they exist
                    host = host.strip('\'"')
                    
                    # Ensure protocol
                    if not host.startswith(('http://', 'https://')):
                        host = f"https://{host}"
                    
                    clean_hosts.append(host)
            
            return clean_hosts
        except Exception as e:
            logger.warning(f"Error parsing hosts: {e}, falling back to default")
            return ["http://localhost:9200"]
    
    async def connect(self, max_retries: int = 3) -> Optional[AsyncElasticsearch]:
        """
        Connect to Elasticsearch with retries and robust error handling.
        
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
                
                # Prepare connection parameters with optimized settings
                conn_params = {
                    "timeout": self.request_timeout,
                    "retry_on_timeout": self.retry_on_timeout,
                    "max_retries": self.max_retries,
                    "retry_on_status": self.retry_on_status,
                    "connection_class": None,  # Use default (AsyncHttpConnection)
                    "connections_per_node": self.maxsize,  # Control connection pooling
                    "verify_certs": False,  # For development ease
                    "ssl_show_warn": False,  # Suppress SSL warnings
                    "request_timeout": self.request_timeout,
                    "connection_timeout": self.connect_timeout
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
                info = await self.client.info(request_timeout=self.connect_timeout)
                es_version = info["version"]["number"]
                cluster_name = info["cluster_name"]
                logger.info(f"Connected to Elasticsearch version {es_version} on cluster '{cluster_name}'")
                
                # Reset circuit breaker state
                self._circuit_open = False
                self._failure_count = 0
                self._connected = True
                
                return self.client
                
            except ConnectionError as e:
                logger.error(f"Failed to connect to Elasticsearch (attempt {attempt+1}): {e}")
                if attempt + 1 < max_retries:
                    # Wait before retrying (exponential backoff with jitter)
                    sleep_time = (2 ** attempt) + (0.1 * random.random())
                    logger.info(f"Retrying in {sleep_time:.2f} seconds...")
                    await asyncio.sleep(sleep_time)
            except (AuthenticationException, AuthorizationException) as e:
                logger.error(f"Authentication error connecting to Elasticsearch: {e}")
                # Authentication errors won't be resolved by retrying
                break
            except Exception as e:
                logger.error(f"Unexpected error connecting to Elasticsearch: {e}")
                if attempt + 1 < max_retries:
                    sleep_time = (2 ** attempt) + (0.1 * random.random())
                    await asyncio.sleep(sleep_time)
                else:
                    raise
        
        logger.error(f"Failed to connect to Elasticsearch after {max_retries} attempts")
        return None
    
    async def close(self):
        """Close the Elasticsearch connection safely."""
        if self.client:
            try:
                await self.client.close()
                logger.info("Elasticsearch connection closed")
                self._connected = False
                self.client = None
            except Exception as e:
                logger.error(f"Error closing Elasticsearch connection: {e}")
    
    def _update_circuit_breaker(self, success: bool):
        """
        Update circuit breaker state based on success/failure of operation.
        
        Args:
            success: Whether the operation was successful
        """
        if not self.circuit_breaker_enabled:
            return
        
        current_time = time.time()
        
        if success:
            # Reset failure count on success
            self._failure_count = 0
            
            # Check if circuit was open and needs to be closed
            if self._circuit_open:
                # Check if timeout elapsed
                if current_time - self._last_failure_time > self.circuit_breaker_timeout:
                    logger.info("Circuit breaker closed - successful operation")
                    self._circuit_open = False
        else:
            # Increment failure count
            self._failure_count += 1
            self._last_failure_time = current_time
            
            # Check if circuit should be opened
            if self._failure_count >= self.circuit_breaker_threshold:
                if not self._circuit_open:
                    logger.warning(f"Circuit breaker opened - {self._failure_count} consecutive failures")
                    self._circuit_open = True
    
    async def _execute_with_retry(self, operation_func, *args, max_retries=None, **kwargs):
        """
        Execute an Elasticsearch operation with retry logic and circuit breaker.
        
        Args:
            operation_func: Function to execute
            *args: Arguments for the function
            max_retries: Maximum number of retries (defaults to self.max_retries)
            **kwargs: Keyword arguments for the function
            
        Returns:
            Result of the operation
            
        Raises:
            Exception: If all retries fail
        """
        if max_retries is None:
            max_retries = self.max_retries
        
        # Check circuit breaker
        if self.circuit_breaker_enabled and self._circuit_open:
            current_time = time.time()
            if current_time - self._last_failure_time <= self.circuit_breaker_timeout:
                logger.warning("Circuit breaker open - skipping operation")
                raise ConnectionError("Circuit breaker open")
            else:
                # Try one request to see if things have recovered
                logger.info("Circuit breaker timeout elapsed - attempting operation")
                self._circuit_open = False
        
        # Ensure client is connected
        if not self.client:
            await self.connect()
        
        last_exception = None
        total_delay = 0
        
        for attempt in range(max_retries + 1):  # +1 for initial attempt
            try:
                # Execute operation
                result = await operation_func(*args, **kwargs)
                
                # Update circuit breaker on success
                self._update_circuit_breaker(True)
                
                return result
                
            except ConnectionTimeout as e:
                logger.warning(f"Connection timeout (attempt {attempt+1}/{max_retries+1}): {e}")
                last_exception = e
                self._update_circuit_breaker(False)
            except TransportError as e:
                if hasattr(e, 'status_code') and e.status_code in self.retry_on_status:
                    logger.warning(f"Transport error with status {e.status_code} (attempt {attempt+1}/{max_retries+1}): {e}")
                    last_exception = e
                    self._update_circuit_breaker(False)
                    
                    # Check for Retry-After header (standard rate limiting response)
                    retry_after = None
                    if hasattr(e, 'info') and isinstance(e.info, dict) and 'retry-after' in e.info:
                        try:
                            retry_after = int(e.info['retry-after'])
                        except (ValueError, TypeError):
                            pass
                    
                    if retry_after is not None:
                        # Use server-specified retry delay
                        delay = retry_after
                    else:
                        # Exponential backoff with jitter
                        delay = min(30, (2 ** attempt) + (random.random() * 0.5))
                else:
                    # Don't retry other transport errors
                    logger.error(f"Non-retriable transport error: {e}")
                    raise
            except Exception as e:
                logger.error(f"Unexpected error in Elasticsearch operation: {e}")
                last_exception = e
                self._update_circuit_breaker(False)
                
                # Only retry certain types of errors
                if isinstance(e, (ConnectionError, RequestError)):
                    delay = min(30, (2 ** attempt) + (random.random() * 0.5))
                else:
                    # Don't retry other errors
                    raise
            
            # Only attempt retry if not on final attempt
            if attempt < max_retries:
                logger.info(f"Retrying in {delay:.2f} seconds (attempt {attempt+1}/{max_retries})...")
                await asyncio.sleep(delay)
                total_delay += delay
                
                # If total delay exceeds request timeout, break early
                if total_delay > self.request_timeout:
                    logger.warning(f"Breaking retry loop after {total_delay:.2f}s cumulative delay")
                    break
                
                # Check if we need to reconnect
                if attempt > 0 and attempt % 2 == 0:
                    try:
                        logger.info("Reconnecting to Elasticsearch...")
                        await self.connect(max_retries=1)
                    except Exception as e:
                        logger.error(f"Error reconnecting to Elasticsearch: {e}")
        
        # If we got here, all retries failed
        logger.error(f"All {max_retries+1} attempts failed for Elasticsearch operation")
        if last_exception:
            raise last_exception
        else:
            raise ConnectionError("Maximum retries exceeded")
    
    async def create_index(self, force: bool = False):
        """
        Create the Elasticsearch index with robust error handling.
        
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
                    await self._execute_with_retry(self.client.indices.delete, index=self.index_name)
                    logger.info(f"Existing index '{self.index_name}' deleted")
                except NotFoundError:
                    logger.info(f"Index '{self.index_name}' does not exist, nothing to delete")
                except Exception as e:
                    logger.warning(f"Error deleting index: {e}")
            
            # Check if index exists
            exists = await self._execute_with_retry(self.client.indices.exists, index=self.index_name)
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
            await self._execute_with_retry(self.client.indices.create, index=self.index_name, body=settings)
            logger.info(f"Index '{self.index_name}' created successfully with advanced vector search settings")
            
        except Exception as e:
            logger.error(f"Error creating index: {e}")
            raise
    
    async def index_document(self, term: Dict[str, Any]) -> bool:
        """
        Index a document in Elasticsearch with retries.
        
        Args:
            term: Business term to index
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure client is connected
            if not self.client:
                await self.connect()
            
            await self._execute_with_retry(
                self.client.index,
                index=self.index_name,
                document=term,
                id=term["id"],
                refresh=True
            )
            logger.debug(f"Document indexed with ID: {term['id']}")
            return True
        except Exception as e:
            logger.error(f"Error indexing document: {e}")
            return False
    
    async def bulk_index_documents(self, terms: List[Dict[str, Any]], batch_size: int = 25) -> int:
        """
        Bulk index documents in Elasticsearch with retries, batching, and rate limiting.
        
        Args:
            terms: List of business terms to index
            batch_size: Size of batches for bulk operations
            
        Returns:
            Number of successfully indexed documents
        """
        if not terms:
            return 0
        
        # Ensure client is connected
        if not self.client:
            await self.connect()
        
        success_count = 0
        failure_count = 0
        
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
            
            try:
                # Execute bulk operation with retry logic
                result = await self._execute_with_retry(
                    self.client.bulk,
                    operations=operations,
                    refresh=True
                )
                
                # Check for errors
                if not result.get("errors", False):
                    success_count += len(batch)
                    logger.info(f"Successfully indexed batch {batch_num}/{total_batches}")
                else:
                    # Count successful operations
                    success_items = sum(1 for item in result.get("items", []) 
                                     if item.get("index", {}).get("status", 500) < 400)
                    success_count += success_items
                    failure_count += len(batch) - success_items
                    
                    # Log errors
                    error_items = [item for item in result.get("items", []) 
                                  if item.get("index", {}).get("error")]
                    if error_items:
                        errors = [f"{item['index']['_id']}: {item['index']['error']['type']}" 
                                 for item in error_items[:3]]
                        logger.error(f"Batch {batch_num} had {len(error_items)} errors: {errors}")
            except Exception as e:
                logger.error(f"Error processing batch {batch_num}: {e}")
                failure_count += len(batch)
            
            # Add delay between batches to avoid overwhelming ES
            if i + batch_size < len(terms):
                await asyncio.sleep(0.5)  # 500ms delay between batches
        
        logger.info(f"Bulk indexing completed: {success_count} successful, {failure_count} failed")
        return success_count
    
    async def search_by_vector_ann(self, 
                               vector: List[float], 
                               filter_query: Optional[Dict] = None,
                               size: int = 10,
                               ef_search: Optional[int] = None) -> List[Dict]:
        """
        Search documents by vector similarity using ANN with HNSW.
        This is the most efficient approach for large datasets.
        
        Args:
            vector: Embedding vector to search
            filter_query: Optional filter query
            size: Number of results to return
            ef_search: Runtime search parameter (higher = better quality but slower)
            
        Returns:
            List of matching documents
        """
        try:
            # Ensure client is connected
            if not self.client:
                await self.connect()
            
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
            
            # Execute kNN search with retry
            response = await self._execute_with_retry(
                self.client.search,
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
                
        except Exception as e:
            logger.error(f"Error in ANN search: {e}")
            return []
    
    async def search_by_vector_exact(self, 
                                  vector: List[float], 
                                  filter_query: Optional[Dict] = None,
                                  size: int = 10) -> List[Dict]:
        """
        Search documents by vector similarity using exact KNN (slower but accurate).
        
        Args:
            vector: Embedding vector to search
            filter_query: Optional filter query
            size: Number of results to return
            
        Returns:
            List of matching documents
        """
        try:
            # Ensure client is connected
            if not self.client:
                await self.connect()
            
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
            
            # Execute search with retry
            response = await self._execute_with_retry(
                self.client.search,
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
                
        except Exception as e:
            logger.error(f"Error in exact vector search: {e}")
            return []
    
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
            # Ensure client is connected
            if not self.client:
                await self.connect()
            
            query = {
                "multi_match": {
                    "query": text,
                    "fields": fields,
                    "type": "best_fields",
                    "operator": "or"
                }
            }
            
            # Execute search with retry
            response = await self._execute_with_retry(
                self.client.search,
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
            
            logger.info(f"Text search returned {len(results)} results")
            return results
                
        except Exception as e:
            logger.error(f"Error in text search: {e}")
            return []
    
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
            # Ensure client is connected
            if not self.client:
                await self.connect()
            
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
            
            # Execute search with retry
            response = await self._execute_with_retry(
                self.client.search,
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
            
            logger.info(f"Keyword search returned {len(results)} results")
            return results
                
        except Exception as e:
            logger.error(f"Error in keyword search: {e}")
            return []
    
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
            
            # Execute search with retry
            response = await self._execute_with_retry(
                self.client.search,
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
            
            logger.info(f"Hybrid search returned {len(results)} results")
            return results
                
        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            
            # Fall back to ANN search if hybrid fails
            logger.warning("Falling back to ANN search due to hybrid search error")
            return await self.search_by_vector_ann(vector, size=size)
    
    async def get_all_documents(self, size: int = 1000) -> List[Dict]:
        """
        Get all documents from the index.
        
        Args:
            size: Maximum number of documents to retrieve
            
        Returns:
            List of documents
        """
        try:
            # Ensure client is connected
            if not self.client:
                await self.connect()
            
            # Execute search with retry
            response = await self._execute_with_retry(
                self.client.search,
                index=self.index_name,
                query={"match_all": {}},
                size=size
            )
            
            # Process results
            results = []
            for hit in response["hits"]["hits"]:
                results.append(hit["_source"])
            
            logger.info(f"Retrieved {len(results)} documents from index")
            return results
                
        except Exception as e:
            logger.error(f"Error getting all documents: {e}")
            return []

# Import for randomization in backoff strategy
import random