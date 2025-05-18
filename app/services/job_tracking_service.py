"""
Enhanced Job Tracking Service for production environments.

Provides reliable asynchronous job tracking with robust error recovery
and optimized Elasticsearch operations.
"""

import logging
import asyncio
import time
import random
import json
from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime
from elasticsearch import AsyncElasticsearch, NotFoundError, ConnectionError, RequestError, TransportError
from elasticsearch.helpers import async_bulk
from app.models.job import MappingJob, JobStatus, MappingJobRequest
from app.models.mapping import MappingRequest, MappingResult
from app.core.environment import get_os_env

logger = logging.getLogger(__name__)

class JobTrackingService:
    """Production-grade service for tracking mapping jobs in Elasticsearch with resilient operations."""
    
    def __init__(self, client: Optional[AsyncElasticsearch] = None):
        """
        Initialize the job tracking service with configurable settings.
        
        Args:
            client: Elasticsearch client (optional)
        """
        # Get environment
        env = get_os_env()
        
        # Configure service 
        self.client = client
        self.index_name = env.get("JOB_INDEX_NAME", "mapping_jobs")
        self.index_alias = f"{self.index_name}_alias"
        self.max_retries = int(env.get("JOB_SERVICE_MAX_RETRIES", "5"))
        self.initial_backoff = float(env.get("JOB_SERVICE_INITIAL_BACKOFF", "1.0"))
        self.max_backoff = float(env.get("JOB_SERVICE_MAX_BACKOFF", "30.0"))
        self.bulk_batch_size = int(env.get("JOB_SERVICE_BULK_SIZE", "50"))
        
        # Performance settings
        self.use_refresh_window = True  # Batch refresh operations
        self.refresh_interval = float(env.get("JOB_REFRESH_INTERVAL", "5.0"))  # seconds
        self.last_refresh_time = 0
        
        # In-memory cache for job data (reduces ES queries)
        self.job_cache = {}
        self.job_cache_ttl = int(env.get("JOB_CACHE_TTL", "60"))  # seconds
        self.job_cache_timestamps = {}
        self.max_cache_size = int(env.get("JOB_CACHE_SIZE", "100"))
        
        # Status
        self._connected = False
        self._connection_lock = asyncio.Lock()  # Prevent multiple concurrent connection attempts
        
        logger.info(f"JobTrackingService initialized with index: {self.index_name}")
    
    async def connect(self, client: Optional[AsyncElasticsearch] = None, max_retries: int = 3) -> bool:
        """
        Connect to Elasticsearch with retry logic and use provided client if available.
        
        Args:
            client: Elasticsearch client to use (optional)
            max_retries: Maximum number of connection attempts
            
        Returns:
            bool: True if connected successfully
        """
        # Use lock to prevent multiple concurrent connection attempts
        async with self._connection_lock:
            # If already connected, return
            if self._connected and self.client:
                return True
                
            # Use provided client if available
            if client:
                self.client = client
                self._connected = True
                logger.info("Connected to Elasticsearch using provided client")
                return True
                
            # Otherwise create a new connection
            # Get environment
            env = get_os_env()
            
            # Get Elasticsearch configuration
            hosts_str = env.get("ELASTICSEARCH_HOSTS", '["http://localhost:9200"]')
            try:
                if hosts_str.startswith('['):
                    hosts = json.loads(hosts_str.replace("'", '"'))
                else:
                    hosts = [hosts_str]
            except:
                hosts = ["http://localhost:9200"]
                
            # Clean up hosts
            clean_hosts = []
            for host in hosts:
                if isinstance(host, str):
                    host = host.strip('\'"')
                    if not host.startswith(('http://', 'https://')):
                        host = f"https://{host}"
                    clean_hosts.append(host)
            
            # Get authentication details
            username = env.get("ELASTICSEARCH_USERNAME", None)
            password = env.get("ELASTICSEARCH_PASSWORD", None)
            
            # Create connection parameters
            conn_params = {
                "timeout": 30,
                "max_retries": 3,
                "retry_on_timeout": True,
                "retry_on_status": [429, 500, 502, 503, 504],
                "connections_per_node": 10,  # Increased connection pool size
                "verify_certs": False,
                "ssl_show_warn": False
            }
            
            # Add authentication if provided
            if username and password:
                conn_params["basic_auth"] = (username, password)
            
            # Try to connect with retries
            for attempt in range(max_retries):
                try:
                    # Create client
                    self.client = AsyncElasticsearch(
                        clean_hosts,
                        **conn_params
                    )
                    
                    # Test connection
                    info = await self.client.info()
                    logger.info(f"Connected to Elasticsearch version {info['version']['number']}")
                    self._connected = True
                    return True
                except Exception as e:
                    logger.error(f"Failed to connect to Elasticsearch (attempt {attempt+1}/{max_retries}): {e}")
                    if attempt + 1 < max_retries:
                        wait_time = self.initial_backoff * (2 ** attempt) + random.uniform(0, 0.5)
                        logger.info(f"Retrying in {wait_time:.2f} seconds...")
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error("Maximum connection attempts reached")
                        raise
            
            return False  # Should not reach here if max_retries is hit (will raise instead)
    
    async def _execute_with_retry(self, operation_func, *args, max_retries=None, **kwargs):
        """
        Execute an Elasticsearch operation with retry logic.
        
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
        
        # Ensure client is connected
        if not self._connected or not self.client:
            await self.connect()
        
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                # Execute operation
                return await operation_func(*args, **kwargs)
                
            except ConnectionError as e:
                logger.warning(f"Connection error (attempt {attempt+1}/{max_retries+1}): {e}")
                last_exception = e
                
                # Reconnect on connection errors
                if attempt < max_retries:
                    self._connected = False
                    try:
                        await self.connect(max_retries=1)
                    except:
                        pass  # Continue with retry even if reconnect fails
                
            except (RequestError, TransportError) as e:
                logger.warning(f"Request/Transport error (attempt {attempt+1}/{max_retries+1}): {e}")
                last_exception = e
                
            except Exception as e:
                logger.error(f"Unexpected error in ES operation: {e}")
                last_exception = e
                
                # Only retry certain types of errors
                if not (isinstance(e, (ConnectionError, RequestError, TransportError))):
                    raise
            
            # If we get here, an error occurred - wait before retrying
            if attempt < max_retries:
                # Calculate backoff with jitter
                backoff = min(
                    self.max_backoff,
                    self.initial_backoff * (2 ** attempt)
                )
                jitter = random.uniform(0, 0.5)
                wait_time = backoff + jitter
                
                logger.info(f"Retrying in {wait_time:.2f} seconds...")
                await asyncio.sleep(wait_time)
        
        # If we get here, all retries failed
        logger.error(f"All {max_retries+1} attempts failed")
        if last_exception:
            raise last_exception
        else:
            raise RuntimeError("Maximum retries exceeded")
    
    async def create_index(self, force: bool = False):
        """
        Create the mapping jobs index with optimized settings if it doesn't exist.
        
        Args:
            force: Whether to force recreate the index (delete if exists)
        """
        if not self._connected:
            await self.connect()
        
        try:
            # Check if index exists
            exists = await self._execute_with_retry(self.client.indices.exists, index=self.index_name)
            
            if exists and force:
                logger.info(f"Deleting existing index '{self.index_name}'")
                await self._execute_with_retry(self.client.indices.delete, index=self.index_name)
                exists = False
            
            if not exists:
                # Define index settings with optimized configuration
                settings = {
                    "settings": {
                        "number_of_shards": 1,
                        "number_of_replicas": 1,
                        "refresh_interval": "5s",  # Slightly delayed refresh for write performance
                        "index.mapping.total_fields.limit": 2000,  # Increased field limit
                        "index.mapping.nested_fields.limit": 100,
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
                            "process_id": {"type": "keyword"},
                            "status": {"type": "keyword"},
                            "created_at": {"type": "date"},
                            "updated_at": {"type": "date"},
                            "completed_at": {"type": "date"},
                            "request": {
                                "properties": {
                                    "name": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                                    "description": {"type": "text"},
                                    "example": {"type": "text"},
                                    "cdm": {"type": "keyword"},
                                    "process_name": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                                    "process_description": {"type": "text"}
                                }
                            },
                            "results": {
                                "type": "nested",  # Use nested for better query support
                                "properties": {
                                    "term_id": {"type": "keyword"},
                                    "term_name": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                                    "similarity_score": {"type": "float"},
                                    "confidence": {"type": "float"},
                                    "mapping_type": {"type": "keyword"},
                                    "matched_attributes": {"type": "keyword"},
                                    "explanation": {"type": "text"}
                                }
                            },
                            "error": {"type": "text"}
                        }
                    }
                }
                
                # Create index
                await self._execute_with_retry(
                    self.client.indices.create,
                    index=self.index_name,
                    body=settings
                )
                logger.info(f"Created index '{self.index_name}' with optimized settings")
                
                # Create alias
                try:
                    await self._execute_with_retry(
                        self.client.indices.put_alias,
                        index=self.index_name,
                        name=self.index_alias
                    )
                    logger.info(f"Created alias '{self.index_alias}' for index '{self.index_name}'")
                except Exception as e:
                    logger.warning(f"Error creating index alias: {e}")
        except Exception as e:
            logger.error(f"Error creating index: {e}")
            raise
    
    async def _maybe_refresh_index(self, force: bool = False):
        """
        Refresh the index if needed, batching refresh operations for performance.
        
        Args:
            force: Whether to force a refresh regardless of timing
        """
        current_time = time.time()
        
        if force or (current_time - self.last_refresh_time > self.refresh_interval):
            try:
                await self._execute_with_retry(
                    self.client.indices.refresh,
                    index=self.index_name
                )
                self.last_refresh_time = current_time
            except Exception as e:
                logger.warning(f"Error refreshing index: {e}")
    
    def _update_cache(self, job_id: str, job: MappingJob):
        """
        Update the job cache with a job.
        
        Args:
            job_id: ID of the job
            job: Job data
        """
        # Add to cache with timestamp
        self.job_cache[job_id] = job
        self.job_cache_timestamps[job_id] = time.time()
        
        # Clean up old cache entries if cache is too large
        if len(self.job_cache) > self.max_cache_size:
            # Find oldest entries
            sorted_items = sorted(
                self.job_cache_timestamps.items(),
                key=lambda x: x[1]
            )
            
            # Remove oldest
            to_remove = len(self.job_cache) - self.max_cache_size
            for i in range(to_remove):
                old_id = sorted_items[i][0]
                if old_id in self.job_cache:
                    del self.job_cache[old_id]
                if old_id in self.job_cache_timestamps:
                    del self.job_cache_timestamps[old_id]
    
    def _get_from_cache(self, job_id: str) -> Optional[MappingJob]:
        """
        Get a job from the cache if available and not expired.
        
        Args:
            job_id: ID of the job to get
            
        Returns:
            Job data or None if not in cache or expired
        """
        # Check if in cache and not expired
        if job_id in self.job_cache and job_id in self.job_cache_timestamps:
            timestamp = self.job_cache_timestamps[job_id]
            current_time = time.time()
            
            if current_time - timestamp <= self.job_cache_ttl:
                return self.job_cache[job_id]
            else:
                # Expired - remove from cache
                del self.job_cache[job_id]
                del self.job_cache_timestamps[job_id]
        
        return None
    
    async def create_job(self, job_id: str, process_id: str, request: MappingRequest) -> MappingJob:
        """
        Create a new mapping job with robust error handling.
        
        Args:
            job_id: Job ID (required)
            process_id: Process ID (required)
            request: Mapping request
            
        Returns:
            Created job
        """
        if not self._connected:
            await self.connect()
        
        # Ensure index exists
        await self.create_index()
        
        # Check if job with this ID already exists
        try:
            existing_job = await self._execute_with_retry(
                self.client.get,
                index=self.index_name,
                id=job_id
            )
            logger.warning(f"Job with ID {job_id} already exists")
            # Return the existing job
            job = MappingJob(**existing_job["_source"])
            
            # Update cache
            self._update_cache(job_id, job)
            
            return job
        except NotFoundError:
            # Job doesn't exist, proceed with creation
            pass
        except Exception as e:
            logger.error(f"Error checking for existing job: {e}")
            # Continue with creation - if there's a conflict, ES will handle it
        
        # Create job with specified IDs
        job = MappingJob(
            id=job_id,
            process_id=process_id,
            request=request
        )
        
        # Save to Elasticsearch with retry
        try:
            await self._execute_with_retry(
                self.client.index,
                index=self.index_name,
                id=job.id,
                document=job.model_dump(),
                refresh=True  # Force refresh for immediate visibility
            )
            logger.info(f"Created mapping job with ID: {job.id}, process ID: {job.process_id}")
            
            # Update cache
            self._update_cache(job_id, job)
            
            return job
        except Exception as e:
            logger.error(f"Error creating mapping job: {e}")
            raise
    
    async def update_job_status(self, job_id: str, status: JobStatus, 
                             results: Optional[List[MappingResult]] = None,
                             error: Optional[str] = None) -> Optional[MappingJob]:
        """
        Update a job's status with optimized operations.
        
        Args:
            job_id: ID of the job to update
            status: New status
            results: Mapping results (if completed)
            error: Error message (if failed)
            
        Returns:
            Updated job or None if not found
        """
        if not self._connected:
            await self.connect()
        
        # Get current job
        try:
            # Check cache first
            job = self._get_from_cache(job_id)
            
            if not job:
                # Not in cache, get from ES
                response = await self._execute_with_retry(
                    self.client.get,
                    index=self.index_name,
                    id=job_id
                )
                job_data = response["_source"]
                
                # Convert to MappingJob
                job = MappingJob(**job_data)
            
            # Update status
            job.update_status(status)
            
            # Add results or error if provided
            if results is not None:
                job.results = results
            if error is not None:
                job.error = error
            
            # Update cache
            self._update_cache(job_id, job)
            
            # Save to Elasticsearch with optimized refresh
            await self._execute_with_retry(
                self.client.index,
                index=self.index_name,
                id=job.id,
                document=job.model_dump(),
                refresh="wait_for" if status in [JobStatus.COMPLETED, JobStatus.FAILED] else False
            )
            
            # Only force refresh for terminal statuses
            if status in [JobStatus.COMPLETED, JobStatus.FAILED]:
                await self._maybe_refresh_index(force=True)
            else:
                # Schedule refresh if in window
                await self._maybe_refresh_index()
            
            logger.info(f"Updated job {job_id} status to {status}")
            return job
        except NotFoundError:
            logger.error(f"Job {job_id} not found")
            return None
        except Exception as e:
            logger.error(f"Error updating job {job_id}: {e}")
            raise
    
    async def get_job(self, job_id: str) -> Optional[MappingJob]:
        """
        Get a job by ID with cache support.
        
        Args:
            job_id: ID of the job to get
            
        Returns:
            Job or None if not found
        """
        if not self._connected:
            await self.connect()
        
        # Check cache first
        job = self._get_from_cache(job_id)
        if job:
            return job
        
        try:
            response = await self._execute_with_retry(
                self.client.get,
                index=self.index_name,
                id=job_id
            )
            job_data = response["_source"]
            
            # Convert to MappingJob
            job = MappingJob(**job_data)
            
            # Update cache
            self._update_cache(job_id, job)
            
            return job
        except NotFoundError:
            logger.warning(f"Job {job_id} not found")
            return None
        except Exception as e:
            logger.error(f"Error getting job {job_id}: {e}")
            raise
    
    async def get_job_by_process_id(self, process_id: str) -> Optional[MappingJob]:
        """
        Get a job by process ID with optimized query.
        
        Args:
            process_id: Process ID to search for
            
        Returns:
            Job or None if not found
        """
        if not self._connected:
            await self.connect()
        
        try:
            # Search for a job with the given process ID using term query for exact match
            response = await self._execute_with_retry(
                self.client.search,
                index=self.index_name,
                body={
                    "query": {"term": {"process_id": process_id}},
                    "size": 1
                }
            )
            
            # Check if a job was found
            if response["hits"]["total"]["value"] > 0:
                job_data = response["hits"]["hits"][0]["_source"]
                job_id = response["hits"]["hits"][0]["_id"]
                
                job = MappingJob(**job_data)
                
                # Update cache
                self._update_cache(job_id, job)
                
                return job
            else:
                logger.info(f"No job found with process ID: {process_id}")
                return None
        except Exception as e:
            logger.error(f"Error getting job by process ID {process_id}: {e}")
            raise
    
    async def get_jobs(self, status: Optional[JobStatus] = None, 
                     page: int = 1, page_size: int = 10) -> Tuple[List[MappingJob], int]:
        """
        Get jobs with optimized pagination and sorting.
        
        Args:
            status: Filter by status (optional)
            page: Page number (1-based)
            page_size: Number of results per page
            
        Returns:
            Tuple of (list of jobs, total count)
        """
        if not self._connected:
            await self.connect()
        
        # Calculate from index
        from_idx = (page - 1) * page_size
        
        # Build query with optimized structure
        query_body = {
            "sort": [{"created_at": {"order": "desc"}}],
            "from": from_idx,
            "size": page_size,
            "track_total_hits": True,  # Ensure we get accurate count
        }
        
        if status:
            query_body["query"] = {"term": {"status": status}}
        
        # Execute search with retry
        try:
            response = await self._execute_with_retry(
                self.client.search,
                index=self.index_name,
                body=query_body
            )
            
            # Extract jobs and total
            jobs = []
            for hit in response["hits"]["hits"]:
                job_data = hit["_source"]
                job_id = hit["_id"]
                job = MappingJob(**job_data)
                
                # Update cache
                self._update_cache(job_id, job)
                
                jobs.append(job)
            
            total = response["hits"]["total"]["value"]
            
            return jobs, total
        except Exception as e:
            logger.error(f"Error getting jobs: {e}")
            raise
    
    async def delete_job(self, job_id: str) -> bool:
        """
        Delete a job with cache cleanup.
        
        Args:
            job_id: ID of the job to delete
            
        Returns:
            True if deleted, False if not found
        """
        if not self._connected:
            await self.connect()
        
        try:
            # Delete from Elasticsearch
            response = await self._execute_with_retry(
                self.client.delete,
                index=self.index_name,
                id=job_id,
                refresh=True  # Force refresh for immediate visibility
            )
            
            # Clean up cache
            if job_id in self.job_cache:
                del self.job_cache[job_id]
            if job_id in self.job_cache_timestamps:
                del self.job_cache_timestamps[job_id]
            
            logger.info(f"Deleted job {job_id}")
            return True
        except NotFoundError:
            logger.warning(f"Job {job_id} not found for deletion")
            return False
        except Exception as e:
            logger.error(f"Error deleting job {job_id}: {e}")
            raise
    
    async def bulk_update_jobs(self, jobs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Bulk update jobs with optimized operations.
        
        Args:
            jobs: List of job data to update (must include id)
            
        Returns:
            Dictionary with success and error counts
        """
        if not self._connected:
            await self.connect()
        
        if not jobs:
            return {"success": 0, "errors": 0}
        
        # Prepare bulk operations
        operations = []
        
        for job in jobs:
            if "id" not in job:
                logger.warning("Job missing ID, skipping")
                continue
                
            job_id = job["id"]
            operations.append({"update": {"_index": self.index_name, "_id": job_id}})
            operations.append({"doc": job})
            
            # Update cache if available
            if job_id in self.job_cache:
                cached_job = self.job_cache[job_id]
                for key, value in job.items():
                    if key != "id":
                        setattr(cached_job, key, value)
                self.job_cache_timestamps[job_id] = time.time()
        
        if not operations:
            return {"success": 0, "errors": 0}
        
        try:
            # Execute bulk operation with retry
            result = await self._execute_with_retry(
                self.client.bulk,
                operations=operations,
                refresh=False  # Don't force refresh for bulk operations
            )
            
            # Schedule refresh
            await self._maybe_refresh_index()
            
            # Count successes and errors
            success_count = 0
            error_count = 0
            
            if "items" in result:
                for item in result["items"]:
                    if "update" in item and item["update"].get("status", 500) < 400:
                        success_count += 1
                    else:
                        error_count += 1
            
            logger.info(f"Bulk updated {success_count} jobs with {error_count} errors")
            return {"success": success_count, "errors": error_count}
        except Exception as e:
            logger.error(f"Error in bulk update: {e}")
            return {"success": 0, "errors": len(jobs)}
    
    async def cleanup_stale_jobs(self, max_age_hours: int = 24) -> int:
        """
        Clean up stale jobs that have been pending for too long.
        
        Args:
            max_age_hours: Maximum age in hours for pending jobs
            
        Returns:
            Number of jobs cleaned up
        """
        if not self._connected:
            await self.connect()
        
        try:
            # Calculate cutoff time
            cutoff_time = (datetime.now() - datetime.timedelta(hours=max_age_hours)).isoformat()
            
            # Query for stale pending jobs
            response = await self._execute_with_retry(
                self.client.search,
                index=self.index_name,
                body={
                    "query": {
                        "bool": {
                            "must": [
                                {"term": {"status": JobStatus.PENDING.value}},
                                {"range": {"created_at": {"lt": cutoff_time}}}
                            ]
                        }
                    },
                    "size": 100  # Process in batches
                }
            )
            
            # Update stale jobs to FAILED status
            if response["hits"]["total"]["value"] > 0:
                bulk_ops = []
                
                for hit in response["hits"]["hits"]:
                    job_id = hit["_id"]
                    
                    # Add to bulk operations
                    bulk_ops.append({"update": {"_index": self.index_name, "_id": job_id}})
                    bulk_ops.append({
                        "doc": {
                            "status": JobStatus.FAILED.value,
                            "updated_at": datetime.now().isoformat(),
                            "error": "Job timed out after being pending too long"
                        }
                    })
                    
                    # Remove from cache
                    if job_id in self.job_cache:
                        del self.job_cache[job_id]
                    if job_id in self.job_cache_timestamps:
                        del self.job_cache_timestamps[job_id]
                
                # Execute bulk update
                if bulk_ops:
                    await self._execute_with_retry(
                        self.client.bulk,
                        operations=bulk_ops,
                        refresh=True
                    )
                
                cleaned_up = len(bulk_ops) // 2  # Each job has 2 operations
                logger.info(f"Cleaned up {cleaned_up} stale jobs")
                return cleaned_up
            else:
                logger.info("No stale jobs to clean up")
                return 0
        except Exception as e:
            logger.error(f"Error cleaning up stale jobs: {e}")
            return 0