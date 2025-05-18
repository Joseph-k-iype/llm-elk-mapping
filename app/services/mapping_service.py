"""
Enhanced mapping service with production-grade error handling, optimized I/O,
and robust job management for high-reliability operations.
"""

import logging
import asyncio
import time
import uuid
import random
from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
from app.models.mapping import MappingRequest, MappingResponse, MappingResult
from app.models.job import MappingJob, JobStatus
from app.services.azure_openai import AzureOpenAIService
from app.services.elasticsearch_service import ElasticsearchService
from app.services.vector_service import VectorService
from app.services.job_tracking_service import JobTrackingService
from app.utils.json_repair import repair_json

logger = logging.getLogger(__name__)

class MappingService:
    """
    Enhanced mapping service with robust error handling and optimized operations.
    """
    
    def __init__(self, 
                azure_service: AzureOpenAIService,
                es_service: ElasticsearchService,
                vector_service: VectorService):
        """
        Initialize the Mapping Service with proper dependency management.
        
        Args:
            azure_service: Azure OpenAI service
            es_service: Elasticsearch service
            vector_service: Vector service
        """
        self.azure_service = azure_service
        self.es_service = es_service
        self.vector_service = vector_service
        self.job_tracking_service = JobTrackingService()
        
        # Task management
        self.background_tasks = {}  # Track async tasks
        self.completed_tasks = {}   # Store completed task results
        self.task_cleanup_interval = 3600  # Clean up completed tasks every hour
        self.last_cleanup_time = time.time()
        
        # Rate limiting
        self.semaphore = asyncio.Semaphore(8)  # Limit concurrent mappings
        
        # Performance tracking
        self.timing_stats = {
            "map_business_term": [],
            "create_mapping_job": [],
            "process_mapping_job": []
        }
        
        # Start background cleanup task
        self._start_cleanup_task()
        
        logger.info("MappingService initialized")
    
    def _start_cleanup_task(self):
        """Start a background task to clean up completed tasks."""
        async def cleanup_worker():
            while True:
                try:
                    await self._cleanup_tasks()
                    await asyncio.sleep(self.task_cleanup_interval)
                except Exception as e:
                    logger.error(f"Error in cleanup worker: {e}")
                    await asyncio.sleep(300)  # Sleep 5 minutes before retrying
        
        # Start the worker
        asyncio.create_task(cleanup_worker())
    
    async def _cleanup_tasks(self):
        """Clean up completed tasks and perform maintenance."""
        current_time = time.time()
        
        # Only clean up every task_cleanup_interval seconds
        if current_time - self.last_cleanup_time < self.task_cleanup_interval:
            return
        
        try:
            # Clean up completed tasks older than 1 hour
            tasks_to_remove = []
            for job_id, data in self.completed_tasks.items():
                if current_time - data.get("completion_time", 0) > 3600:  # 1 hour
                    tasks_to_remove.append(job_id)
            
            # Remove old tasks
            for job_id in tasks_to_remove:
                if job_id in self.completed_tasks:
                    del self.completed_tasks[job_id]
            
            if tasks_to_remove:
                logger.info(f"Cleaned up {len(tasks_to_remove)} completed tasks")
            
            # Clean up any cancelled or done tasks from background_tasks
            tasks_to_remove = []
            for job_id, task in self.background_tasks.items():
                if task.done() or task.cancelled():
                    tasks_to_remove.append(job_id)
            
            for job_id in tasks_to_remove:
                if job_id in self.background_tasks:
                    del self.background_tasks[job_id]
            
            if tasks_to_remove:
                logger.info(f"Cleaned up {len(tasks_to_remove)} background tasks")
            
            # Clean up stale jobs in database
            try:
                await self.job_tracking_service.connect(self.es_service.client)
                cleaned_jobs = await self.job_tracking_service.cleanup_stale_jobs(max_age_hours=24)
                if cleaned_jobs > 0:
                    logger.info(f"Cleaned up {cleaned_jobs} stale jobs from database")
            except Exception as e:
                logger.error(f"Error cleaning up stale jobs: {e}")
            
            # Update last cleanup time
            self.last_cleanup_time = current_time
            
        except Exception as e:
            logger.error(f"Error in _cleanup_tasks: {e}")
    
    def _record_timing(self, operation: str, duration: float):
        """
        Record timing information for performance tracking.
        
        Args:
            operation: Name of the operation
            duration: Duration in seconds
        """
        if operation in self.timing_stats:
            # Keep only the last 100 timings
            if len(self.timing_stats[operation]) >= 100:
                self.timing_stats[operation].pop(0)
            
            self.timing_stats[operation].append(duration)
            
            # Log average timing every 10 records
            if len(self.timing_stats[operation]) % 10 == 0:
                avg_time = sum(self.timing_stats[operation]) / len(self.timing_stats[operation])
                logger.info(f"Average {operation} time over last {len(self.timing_stats[operation])} operations: {avg_time:.2f}s")
    
    async def map_business_term(self, request: MappingRequest) -> MappingResponse:
        """
        Map a business term to existing terms with robust error handling.
        
        Args:
            request: Mapping request
            
        Returns:
            Mapping response with results
        """
        start_time = time.time()
        
        try:
            async with self.semaphore:  # Limit concurrent executions
                logger.info(f"Processing mapping request for '{request.name}'")
                
                # Use the LangGraph workflow with AI agent evaluation
                try:
                    results = await self.vector_service.create_langgraph_workflow(request)
                    logger.info(f"LangGraph workflow completed with {len(results)} results")
                except Exception as e:
                    logger.error(f"Error in LangGraph workflow: {e}")
                    # Fallback to direct semantic search
                    logger.info("Falling back to direct semantic search")
                    results = await self.vector_service.semantic_search(request)
                
                # Add detailed reasoning for each result
                try:
                    enriched_results = await self._enrich_results_with_explanation(request, results)
                except Exception as e:
                    logger.error(f"Error enriching results: {e}")
                    # Continue with original results
                    enriched_results = results
                
                # Create response
                response = MappingResponse(
                    success=True,
                    message="Mapping completed successfully",
                    results=enriched_results,
                    query=request,
                    timestamp=datetime.now()
                )
                
                logger.info(f"Mapping completed with {len(results)} results")
                
                # Record timing
                self._record_timing("map_business_term", time.time() - start_time)
                
                return response
        except Exception as e:
            logger.error(f"Error mapping business term: {e}")
            duration = time.time() - start_time
            logger.warning(f"Failed mapping operation took {duration:.2f}s")
            
            # Return error response
            return MappingResponse(
                success=False,
                message=f"Error mapping business term: {str(e)}",
                results=[],
                query=request,
                timestamp=datetime.now()
            )
    
    async def create_mapping_job(self, job_id: str, process_id: str, request: MappingRequest) -> MappingJob:
        """
        Create a new asynchronous mapping job with reliable ID handling.
        
        Args:
            job_id: Job ID (required)
            process_id: Process ID (required)
            request: Mapping request
            
        Returns:
            Created job
        """
        start_time = time.time()
        
        try:
            # Connect to job tracking service
            await self.job_tracking_service.connect(self.es_service.client)
            
            # Validate IDs
            if not job_id:
                job_id = str(uuid.uuid4())
                logger.warning(f"No job ID provided, generated: {job_id}")
            
            if not process_id:
                process_id = str(uuid.uuid4())
                logger.warning(f"No process ID provided, generated: {process_id}")
            
            # Create job record with the provided IDs
            job = await self.job_tracking_service.create_job(job_id, process_id, request)
            
            # Start background task to process the mapping
            task = asyncio.create_task(self._process_mapping_job(job))
            
            # Store task reference to prevent it from being garbage collected
            self.background_tasks[job.id] = task
            
            logger.info(f"Created mapping job {job.id} for process {job.process_id}")
            
            # Record timing
            self._record_timing("create_mapping_job", time.time() - start_time)
            
            return job
        except Exception as e:
            logger.error(f"Error creating mapping job: {e}")
            
            # Record timing for failed operation
            self._record_timing("create_mapping_job", time.time() - start_time)
            
            # Re-raise for controller to handle
            raise
    
    async def _process_mapping_job(self, job: MappingJob):
        """
        Process a mapping job in the background with robust error handling.
        
        Args:
            job: Mapping job to process
        """
        start_time = time.time()
        
        try:
            # Update status to in progress
            await self.job_tracking_service.connect(self.es_service.client)
            await self.job_tracking_service.update_job_status(job.id, JobStatus.IN_PROGRESS)
            
            logger.info(f"Started processing mapping job {job.id} (process ID: {job.process_id}) for '{job.request.name}'")
            
            # Process with retry and timeout
            max_retries = 3
            timeout_seconds = 120  # 2 minutes timeout
            
            for attempt in range(max_retries):
                try:
                    # Process the mapping request with timeout
                    results = await asyncio.wait_for(
                        self.vector_service.create_langgraph_workflow(job.request),
                        timeout=timeout_seconds
                    )
                    
                    logger.info(f"Mapping workflow completed for job {job.id} with {len(results)} results")
                    
                    # Add detailed reasoning with timeout
                    try:
                        enriched_results = await asyncio.wait_for(
                            self._enrich_results_with_explanation(job.request, results),
                            timeout=60  # 1 minute timeout for enrichment
                        )
                    except asyncio.TimeoutError:
                        logger.warning(f"Explanation enrichment timed out for job {job.id}, using original results")
                        enriched_results = results
                    except Exception as enrichment_error:
                        logger.error(f"Error enriching results for job {job.id}: {enrichment_error}")
                        enriched_results = results
                    
                    # Update job with completed status and results
                    await self.job_tracking_service.update_job_status(
                        job.id, 
                        JobStatus.COMPLETED,
                        results=enriched_results
                    )
                    
                    # Record job completion
                    self.completed_tasks[job.id] = {
                        "status": "completed",
                        "completion_time": time.time(),
                        "result_count": len(enriched_results)
                    }
                    
                    logger.info(f"Completed mapping job {job.id} with {len(enriched_results)} results")
                    
                    # Record timing
                    self._record_timing("process_mapping_job", time.time() - start_time)
                    
                    # Return successfully
                    return
                    
                except asyncio.TimeoutError:
                    logger.warning(f"Timeout processing job {job.id} (attempt {attempt+1}/{max_retries})")
                    
                    # If this is the last attempt, mark as failed
                    if attempt == max_retries - 1:
                        logger.error(f"Job {job.id} failed after {max_retries} timeout attempts")
                        await self.job_tracking_service.update_job_status(
                            job.id,
                            JobStatus.FAILED,
                            error=f"Operation timed out after {timeout_seconds}s ({max_retries} attempts)"
                        )
                    else:
                        # Wait before retrying
                        await asyncio.sleep(1 * (attempt + 1))  # Progressive delay
                
                except Exception as processing_error:
                    logger.error(f"Error processing job {job.id} (attempt {attempt+1}/{max_retries}): {processing_error}")
                    
                    # If this is the last attempt, mark as failed
                    if attempt == max_retries - 1:
                        logger.error(f"Job {job.id} failed after {max_retries} attempts")
                        await self.job_tracking_service.update_job_status(
                            job.id,
                            JobStatus.FAILED,
                            error=str(processing_error)
                        )
                    else:
                        # Wait before retrying
                        await asyncio.sleep(1 * (attempt + 1))  # Progressive delay
            
        except Exception as e:
            logger.error(f"Unhandled error processing mapping job {job.id}: {e}")
            
            # Ensure job is marked as failed
            try:
                await self.job_tracking_service.update_job_status(
                    job.id,
                    JobStatus.FAILED,
                    error=f"Unhandled error: {str(e)}"
                )
            except Exception as update_error:
                logger.error(f"Error updating failed job status: {update_error}")
        
        finally:
            # Clean up task reference
            if job.id in self.background_tasks:
                del self.background_tasks[job.id]
            
            # Record failed timing if not already recorded
            if start_time > 0 and "process_mapping_job" not in self.timing_stats:
                self._record_timing("process_mapping_job", time.time() - start_time)
    
    async def get_mapping_job(self, job_id: str) -> Optional[MappingJob]:
        """
        Get a mapping job by ID with robust error handling.
        
        Args:
            job_id: ID of the job to get
            
        Returns:
            Job or None if not found
        """
        try:
            await self.job_tracking_service.connect(self.es_service.client)
            return await self.job_tracking_service.get_job(job_id)
        except Exception as e:
            logger.error(f"Error retrieving mapping job {job_id}: {e}")
            return None
    
    async def get_mapping_job_by_process_id(self, process_id: str) -> Optional[MappingJob]:
        """
        Get a mapping job by process ID with robust error handling.
        
        Args:
            process_id: Process ID to search for
            
        Returns:
            Job or None if not found
        """
        try:
            await self.job_tracking_service.connect(self.es_service.client)
            return await self.job_tracking_service.get_job_by_process_id(process_id)
        except Exception as e:
            logger.error(f"Error retrieving mapping job by process ID {process_id}: {e}")
            return None
    
    async def get_mapping_jobs(self, status: Optional[JobStatus] = None, 
                            page: int = 1, page_size: int = 10) -> Tuple[List[MappingJob], int]:
        """
        Get mapping jobs with robust error handling and pagination.
        
        Args:
            status: Filter by status (optional)
            page: Page number (1-based)
            page_size: Number of results per page
            
        Returns:
            Tuple of (list of jobs, total count)
        """
        try:
            await self.job_tracking_service.connect(self.es_service.client)
            return await self.job_tracking_service.get_jobs(status, page, page_size)
        except Exception as e:
            logger.error(f"Error retrieving mapping jobs: {e}")
            return [], 0
    
    async def _enrich_results_with_explanation(self, request: MappingRequest, results: List[MappingResult]) -> List[MappingResult]:
        """
        Enrich mapping results with detailed explanations using the LLM.
        
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
        
        Format your response as follows:
        Match 1: [Detailed explanation for first match]
        Match 2: [Detailed explanation for second match]
        Match 3: [Detailed explanation for third match]
        ... and so on for all matches.
        
        Do not include any other text, headers, or formatting beyond these match explanations.
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
        
        user_prompt += "\nPlease provide a concise explanation for each match as specified in the instructions."
        
        # Create messages for the LLM
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Try to get explanations with retry logic
        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                # Get explanations from LLM - use a simpler model for faster processing
                explanation_text = await self.azure_service.generate_completion(
                    messages,
                    temperature=0.1,  # Lower temperature for consistent, precise explanations
                    max_tokens=1500
                )
                
                # Parse explanations and add to results
                explanations = self._parse_explanations(explanation_text, len(results))
                
                # Update results with explanations
                for i, result in enumerate(results):
                    if i < len(explanations):
                        # Clone the result to avoid modifying the original
                        updated_result = MappingResult(
                            term_id=result.term_id,
                            term_name=result.term_name,
                            similarity_score=result.similarity_score,
                            confidence=result.confidence,
                            mapping_type=result.mapping_type,
                            matched_attributes=result.matched_attributes
                        )
                        
                        # Add explanation
                        setattr(updated_result, "explanation", explanations[i])
                        
                        # Add any custom attributes from the original result
                        for attr_name in dir(result):
                            if (not attr_name.startswith('_') and 
                                attr_name not in ['term_id', 'term_name', 'similarity_score', 
                                                 'confidence', 'mapping_type', 'matched_attributes'] and
                                not callable(getattr(result, attr_name))):
                                setattr(updated_result, attr_name, getattr(result, attr_name))
                        
                        results[i] = updated_result
                
                return results
                
            except Exception as e:
                logger.error(f"Error enriching results with explanations (attempt {attempt+1}/{max_retries+1}): {e}")
                
                if attempt < max_retries:
                    # Wait before retrying
                    backoff = (2 ** attempt) + (random.random() * 0.5)
                    logger.info(f"Retrying explanation in {backoff:.2f}s...")
                    await asyncio.sleep(backoff)
                else:
                    # Return original results if all retries fail
                    logger.warning("All explanation attempts failed, returning original results")
                    return results
    
    def _parse_explanations(self, explanation_text: str, num_results: int) -> List[str]:
        """
        Parse explanations from LLM response with robust error handling.
        
        Args:
            explanation_text: Text containing explanations
            num_results: Number of results to expect explanations for
            
        Returns:
            List of explanations
        """
        explanations = []
        
        # Try parsing with regex pattern
        try:
            # Find all "Match X:" sections
            pattern = r"Match (\d+):?\s*(.*?)(?=Match \d+:|$)"
            matches = list(re.finditer(pattern, explanation_text, re.DOTALL))
            
            # Process each match
            for match in matches:
                match_num = int(match.group(1))
                explanation = match.group(2).strip()
                
                # Store at the correct index
                while len(explanations) < match_num:
                    explanations.append("No detailed explanation available.")
                    
                if match_num <= len(explanations):
                    explanations[match_num-1] = explanation
                else:
                    explanations.append(explanation)
        except Exception as e:
            logger.error(f"Error parsing explanations with regex: {e}")
            
            # Fallback: try line-by-line parsing
            try:
                lines = explanation_text.split('\n')
                current_explanation = []
                current_match = 0
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                        
                    # Check if this is a new match header
                    match_header = re.match(r"Match (\d+):?", line)
                    if match_header:
                        # Save previous explanation if any
                        if current_explanation and current_match > 0:
                            while len(explanations) < current_match:
                                explanations.append("No detailed explanation available.")
                            explanations.append(" ".join(current_explanation))
                            
                        # Start new explanation
                        current_match = int(match_header.group(1))
                        current_explanation = []
                        
                        # Add the rest of this line to the explanation
                        rest_of_line = line[match_header.end():].strip()
                        if rest_of_line:
                            current_explanation.append(rest_of_line)
                    else:
                        # Add to current explanation
                        if current_match > 0:
                            current_explanation.append(line)
                
                # Add the last explanation if any
                if current_explanation and current_match > 0:
                    while len(explanations) < current_match:
                        explanations.append("No detailed explanation available.")
                    explanations.append(" ".join(current_explanation))
                
            except Exception as e2:
                logger.error(f"Error in fallback explanation parsing: {e2}")
        
        # Ensure we have the right number of explanations
        while len(explanations) < num_results:
            explanations.append("No detailed explanation available.")
        
        # Trim to the number of results
        return explanations[:num_results]
    
    async def get_all_business_terms(self) -> List[Dict[str, Any]]:
        """
        Get all indexed business terms with robust error handling.
        
        Returns:
            List of business terms
        """
        try:
            logger.info("Retrieving all business terms")
            terms = await self.es_service.get_all_documents(size=1000)
            logger.info(f"Retrieved {len(terms)} business terms")
            return terms
        except Exception as e:
            logger.error(f"Error retrieving business terms: {e}")
            return []