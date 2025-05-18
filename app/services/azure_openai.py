"""
Enhanced Azure OpenAI service with production-grade token management,
retry strategies, and robust error handling.
"""

import os
import logging
import time
import asyncio
import json
import random
from typing import List, Dict, Any, Optional, Tuple, Union
from openai import AzureOpenAI
from openai.types.chat import ChatCompletion
from openai.types.embedding import Embedding
from azure.identity import DefaultAzureCredential, ClientSecretCredential
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    wait_random_exponential,
    before_sleep_log
)
from app.core.auth_helper import get_azure_token_provider, refresh_token_if_needed
from app.core.environment import get_os_env

logger = logging.getLogger(__name__)

class RateLimiter:
    """Rate limiter for API requests."""
    
    def __init__(self, max_requests_per_minute: int = 300, max_tokens_per_minute: int = 240000):
        """
        Initialize the rate limiter.
        
        Args:
            max_requests_per_minute: Maximum requests per minute (RPM)
            max_tokens_per_minute: Maximum tokens per minute (TPM)
        """
        self.max_rpm = max_requests_per_minute
        self.max_tpm = max_tokens_per_minute
        self.request_timestamps = []
        self.token_timestamps = []
        self.tokens_used = []
        self.window_seconds = 60  # 1 minute sliding window
        self._lock = asyncio.Lock()
    
    async def wait_if_needed(self, estimated_tokens: int = 1000):
        """
        Wait if rate limits would be exceeded.
        
        Args:
            estimated_tokens: Estimated tokens to be used in this request
        """
        current_time = time.time()
        
        async with self._lock:
            # Clean up old timestamps
            self.request_timestamps = [t for t in self.request_timestamps 
                                      if current_time - t < self.window_seconds]
            
            # Clean up old token usage
            recent_tokens = []
            recent_token_timestamps = []
            for t, tokens in zip(self.token_timestamps, self.tokens_used):
                if current_time - t < self.window_seconds:
                    recent_token_timestamps.append(t)
                    recent_tokens.append(tokens)
            
            self.token_timestamps = recent_token_timestamps
            self.tokens_used = recent_tokens
            
            # Calculate current rates
            current_rpm = len(self.request_timestamps)
            current_tpm = sum(self.tokens_used)
            
            logger.debug(f"Current RPM: {current_rpm}/{self.max_rpm}, TPM: {current_tpm}/{self.max_tpm}")
            
            # Check if we need to wait
            if current_rpm >= self.max_rpm or current_tpm + estimated_tokens >= self.max_tpm:
                # How long to wait - time until oldest request/token expires from window
                wait_time = 0
                
                if current_rpm >= self.max_rpm and self.request_timestamps:
                    oldest_request = min(self.request_timestamps)
                    request_wait = oldest_request + self.window_seconds - current_time
                    wait_time = max(wait_time, request_wait)
                
                if current_tpm + estimated_tokens >= self.max_tpm and self.token_timestamps:
                    oldest_token = min(self.token_timestamps)
                    token_wait = oldest_token + self.window_seconds - current_time
                    wait_time = max(wait_time, token_wait)
                
                if wait_time > 0:
                    logger.info(f"Rate limit approaching, waiting {wait_time:.2f}s before proceeding")
                    await asyncio.sleep(wait_time + 0.1)  # Add small buffer
            
            # Record this request
            self.request_timestamps.append(current_time)
    
    def record_tokens(self, total_tokens: int):
        """
        Record token usage for rate limiting.
        
        Args:
            total_tokens: Total tokens used in a request
        """
        current_time = time.time()
        self.token_timestamps.append(current_time)
        self.tokens_used.append(total_tokens)


class AzureOpenAIService:
    """Enhanced service for interacting with Azure OpenAI models with production-grade reliability."""
    
    def __init__(self):
        """Initialize the Azure OpenAI service with robust configuration."""
        # Get environment
        env = get_os_env()
        
        # Get configuration from environment
        self.endpoint = env.get("AZURE_OPENAI_ENDPOINT", "")
        self.api_version = env.get("AZURE_API_VERSION", "2023-05-15")
        self.embedding_model = env.get("AZURE_EMBEDDING_MODEL", "text-embedding-3-large")
        self.embedding_deployment = env.get("AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-3-large")
        self.llm_model = env.get("AZURE_LLM_MODEL", "gpt-4o-mini") 
        self.llm_deployment = env.get("AZURE_LLM_DEPLOYMENT", "gpt-4o-mini")
        
        # Rate limiting settings
        self.rate_limit_rpm = int(env.get("AZURE_RATE_LIMIT_RPM", "300"))
        self.rate_limit_tpm = int(env.get("AZURE_RATE_LIMIT_TPM", "240000"))
        
        # Configure retry settings
        self.max_retries = int(env.get("AZURE_MAX_RETRIES", "5"))
        self.min_retry_seconds = float(env.get("AZURE_MIN_RETRY_SECONDS", "1.0"))
        self.max_retry_seconds = float(env.get("AZURE_MAX_RETRY_SECONDS", "60.0"))
        
        # Fallback models (for progressively moving to simpler models on failures)
        self.fallback_models = env.get("AZURE_FALLBACK_MODELS", f"{self.llm_model}").split(",")
        if self.llm_model not in self.fallback_models:
            self.fallback_models.insert(0, self.llm_model)
        
        logger.info(f"AzureOpenAIService initializing with:")
        logger.info(f"  - API Version: {self.api_version}")
        logger.info(f"  - Endpoint: {self.endpoint}")
        logger.info(f"  - Embedding Model: {self.embedding_model}")
        logger.info(f"  - Embedding Deployment: {self.embedding_deployment}")
        logger.info(f"  - LLM Model: {self.llm_model}")
        logger.info(f"  - LLM Deployment: {self.llm_deployment}")
        logger.info(f"  - Rate limits: {self.rate_limit_rpm} RPM, {self.rate_limit_tpm} TPM")
        logger.info(f"  - Retry settings: {self.max_retries} max retries, {self.min_retry_seconds}-{self.max_retry_seconds}s backoff")
        logger.info(f"  - Fallback models: {self.fallback_models}")
        
        # Create rate limiter
        self.rate_limiter = RateLimiter(
            max_requests_per_minute=self.rate_limit_rpm,
            max_tokens_per_minute=self.rate_limit_tpm
        )
        
        # Initialize the client and token refresh time
        self.client = None
        self.token_refresh_time = 0
        self._initialize_client()
        logger.info("AzureOpenAIService initialized successfully")
    
    def _initialize_client(self):
        """Initialize the Azure OpenAI client with token provider."""
        try:
            # Get token provider from auth_helper
            token_provider = get_azure_token_provider()
            logger.info("Got token provider from auth_helper")
            
            # Create client with token provider
            self.client = AzureOpenAI(
                azure_endpoint=self.endpoint,
                api_version=self.api_version,
                azure_ad_token_provider=token_provider
            )
            
            # Record token refresh time
            self.token_refresh_time = time.time()
            
            logger.info("Azure OpenAI client initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing Azure OpenAI client: {e}")
            raise
    
    async def refresh_tokens(self):
        """Refresh the token provider and client with exponential backoff."""
        try:
            logger.info("Refreshing token provider...")
            
            # Use exponential backoff for retries
            for attempt in range(3):  # Try up to 3 times
                try:
                    # Refresh token
                    refresh_token_if_needed()
                    token_provider = get_azure_token_provider(force_refresh=True)
                    
                    # Re-initialize client with new token provider
                    self.client = AzureOpenAI(
                        azure_endpoint=self.endpoint,
                        api_version=self.api_version,
                        azure_ad_token_provider=token_provider
                    )
                    
                    # Record refresh time
                    self.token_refresh_time = time.time()
                    
                    logger.info("Token provider and client refreshed successfully")
                    return True
                except Exception as e:
                    wait_time = (2 ** attempt) + (random.random() * 0.5)
                    logger.warning(f"Token refresh failed (attempt {attempt+1}/3): {e}. Retrying in {wait_time:.2f}s...")
                    await asyncio.sleep(wait_time)
            
            logger.error("Failed to refresh tokens after multiple attempts")
            return False
        except Exception as e:
            logger.error(f"Error refreshing tokens: {e}")
            return False
    
    async def _check_token_freshness(self):
        """Check if token needs refreshing (called before API operations)."""
        current_time = time.time()
        # Refresh token if it's older than 55 minutes
        if current_time - self.token_refresh_time > 55 * 60:
            logger.info("Token is older than 55 minutes, refreshing...")
            await self.refresh_tokens()
    
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts with production-grade reliability.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        try:
            logger.info(f"Generating embeddings for {len(texts)} texts")
            
            # Check token freshness
            await self._check_token_freshness()
            
            # Process in small batches with rate limiting
            batch_size = 5  # Process 5 texts at a time for reliability
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                batch_num = (i // batch_size) + 1
                total_batches = (len(texts) + batch_size - 1) // batch_size
                
                logger.info(f"Processing batch {batch_num}/{total_batches}")
                
                # Wait for rate limiter - estimate 1000 tokens per text
                estimated_tokens = sum(min(len(text.split()) * 2, 1000) for text in batch_texts)
                await self.rate_limiter.wait_if_needed(estimated_tokens)
                
                # Use tenacity for retries with backoff
                @retry(
                    stop=stop_after_attempt(self.max_retries),
                    wait=wait_random_exponential(
                        multiplier=self.min_retry_seconds,
                        max=self.max_retry_seconds
                    ),
                    retry=retry_if_exception_type((Exception)),
                    before_sleep=before_sleep_log(logger, logging.WARNING)
                )
                async def _generate_batch_embeddings():
                    try:
                        # Generate embeddings
                        response = self.client.embeddings.create(
                            input=batch_texts,
                            model=self.embedding_deployment
                        )
                        
                        batch_embeddings = [item.embedding for item in response.data]
                        
                        # Record token usage
                        if hasattr(response, 'usage') and hasattr(response.usage, 'total_tokens'):
                            self.rate_limiter.record_tokens(response.usage.total_tokens)
                        
                        return batch_embeddings
                    except Exception as e:
                        logger.warning(f"Error generating embeddings (will retry): {e}")
                        
                        # Check if token refresh might help
                        if "authentication" in str(e).lower() or "unauthorized" in str(e).lower():
                            await self.refresh_tokens()
                        
                        raise
                
                try:
                    batch_embeddings = await _generate_batch_embeddings()
                    all_embeddings.extend(batch_embeddings)
                    logger.info(f"Successfully processed batch {batch_num}/{total_batches}")
                except Exception as e:
                    logger.error(f"Failed to generate embeddings for batch {batch_num} after multiple retries: {e}")
                    # Add placeholder embeddings for failed batch
                    for _ in batch_texts:
                        all_embeddings.append([0.0] * 1536)  # Default embedding dimension
                
                # Add delay between batches to reduce pressure on API
                if i + batch_size < len(texts):
                    await asyncio.sleep(0.5)  # 500ms delay between batches
            
            logger.info(f"Successfully generated {len(all_embeddings)} embeddings")
            return all_embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            # Return empty embeddings as a fallback
            return [[0.0] * 1536 for _ in range(len(texts))]
    
    async def generate_single_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text with error handling.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        try:
            logger.info("Generating single embedding")
            
            # Use the batch function with a single text
            embeddings = await self.generate_embeddings([text])
            
            if embeddings and len(embeddings) > 0:
                return embeddings[0]
            else:
                logger.error("Failed to generate embedding")
                return [0.0] * 1536  # Default embedding dimension
                
        except Exception as e:
            logger.error(f"Error generating single embedding: {e}")
            return [0.0] * 1536  # Default embedding dimension
    
    async def generate_completion(self, 
                                messages: List[Dict[str, str]], 
                                temperature: float = 0.0,
                                max_tokens: int = 2000,
                                response_format: Optional[Dict] = None,
                                model: Optional[str] = None) -> Union[str, Dict]:
        """
        Generate a completion with robust error handling and fallback strategies.
        
        Args:
            messages: List of messages for chat completion
            temperature: Temperature for generation (0-1)
            max_tokens: Maximum tokens to generate
            response_format: Optional format specification (e.g., JSON mode)
            model: Optional model override
            
        Returns:
            Generated completion text or raw response if json_mode is True
        """
        try:
            # If model not specified, use default
            model_deployment = model or self.llm_deployment
            
            # Prepare fallback models list (exclude specified model from fallbacks)
            fallbacks = [m for m in self.fallback_models if m != model_deployment]
            if not model:  # Only include main model in fallbacks if it wasn't explicitly specified
                fallbacks = [model_deployment] + fallbacks
            
            # Check token freshness
            await self._check_token_freshness()
            
            # Prepare request parameters
            params = {
                "model": model_deployment,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            # Add response format if specified (JSON mode or other structured formats)
            if response_format:
                params["response_format"] = response_format
            
            # Estimate token count for rate limiting
            # Rough estimate: 4 tokens per word for all messages + max_tokens for response
            estimated_tokens = sum(len(m.get("content", "").split()) * 4 for m in messages) + max_tokens
            
            # Wait for rate limiting if needed
            await self.rate_limiter.wait_if_needed(estimated_tokens)
            
            # Try each model in fallback order
            for current_model in fallbacks:
                params["model"] = current_model
                
                # Check if model is a deployment or needs to be mapped to a deployment
                if current_model != self.llm_deployment and current_model != model_deployment:
                    logger.info(f"Using fallback model {current_model}")
                
                logger.info(f"Generating completion with model {current_model}")
                
                # Try with exponential backoff
                for attempt in range(self.max_retries):
                    try:
                        # Generate completion
                        response = self.client.chat.completions.create(**params)
                        
                        # Record token usage for rate limiting
                        if hasattr(response, 'usage') and hasattr(response.usage, 'total_tokens'):
                            self.rate_limiter.record_tokens(response.usage.total_tokens)
                        
                        # Extract content - handle structured response if needed
                        if response_format and response_format.get("type") == "json_object":
                            # Parse JSON and validate it
                            try:
                                content = response.choices[0].message.content
                                # Validate that it's valid JSON before returning
                                json_obj = json.loads(content)
                                return content  # Return raw JSON string
                            except json.JSONDecodeError:
                                # If JSON is invalid but we're in JSON mode, this is an error
                                logger.error("Invalid JSON returned from OpenAI API in JSON mode")
                                if attempt == self.max_retries - 1 and current_model == fallbacks[-1]:
                                    # Last attempt with last model
                                    from app.utils.json_repair import repair_json
                                    # Try to repair the JSON
                                    try:
                                        repaired = repair_json(content)
                                        logger.info("Successfully repaired JSON")
                                        return repaired
                                    except:
                                        logger.error("Failed to repair JSON")
                                        return content  # Return as-is as last resort
                                # Otherwise, retry or try next model
                                raise ValueError("Invalid JSON returned")
                        else:
                            # Normal text mode
                            content = response.choices[0].message.content
                            logger.info("Completion generated successfully")
                            return content
                    
                    except (ValueError, json.JSONDecodeError) as json_err:
                        # JSON parsing errors should trigger a retry or model fallback
                        logger.warning(f"JSON error (attempt {attempt+1}/{self.max_retries}): {json_err}")
                        
                        # If this is the last attempt for this model, try the next model
                        if attempt == self.max_retries - 1:
                            logger.warning(f"Moving to next fallback model after JSON error")
                            break
                        
                        # Exponential backoff for retry
                        delay = min(self.max_retry_seconds, 
                                   self.min_retry_seconds * (2 ** attempt) + random.random())
                        logger.info(f"Retrying in {delay:.2f}s...")
                        await asyncio.sleep(delay)
                    
                    except Exception as e:
                        logger.warning(f"Error generating completion (attempt {attempt+1}/{self.max_retries}): {e}")
                        
                        # Check if token refresh might help
                        if "authentication" in str(e).lower() or "unauthorized" in str(e).lower():
                            await self.refresh_tokens()
                        
                        # Last attempt for this model?
                        if attempt == self.max_retries - 1:
                            logger.warning(f"Moving to next fallback model")
                            break
                        
                        # Exponential backoff for retry
                        delay = min(self.max_retry_seconds, 
                                   self.min_retry_seconds * (2 ** attempt) + random.random())
                        logger.info(f"Retrying in {delay:.2f}s...")
                        await asyncio.sleep(delay)
            
            # If we get here, all models failed
            logger.error("All models failed after retries")
            return "Error generating completion after multiple retries. Please try again."
        
        except Exception as e:
            logger.error(f"Unhandled error in generate_completion: {e}")
            return f"An unexpected error occurred: {str(e)}"