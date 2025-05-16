"""
Azure OpenAI service that integrates with your environment.py.
"""

import os
import logging
import time
from typing import List, Dict, Any, Optional
from openai import AzureOpenAI
from app.core.auth_helper import get_azure_token_provider
from app.core.environment import get_os_env

logger = logging.getLogger(__name__)

class AzureOpenAIService:
    """Service for interacting with Azure OpenAI models using environment integration."""
    
    def __init__(self):
        """Initialize the Azure OpenAI service."""
        # Get environment
        env = get_os_env()
        
        # Get configuration from environment
        self.endpoint = env.get("AZURE_OPENAI_ENDPOINT", "")
        self.api_version = env.get("AZURE_API_VERSION", "2023-05-15")
        self.embedding_model = env.get("AZURE_EMBEDDING_MODEL", "text-embedding-3-large")
        self.embedding_deployment = env.get("AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-3-large")
        self.llm_model = env.get("AZURE_LLM_MODEL", "gpt-4o-mini")
        self.llm_deployment = env.get("AZURE_LLM_DEPLOYMENT", "gpt-4o-mini")
        
        logger.info(f"AzureOpenAIService initializing with:")
        logger.info(f"  - API Version: {self.api_version}")
        logger.info(f"  - Endpoint: {self.endpoint}")
        logger.info(f"  - Embedding Model: {self.embedding_model}")
        logger.info(f"  - Embedding Deployment: {self.embedding_deployment}")
        logger.info(f"  - LLM Model: {self.llm_model}")
        logger.info(f"  - LLM Deployment: {self.llm_deployment}")
        
        # Initialize the client
        self._initialize_client()
        logger.info("AzureOpenAIService initialized successfully")
    
    def _initialize_client(self):
        """Initialize the Azure OpenAI client."""
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
            
            logger.info("Azure OpenAI client initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing Azure OpenAI client: {e}")
            raise
    
    def refresh_tokens(self):
        """Refresh the token provider and client."""
        try:
            logger.info("Refreshing token provider...")
            token_provider = get_azure_token_provider(force_refresh=True)
            
            # Re-initialize client with new token provider
            self.client = AzureOpenAI(
                azure_endpoint=self.endpoint,
                api_version=self.api_version,
                azure_ad_token_provider=token_provider
            )
            
            logger.info("Token provider and client refreshed successfully")
            return True
        except Exception as e:
            logger.error(f"Error refreshing tokens: {e}")
            return False
    
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        try:
            logger.info(f"Generating embeddings for {len(texts)} texts")
            
            # Process in small batches to avoid rate limiting
            batch_size = 3  # Very small batch size for reliability
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                batch_num = (i // batch_size) + 1
                total_batches = (len(texts) + batch_size - 1) // batch_size
                
                logger.info(f"Processing batch {batch_num}/{total_batches}")
                
                max_retries = 3
                for retry in range(max_retries):
                    try:
                        # Generate embeddings
                        response = self.client.embeddings.create(
                            input=batch_texts,
                            model=self.embedding_deployment
                        )
                        
                        batch_embeddings = [item.embedding for item in response.data]
                        all_embeddings.extend(batch_embeddings)
                        
                        logger.info(f"Successfully processed batch {batch_num}/{total_batches}")
                        break  # Success, exit retry loop
                        
                    except Exception as batch_error:
                        logger.error(f"Error processing batch {batch_num} (attempt {retry+1}/{max_retries}): {batch_error}")
                        
                        if retry == max_retries - 1:  # Last retry
                            logger.error(f"All retries failed for batch {batch_num}")
                            # Add empty vectors for failed embeddings
                            for _ in batch_texts:
                                all_embeddings.append([0.0] * 1536)  # Default embedding dimension
                        else:
                            # Refresh token and try again
                            self.refresh_tokens()
                            # Wait before retrying
                            delay = 2.0 * (retry + 1)  # Increasing delay
                            logger.info(f"Waiting {delay} seconds before retry...")
                            time.sleep(delay)
                
                # Add a significant delay between batches to avoid rate limits
                if i + batch_size < len(texts):
                    delay = 2.0  # 2 seconds between batches
                    logger.info(f"Waiting {delay} seconds before next batch...")
                    time.sleep(delay)
            
            logger.info(f"Successfully generated {len(all_embeddings)} embeddings")
            return all_embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    async def generate_single_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
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
                                max_tokens: int = 2000) -> str:
        """
        Generate a completion.
        
        Args:
            messages: List of messages
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text
        """
        try:
            logger.info("Generating completion")
            
            max_retries = 2
            for retry in range(max_retries):
                try:
                    # Generate completion
                    response = self.client.chat.completions.create(
                        model=self.llm_deployment,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                    
                    content = response.choices[0].message.content
                    logger.info("Completion generated successfully")
                    
                    return content
                
                except Exception as completion_error:
                    logger.error(f"Error generating completion (attempt {retry+1}/{max_retries}): {completion_error}")
                    
                    if retry == max_retries - 1:  # Last retry
                        logger.error("All retries failed for completion")
                        return "Error generating completion. Please try again."
                    else:
                        # Refresh token and try again
                        self.refresh_tokens()
                        # Wait before retrying
                        delay = 2.0 * (retry + 1)  # Increasing delay
                        logger.info(f"Waiting {delay} seconds before retry...")
                        time.sleep(delay)
            
        except Exception as e:
            logger.error(f"Error generating completion: {e}")
            return "Error generating completion. Please try again."