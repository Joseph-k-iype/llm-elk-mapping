"""
Service for interacting with Azure OpenAI models.
"""

import os
import logging
import openai
from typing import List, Dict, Any, Optional, Union
from langchain_openai import AzureOpenAIEmbeddings
from langchain_openai import AzureChatOpenAI
from app.config.settings import get_settings
from app.core.auth_helper import get_azure_token_cached, refresh_token_if_needed

logger = logging.getLogger(__name__)

class AzureOpenAIService:
    """Service for interacting with Azure OpenAI models."""
    
    def __init__(self):
        """Initialize the Azure OpenAI service."""
        settings = get_settings()
        self.tenant_id = settings.azure.tenant_id
        self.client_id = settings.azure.client_id
        self.client_secret = settings.azure.client_secret
        self.endpoint = settings.azure.openai_endpoint
        self.api_key = settings.azure.openai_api_key
        self.embedding_model = settings.azure.embedding_model
        self.embedding_deployment = settings.azure.deployment_name
        self.llm_model = settings.azure.llm_model
        self.llm_deployment = settings.azure.llm_deployment
        
        self._setup_client()
        self._setup_langchain()
        
        logger.info(f"AzureOpenAIService initialized with endpoint: {self.endpoint}")
    
    def _setup_client(self):
        """Set up the OpenAI client for Azure."""
        # Initialize the Azure OpenAI client
        self.client = openai.AzureOpenAI(
            api_key=self.api_key,
            api_version="2023-05-15",  # Update with the latest version
            azure_endpoint=self.endpoint
        )
        
        logger.info(f"Azure OpenAI client configured with endpoint: {self.endpoint}")
    
    def _setup_langchain(self):
        """Set up LangChain components for Azure OpenAI."""
        try:
            # Create LangChain embeddings
            self.embeddings = AzureOpenAIEmbeddings(
                azure_endpoint=self.endpoint,
                api_key=self.api_key,
                api_version="2023-05-15",
                deployment=self.embedding_deployment,
                model=self.embedding_model,
                dimensions=3072,  # Maximum dimensions for text-embedding-3-large
                chunk_size=1
            )
            
            # Create LangChain chat model
            self.chat_model = AzureChatOpenAI(
                azure_endpoint=self.endpoint,
                api_key=self.api_key,
                api_version="2023-05-15",
                deployment=self.llm_deployment,
                model=self.llm_model,
                temperature=0.0,
                max_tokens=2000
            )
            
            logger.info(f"LangChain components initialized with embedding model: {self.embedding_model} and LLM model: {self.llm_model}")
        except Exception as e:
            logger.error(f"Error setting up LangChain components: {e}")
            raise
    
    def refresh_tokens(self):
        """Refresh the Azure tokens."""
        token = get_azure_token_cached(
            tenant_id=self.tenant_id,
            client_id=self.client_id,
            client_secret=self.client_secret,
            scope="https://cognitiveservices.azure.com/.default"
        )
        if token:
            # Update the client
            self.client = openai.AzureOpenAI(
                api_key=token,
                api_version="2023-05-15",
                azure_endpoint=self.endpoint
            )
            
            # Update LangChain components
            self._setup_langchain()
            
            logger.info("Azure OpenAI token refreshed")
            return True
        else:
            logger.error("Failed to refresh Azure OpenAI token")
            return False
    
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        try:
            embeddings = await self.embeddings.aembed_documents(texts)
            logger.debug(f"Generated embeddings for {len(texts)} texts")
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    async def generate_single_embedding(self, text: str) -> List[float]:
        """
        Generate embeddings for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        try:
            embedding = await self.embeddings.aembed_query(text)
            return embedding
        except Exception as e:
            logger.error(f"Error generating single embedding: {e}")
            raise
    
    async def generate_completion(self, 
                                messages: List[Dict[str, str]], 
                                temperature: float = 0.0,
                                max_tokens: int = 2000) -> str:
        """
        Generate a completion using the chat model.
        
        Args:
            messages: List of messages (system, user, assistant)
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text
        """
        try:
            response = await self.client.chat.completions.create(
                model=self.llm_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating completion: {e}")
            # Check if token refresh is needed
            if "401" in str(e) or "unauthorized" in str(e).lower():
                logger.info("Attempting to refresh token and retry request")
                if self.refresh_tokens():
                    # Retry the request
                    response = await self.client.chat.completions.create(
                        model=self.llm_model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                    return response.choices[0].message.content
            raise