"""
Azure AD token management that properly integrates with your environment.py.
"""

import logging
import time
import threading
import os
from typing import Optional, Dict, Any
from azure.identity import DefaultAzureCredential, ClientSecretCredential, get_bearer_token_provider
from app.core.environment import get_os_env

logger = logging.getLogger(__name__)

# Global token provider that will be shared across the application
_token_provider = None
_credential = None
_last_refresh_time = 0
_provider_lock = threading.RLock()

def get_azure_credential(force_refresh: bool = False):
    """
    Get Azure credential from environment.py.
    
    Args:
        force_refresh: Force refresh the credential
        
    Returns:
        Azure credential
    """
    global _credential, _last_refresh_time
    
    # Use a lock to ensure thread safety
    with _provider_lock:
        current_time = time.time()
        
        # Check if we need to refresh (every 55 minutes or forced)
        if _credential is None or force_refresh or (current_time - _last_refresh_time > 55 * 60):
            try:
                # Get environment instance
                env = get_os_env()
                logger.info("Getting Azure credential from environment")
                
                # Get credential using environment's method
                if hasattr(env, '_get_credential') and callable(env._get_credential):
                    _credential = env._get_credential()
                    logger.info("Successfully retrieved credential from environment")
                else:
                    # Fallback to directly creating credential
                    logger.info("Using fallback credential creation")
                    
                    # Check if we should use managed identity
                    use_managed_identity = env.get("USE_MANAGED_IDENTITY", "False").lower() in ('true', 't', 'yes', 'y', '1')
                    
                    if use_managed_identity:
                        logger.info("Using DefaultAzureCredential (managed identity)")
                        _credential = DefaultAzureCredential()
                    else:
                        # Get values from environment
                        tenant_id = env.get("AZURE_TENANT_ID")
                        client_id = env.get("AZURE_CLIENT_ID")
                        client_secret = env.get("AZURE_CLIENT_SECRET")
                        
                        # Create credential
                        logger.info("Using ClientSecretCredential")
                        _credential = ClientSecretCredential(
                            tenant_id=tenant_id,
                            client_id=client_id,
                            client_secret=client_secret
                        )
                
                _last_refresh_time = current_time
            except Exception as e:
                logger.error(f"Error getting credential: {e}")
                raise
    
    return _credential

def get_azure_token_provider(force_refresh: bool = False):
    """
    Get Azure token provider, using the credential from environment.py.
    
    Args:
        force_refresh: Force refresh the token provider
        
    Returns:
        Azure token provider
    """
    global _token_provider
    
    # Use a lock to ensure thread safety
    with _provider_lock:
        if _token_provider is None or force_refresh:
            # Get credential
            credential = get_azure_credential(force_refresh)
            
            # Create token provider
            logger.info("Creating token provider with credential")
            _token_provider = get_bearer_token_provider(
                credential,
                "https://cognitiveservices.azure.com/.default"
            )
            logger.info("Successfully created token provider")
    
    return _token_provider

def get_azure_token_cached(tenant_id: str = None, client_id: str = None, client_secret: str = None, 
                          scope: str = "https://cognitiveservices.azure.com/.default") -> Optional[str]:
    """
    Get an Azure AD token directly, preferring environment's token if available.
    
    Args:
        tenant_id: Azure tenant ID (not used, included for compatibility)
        client_id: Azure client ID (not used, included for compatibility)
        client_secret: Azure client secret (not used, included for compatibility)
        scope: OAuth scope
        
    Returns:
        Access token if successful, None otherwise
    """
    try:
        # Get environment instance
        env = get_os_env()
        
        # If environment has a token, use it
        if hasattr(env, 'token') and env.token:
            logger.info("Using token from environment")
            return env.token
        
        # Otherwise get a token using credential
        logger.info("Getting token using credential")
        credential = get_azure_credential()
        token = credential.get_token(scope)
        
        return token.token if token else None
    except Exception as e:
        logger.error(f"Error getting Azure token: {e}")
        return None

def refresh_token_if_needed(tenant_id: str = None, client_id: str = None, client_secret: str = None, 
                           scope: str = "https://cognitiveservices.azure.com/.default",
                           min_validity_seconds: int = 600) -> bool:
    """
    Refresh token if needed - this will ensure the token provider is refreshed.
    
    Args:
        tenant_id: Azure tenant ID (not used, included for compatibility)
        client_id: Azure client ID (not used, included for compatibility)
        client_secret: Azure client secret (not used, included for compatibility)
        scope: OAuth scope
        min_validity_seconds: Minimum validity in seconds
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Force refresh the token provider
        get_azure_token_provider(force_refresh=True)
        return True
    except Exception as e:
        logger.error(f"Error refreshing token: {e}")
        return False

# Global refresh thread reference
_token_refresh_thread = None

def start_token_refresh_service(refresh_interval: int = 1800) -> threading.Thread:
    """
    Start a background thread to refresh tokens periodically.
    
    Args:
        refresh_interval: Interval in seconds
        
    Returns:
        Thread object
    """
    global _token_refresh_thread
    
    # Check if thread is already running
    if _token_refresh_thread is not None and _token_refresh_thread.is_alive():
        logger.info("Token refresh thread is already running")
        return _token_refresh_thread
    
    def _token_refresh_worker():
        """Worker function for token refresh thread."""
        # Initial delay
        time.sleep(10)
        
        logger.info(f"Token refresh service started (interval: {refresh_interval}s)")
        
        while True:
            try:
                # Refresh token provider
                refresh_token_if_needed()
            except Exception as e:
                logger.error(f"Error in token refresh worker: {e}")
            
            # Sleep until next refresh
            time.sleep(refresh_interval)
    
    # Create and start thread
    _token_refresh_thread = threading.Thread(
        target=_token_refresh_worker,
        daemon=True,
        name="TokenRefreshThread"
    )
    _token_refresh_thread.start()
    
    return _token_refresh_thread

# Simplified auth functions for FastAPI dependency injection
async def verify_api_key(api_key: str = None):
    """No API key verification required."""
    return True

async def get_current_user(token: str = None):
    """Simplified user authentication."""
    return {"username": "default_user"}