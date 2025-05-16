"""
Environment Configuration - Utilities for managing environment variables and configuration.
Loads environment variables from both config.env and credentials.env.
"""

import os
import sys
import logging
import chardet
from typing import Optional, Any, Dict, List, Union
from pathlib import Path
from dotenv import dotenv_values
from azure.identity import DefaultAzureCredential, get_bearer_token_provider, ClientSecretCredential

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Default environment file locations
ENV_DIR = "env"
CONFIG_PATH = f"{ENV_DIR}/config.env"
CREDS_PATH = f"{ENV_DIR}/credentials.env"
CERT_PATH = f"{ENV_DIR}/cacert.pem"

def is_file_readable(filepath: str) -> bool:
    """
    Check if a file is readable.
    
    Args:
        filepath: Path to the file
        
    Returns:
        bool: True if readable, raises FileNotFoundError otherwise
    
    Raises:
        FileNotFoundError: If file doesn't exist or isn't readable
    """
    if not os.path.isfile(filepath) or not os.access(filepath, os.R_OK):
        raise FileNotFoundError(f"The file '{filepath}' does not exist or is not readable")
    return True

def str_to_bool(s: str) -> bool:
    """
    Convert a string to a boolean value.
    
    Args:
        s: String to convert
        
    Returns:
        bool: Converted boolean value
        
    Raises:
        ValueError: If string can't be converted to boolean
    """
    if s.lower() in ('true', 't', 'yes', 'y', '1'):
        return True
    elif s.lower() in ('false', 'f', 'no', 'n', '0'):
        return False
    else:
        raise ValueError(f"Invalid boolean value: {s}")

class OSEnv:
    """
    Environment variable manager for application configuration.
    
    This class provides methods for loading, accessing, and managing environment
    variables from multiple sources, including .env files and the OS environment.
    """
    
    def __init__(self, config_file: str, creds_file: str, certificate_path: str, proxy_enabled: Optional[bool] = None):
        """
        Initialize environment configuration.
        
        Args:
            config_file: Path to configuration .env file
            creds_file: Path to credentials .env file
            certificate_path: Path to SSL certificate
            proxy_enabled: Override the PROXY_ENABLED setting
        """
        self.var_list = []
        
        # Check if files exist first to provide better error messages
        config_exists = os.path.isfile(config_file) and os.access(config_file, os.R_OK)
        creds_exists = os.path.isfile(creds_file) and os.access(creds_file, os.R_OK)
        
        if not config_exists:
            logger.warning(f"Warning: Configuration file '{config_file}' not found or not readable")
        
        if not creds_exists:
            logger.warning(f"Warning: Credentials file '{creds_file}' not found or not readable")
            
        # Load configuration (even if files don't exist, we'll just log warnings)
        self.bulk_set(config_file, True)
        
        # Explicitly log that we're loading the credentials file
        logger.info(f"Loading credentials from {creds_file}")
        self.bulk_set(creds_file, False)
        
        # Check for Azure credentials after loading both files
        self._check_azure_credentials()
        
        self.set_certificate_path(certificate_path)
        
        # Handle proxy_enabled override
        if proxy_enabled is not None:
            self.set("PROXY_ENABLED", str(proxy_enabled))
            logger.info(f"Proxy enabled setting overridden to: {proxy_enabled}")
        
        # Set proxy if enabled
        proxy_enabled_env = str_to_bool(self.get("PROXY_ENABLED", "False"))
        if proxy_enabled_env:
            self.set_proxy()
            logger.info("Proxy settings applied")
        else:
            logger.info("Running without proxy")
        
        # Handle secured endpoints
        if str_to_bool(self.get("SECURED_ENDPOINTS", "False")):
            self.token = self.get_azure_token()
        else:
            self.token = None
        
        self.credential = self._get_credential()
        
        # Set PostgreSQL environment variables if not already set
        self._set_postgres_defaults()
    
    def _check_azure_credentials(self):
        """
        Check if required Azure credentials are set and log warnings if not.
        """
        required_vars = [
            "AZURE_TENANT_ID", 
            "AZURE_CLIENT_ID", 
            "AZURE_CLIENT_SECRET",
            "AZURE_OPENAI_ENDPOINT"
        ]
        
        for var in required_vars:
            value = self.get(var)
            if not value:
                logger.warning(f"Required Azure variable {var} is not set in environment or config files")
            elif var == "AZURE_TENANT_ID" and value == "default-tenant-id":
                logger.warning(f"{var} is set to default value. Check if credentials.env was properly loaded.")
            elif var == "AZURE_CLIENT_ID" and value == "default-client-id":
                logger.warning(f"{var} is set to default value. Check if credentials.env was properly loaded.")
            elif var == "AZURE_CLIENT_SECRET" and value == "default-client-secret":
                logger.warning(f"{var} is set to default value. Check if credentials.env was properly loaded.")
        
    def _set_postgres_defaults(self):
        """Set default PostgreSQL environment variables if not already set."""
        # Default PostgreSQL configuration
        postgres_defaults = {
            "PG_HOST": self.get("PG_HOST", "localhost"),
            "PG_PORT": self.get("PG_PORT", "5432"),
            "PG_USER": self.get("PG_USER", "postgres"),
            "PG_PASSWORD": self.get("PG_PASSWORD", "postgres"),
            "PG_DB": self.get("PG_DB", "metadata_db")
        }
        
        # Set defaults if not already set
        for key, value in postgres_defaults.items():
            if not self.get(key):
                self.set(key, value)
                
        # Log PostgreSQL configuration (without password)
        logger.info(f"PostgreSQL configuration: host={self.get('PG_HOST')}, port={self.get('PG_PORT')}, db={self.get('PG_DB')}, user={self.get('PG_USER')}")
    
    def _get_credential(self):
        """
        Get the appropriate Azure credential.
        
        Returns:
            Azure credential instance
        """
        if str_to_bool(self.get("USE_MANAGED_IDENTITY", "False")):
            logger.info("Using DefaultAzureCredential (managed identity)")
            return DefaultAzureCredential()
        else:
            # Get Azure credentials from environment
            tenant_id = self.get("AZURE_TENANT_ID")
            client_id = self.get("AZURE_CLIENT_ID")
            client_secret = self.get("AZURE_CLIENT_SECRET")
            
            # Log credentials (masking secrets)
            if tenant_id and client_id and client_secret:
                masked_tenant = f"{tenant_id[:4]}...{tenant_id[-4:]}" if len(tenant_id) > 8 else "***"
                masked_client = f"{client_id[:4]}...{client_id[-4:]}" if len(client_id) > 8 else "***"
                logger.info(f"Using ClientSecretCredential with tenant_id: {masked_tenant}, client_id: {masked_client}")
                
                return ClientSecretCredential(
                    tenant_id=tenant_id, 
                    client_id=client_id, 
                    client_secret=client_secret
                )
            else:
                logger.warning("Missing Azure AD credentials, cannot create ClientSecretCredential")
                logger.warning("Make sure AZURE_TENANT_ID, AZURE_CLIENT_ID, and AZURE_CLIENT_SECRET are set in credentials.env")
                return None
    
    def set_certificate_path(self, path: str):
        """
        Set the certificate path for SSL verification.
        
        Args:
            path: Path to SSL certificate
        """
        try:
            if not os.path.isabs(path):
                path = os.path.abspath(path)
            
            if os.path.exists(path):
                if is_file_readable(path):
                    self.set("REQUESTS_CA_BUNDLE", path)
                    self.set("SSL_CERT_FILE", path)
                    self.set("CURL_CA_BUNDLE", path)
                    logger.info(f"Certificate path set to {path}")
                else:
                    logger.warning(f"Certificate file {path} exists but is not readable")
            else:
                logger.warning(f"Certificate file {path} not found, using system certificates")
        except Exception as e:
            logger.error(f"Error setting certificate path: {e}")
            raise
    
    def bulk_set(self, dotenvfile: str, print_val: bool = False) -> None:
        """
        Load environment variables from a dotenv file.
        
        Args:
            dotenvfile: Path to .env file
            print_val: Whether to print values to logs
        """
        try:
            if not os.path.isabs(dotenvfile):
                dotenvfile = os.path.abspath(dotenvfile)
            
            if os.path.exists(dotenvfile) and is_file_readable(dotenvfile):
                temp_dict = dotenv_values(dotenvfile)
                
                # Log how many variables were loaded
                logger.info(f"Loading {len(temp_dict)} environment variables from {dotenvfile}")
                
                for key, value in temp_dict.items():
                    self.set(key, value, print_val)
                del temp_dict
                logger.info(f"Loaded environment from {dotenvfile}")
            else:
                logger.warning(f"Environment file {dotenvfile} not found or not readable, skipping")
        except Exception as e:
            logger.error(f"Error loading environment variables from {dotenvfile}: {e}")
            raise
    
    def set(self, key: str, value: str, print_val: bool = False) -> None:
        """
        Set an environment variable.
        
        Args:
            key: Environment variable name
            value: Environment variable value
            print_val: Whether to print value to logs
        """
        try:
            os.environ[key] = value
            if key not in self.var_list:
                self.var_list.append(key)
            if print_val:
                if key in {'AZURE_CLIENT_SECRET', 'AD_USER_PW', 'PG_PASSWORD'}:
                    logger.info(f"{key}: [REDACTED]")
                else:
                    logger.info(f"{key}: {value}")
        except Exception as e:
            logger.error(f"Error setting environment variable {key}: {e}")
            raise
    
    def get(self, key: str, default: Optional[str] = None) -> str:
        """
        Get an environment variable.
        
        Args:
            key: Environment variable name
            default: Default value if not found
            
        Returns:
            str: Environment variable value or default
        """
        try:
            return os.environ.get(key, default)
        except Exception as e:
            logger.error(f"Error getting environment variable {key}: {e}")
            raise
    
    def set_proxy(self) -> None:
        """
        Set up proxy configuration for HTTP requests.
        """
        try:
            ad_username = self.get("AD_USERNAME")
            ad_password = self.get("AD_USER_PW")
            proxy_domain = self.get("HTTPS_PROXY_DOMAIN")
            
            if not all([ad_username, ad_password, proxy_domain]):
                logger.error("Proxy settings are incomplete, cannot configure proxy")
                raise ValueError("Proxy settings are incomplete. Check AD_USERNAME, AD_USER_PW, and HTTPS_PROXY_DOMAIN")
            
            # Use http:// instead of https:// as the proxy only supports HTTP
            proxy_url = f"http://{ad_username}:{ad_password}@{proxy_domain}"
            self.set("HTTP_PROXY", proxy_url, print_val=False)
            self.set("HTTPS_PROXY", proxy_url, print_val=False)
            
            # Set no_proxy domains
            no_proxy_domains = [
                'cognitiveservices.azure.com',
                'search.windows.net',
                'openai.azure.com',
                'core.windows.net',
                'azurewebsites.net'
            ]
            
            # Add any custom NO_PROXY entries from environment
            custom_no_proxy = self.get("CUSTOM_NO_PROXY", "")
            if custom_no_proxy:
                no_proxy_domains.extend(custom_no_proxy.split(","))
            
            # Add PostgreSQL host to NO_PROXY if it's not localhost
            pg_host = self.get("PG_HOST")
            if pg_host and pg_host not in ["localhost", "127.0.0.1"]:
                no_proxy_domains.append(pg_host)
            
            self.set("NO_PROXY", ",".join(no_proxy_domains), print_val=False)
            logger.info("Proxy settings configured with HTTP protocol")
            logger.info(f"NO_PROXY domains: {', '.join(no_proxy_domains)}")
        except Exception as e:
            logger.error(f"Error setting proxy: {e}")
            raise
    
    def get_azure_token(self) -> str:
        """
        Get an Azure AD token for authenticated API calls.
        
        Returns:
            str: Azure AD token
        """
        try:
            # Get Azure credentials from environment
            tenant_id = self.get("AZURE_TENANT_ID")
            client_id = self.get("AZURE_CLIENT_ID")
            client_secret = self.get("AZURE_CLIENT_SECRET")
            
            # Log which credentials are being used (masked for security)
            if tenant_id and client_id and client_secret:
                masked_tenant = f"{tenant_id[:4]}...{tenant_id[-4:]}" if len(tenant_id) > 8 else "***"
                masked_client = f"{client_id[:4]}...{client_id[-4:]}" if len(client_id) > 8 else "***"
                logger.info(f"Acquiring Azure token with tenant_id: {masked_tenant}, client_id: {masked_client}")
            
                # Create credential
                credential = ClientSecretCredential(
                    tenant_id=tenant_id,
                    client_id=client_id,
                    client_secret=client_secret
                )
                
                # Get token
                token = credential.get_token("https://cognitiveservices.azure.com/.default")
                
                if token and token.token:
                    self.set("AZURE_TOKEN", token.token, print_val=False)
                    logger.info("Azure token acquired successfully")
                    return token.token
                else:
                    logger.error("Failed to get Azure token (token is None)")
            else:
                logger.error("Missing Azure credentials, cannot get token")
                if not tenant_id:
                    logger.error("AZURE_TENANT_ID is not set")
                if not client_id:
                    logger.error("AZURE_CLIENT_ID is not set")
                if not client_secret:
                    logger.error("AZURE_CLIENT_SECRET is not set")
            
            return None
        except Exception as e:
            logger.error(f"Error retrieving Azure token: {e}")
            return None
    
    def list_env_vars(self) -> None:
        """
        List all environment variables (with sensitive values redacted).
        """
        logger.info("Environment variables currently set:")
        for var in sorted(self.var_list):
            if var in {'AZURE_TOKEN', 'AD_USER_PW', 'AZURE_CLIENT_SECRET', 'PG_PASSWORD'}:
                logger.info(f"{var}: [REDACTED]")
            else:
                value = os.getenv(var)
                # Truncate very long values
                if value and len(value) > 100:
                    value = value[:97] + "..."
                logger.info(f"{var}: {value}")

# Singleton instance for application-wide access
_os_env_instance = None

def get_os_env(
    config_file: str = CONFIG_PATH, 
    creds_file: str = CREDS_PATH, 
    certificate_path: str = CERT_PATH,
    proxy_enabled: Optional[bool] = None
) -> OSEnv:
    """
    Get the OSEnv instance.
    
    Args:
        config_file: Path to configuration file
        creds_file: Path to credentials file
        certificate_path: Path to SSL certificate
        proxy_enabled: Override the PROXY_ENABLED setting in the config file
        
    Returns:
        OSEnv: Environment variable manager instance
    """
    global _os_env_instance
    if _os_env_instance is None:
        logger.info(f"Initializing environment from {config_file} and {creds_file}")
        _os_env_instance = OSEnv(config_file, creds_file, certificate_path, proxy_enabled)
    elif proxy_enabled is not None and proxy_enabled != str_to_bool(_os_env_instance.get("PROXY_ENABLED", "False")):
        # If proxy_enabled changes, recreate the instance
        logger.info("Re-initializing environment with new proxy settings")
        _os_env_instance = OSEnv(config_file, creds_file, certificate_path, proxy_enabled)
    
    return _os_env_instance