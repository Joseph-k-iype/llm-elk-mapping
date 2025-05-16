"""
Application settings configuration with direct file reading for credentials.
"""

import os
import sys
from pydantic import BaseModel, Field, field_validator
from typing import Optional, Dict, Any, List
from pydantic_settings import BaseSettings
from dotenv import dotenv_values
import logging

logger = logging.getLogger(__name__)

# Read environment variables directly from files
config_path = os.path.join("env", "config.env")
credentials_path = os.path.join("env", "credentials.env")

# Load values from files
config_values = {}
credentials_values = {}

# Load config.env
try:
    if os.path.isfile(config_path):
        logger.info(f"Loading configuration from {config_path}")
        config_values = dotenv_values(config_path)
        logger.info(f"Loaded {len(config_values)} values from {config_path}")
    else:
        logger.warning(f"Config file not found: {config_path}")
except Exception as e:
    logger.error(f"Error loading config file: {e}")

# Load credentials.env
try:
    if os.path.isfile(credentials_path):
        logger.info(f"Loading credentials from {credentials_path}")
        credentials_values = dotenv_values(credentials_path)
        logger.info(f"Loaded {len(credentials_values)} values from {credentials_path}")
    else:
        logger.warning(f"Credentials file not found: {credentials_path}")
except Exception as e:
    logger.error(f"Error loading credentials file: {e}")

# Combine both sets of values, with credentials taking precedence
all_values = {**config_values, **credentials_values}

# Set environment variables from the loaded values
for key, value in all_values.items():
    os.environ[key] = value

class AzureSettings(BaseModel):
    """Azure settings with improved validation and error messages."""
    tenant_id: str = Field(all_values.get("AZURE_TENANT_ID", ""), env="AZURE_TENANT_ID")
    client_id: str = Field(all_values.get("AZURE_CLIENT_ID", ""), env="AZURE_CLIENT_ID")
    client_secret: str = Field(all_values.get("AZURE_CLIENT_SECRET", ""), env="AZURE_CLIENT_SECRET")
    openai_endpoint: str = Field(all_values.get("AZURE_OPENAI_ENDPOINT", ""), env="AZURE_OPENAI_ENDPOINT")
    embedding_model: str = Field(all_values.get("AZURE_EMBEDDING_MODEL", "text-embedding-3-large"), env="AZURE_EMBEDDING_MODEL")
    deployment_name: str = Field(all_values.get("AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-3-large"), env="AZURE_EMBEDDING_DEPLOYMENT") 
    llm_model: str = Field(all_values.get("AZURE_LLM_MODEL", "gpt-4o-mini"), env="AZURE_LLM_MODEL")
    llm_deployment: str = Field(all_values.get("AZURE_LLM_DEPLOYMENT", "gpt-4o-mini"), env="AZURE_LLM_DEPLOYMENT")
    
    @field_validator('tenant_id', 'client_id', 'client_secret', 'openai_endpoint')
    def check_required_azure_settings(cls, v, info):
        """Validate that required Azure settings are provided."""
        if not v:
            raise ValueError(f"Required Azure setting {info.field_name} is missing. Check your credentials.env file.")
        
        # Check for default values that should be replaced
        if info.field_name in ['tenant_id', 'client_id', 'client_secret'] and v in ['default-tenant-id', 'default-client-id', 'default-client-secret']:
            raise ValueError(f"Azure {info.field_name} is using a default value. Please set actual values in credentials.env.")
            
        return v

class ElasticsearchSettings(BaseModel):
    """Elasticsearch settings with proper validation."""
    hosts: List[str] = Field(eval(all_values.get("ELASTICSEARCH_HOSTS", '["http://localhost:9200"]')), env="ELASTICSEARCH_HOSTS")
    index_name: str = Field(all_values.get("ELASTICSEARCH_INDEX_NAME", "business_terms"), env="ELASTICSEARCH_INDEX_NAME")
    username: Optional[str] = Field(all_values.get("ELASTICSEARCH_USERNAME", None), env="ELASTICSEARCH_USERNAME")
    password: Optional[str] = Field(all_values.get("ELASTICSEARCH_PASSWORD", None), env="ELASTICSEARCH_PASSWORD")
    
    @field_validator('hosts')
    def check_elasticsearch_hosts(cls, v):
        """Validate Elasticsearch hosts."""
        if not v:
            raise ValueError("Elasticsearch hosts cannot be empty")
        
        # Convert string to list if needed
        if isinstance(v, str):
            try:
                v = eval(v)
            except:
                v = [v]
        
        return v

class SecuritySettings(BaseModel):
    secret_key: str = Field(all_values.get("SECRET_KEY", "your-secret-key-here"), env="SECRET_KEY")
    algorithm: str = Field(all_values.get("JWT_ALGORITHM", "HS256"), env="JWT_ALGORITHM")
    access_token_expire_minutes: int = Field(int(all_values.get("ACCESS_TOKEN_EXPIRE_MINUTES", "30")), env="ACCESS_TOKEN_EXPIRE_MINUTES")

class LoggingSettings(BaseModel):
    log_level: str = Field(all_values.get("LOG_LEVEL", "INFO"), env="LOG_LEVEL")
    log_format: str = Field(all_values.get("LOG_FORMAT", "detailed"), env="LOG_FORMAT")
    log_to_file: bool = Field(all_values.get("LOG_TO_FILE", "True").lower() in ('true', 't', 'yes', 'y', '1'), env="LOG_TO_FILE")
    log_file: str = Field(all_values.get("LOG_FILE", "logs/app.log"), env="LOG_FILE")

class Settings(BaseSettings):
    app_name: str = Field(all_values.get("APP_NAME", "Data Governance Mapping Service"), env="APP_NAME")
    version: str = Field(all_values.get("APP_VERSION", "0.1.0"), env="APP_VERSION")
    debug: bool = Field(all_values.get("DEBUG", "False").lower() in ('true', 't', 'yes', 'y', '1'), env="DEBUG")
    environment: str = Field(all_values.get("ENVIRONMENT", "development"), env="ENVIRONMENT")
    allowed_hosts: List[str] = Field(eval(all_values.get("ALLOWED_HOSTS", '["*"]')), env="ALLOWED_HOSTS")
    azure: AzureSettings = AzureSettings()
    elasticsearch: ElasticsearchSettings = ElasticsearchSettings()
    security: SecuritySettings = SecuritySettings()
    logging: LoggingSettings = LoggingSettings()

    class Config:
        env_file_encoding = "utf-8"
        extra = "ignore"

_settings = None

def get_settings() -> Settings:
    """
    Get application settings.
    Returns a singleton instance of Settings.
    
    Returns:
        Settings: Application settings
    """
    global _settings
    if _settings is None:
        # Initialize settings
        _settings = Settings()
        
        # Debug the Azure settings
        logger.info(f"App name: {_settings.app_name} v{_settings.version} ({_settings.environment})")
        logger.info(f"Debug mode: {_settings.debug}")
        logger.info(f"Azure tenant_id: {_settings.azure.tenant_id[:4]}***" if _settings.azure.tenant_id else "Azure tenant_id: None")
        logger.info(f"Azure client_id: {_settings.azure.client_id[:4]}***" if _settings.azure.client_id else "Azure client_id: None")
        logger.info(f"Azure client_secret: {_settings.azure.client_secret[:4]}***" if _settings.azure.client_secret else "Azure client_secret: None")
        logger.info(f"Azure OpenAI endpoint: {_settings.azure.openai_endpoint}")
        logger.info(f"Elasticsearch hosts: {_settings.elasticsearch.hosts}")
    return _settings