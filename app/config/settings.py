"""
Application settings configuration with default values for required fields.
"""

import os
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from pydantic_settings import BaseSettings

class AzureSettings(BaseModel):
    # Use default values for required fields
    tenant_id: str = Field(os.getenv("AZURE_TENANT_ID", "default-tenant-id"), env="AZURE_TENANT_ID")
    client_id: str = Field(os.getenv("AZURE_CLIENT_ID", "default-client-id"), env="AZURE_CLIENT_ID")
    client_secret: str = Field(os.getenv("AZURE_CLIENT_SECRET", "default-client-secret"), env="AZURE_CLIENT_SECRET")
    openai_endpoint: str = Field(os.getenv("AZURE_OPENAI_ENDPOINT", "https://default.openai.azure.com/"), env="AZURE_OPENAI_ENDPOINT")
    embedding_model: str = Field(os.getenv("AZURE_EMBEDDING_MODEL", "text-embedding-3-large"), env="AZURE_EMBEDDING_MODEL")
    deployment_name: str = Field(os.getenv("AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-3-large"), env="AZURE_EMBEDDING_DEPLOYMENT") 
    llm_model: str = Field(os.getenv("AZURE_LLM_MODEL", "gpt-4o-mini"), env="AZURE_LLM_MODEL")
    llm_deployment: str = Field(os.getenv("AZURE_LLM_DEPLOYMENT", "gpt-4o-mini"), env="AZURE_LLM_DEPLOYMENT")

class ElasticsearchSettings(BaseModel):
    hosts: List[str] = Field(eval(os.getenv("ELASTICSEARCH_HOSTS", '["http://localhost:9200"]')), env="ELASTICSEARCH_HOSTS")
    index_name: str = Field(os.getenv("ELASTICSEARCH_INDEX_NAME", "business_terms"), env="ELASTICSEARCH_INDEX_NAME")
    username: Optional[str] = Field(os.getenv("ELASTICSEARCH_USERNAME", None), env="ELASTICSEARCH_USERNAME")
    password: Optional[str] = Field(os.getenv("ELASTICSEARCH_PASSWORD", None), env="ELASTICSEARCH_PASSWORD")

class SecuritySettings(BaseModel):
    secret_key: str = Field(os.getenv("SECRET_KEY", "your-secret-key-here"), env="SECRET_KEY")
    algorithm: str = Field(os.getenv("JWT_ALGORITHM", "HS256"), env="JWT_ALGORITHM")
    access_token_expire_minutes: int = Field(int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30")), env="ACCESS_TOKEN_EXPIRE_MINUTES")

class LoggingSettings(BaseModel):
    log_level: str = Field(os.getenv("LOG_LEVEL", "INFO"), env="LOG_LEVEL")
    log_format: str = Field(os.getenv("LOG_FORMAT", "detailed"), env="LOG_FORMAT")
    log_to_file: bool = Field(os.getenv("LOG_TO_FILE", "True").lower() in ('true', 't', 'yes', 'y', '1'), env="LOG_TO_FILE")
    log_file: str = Field(os.getenv("LOG_FILE", "logs/app.log"), env="LOG_FILE")

class Settings(BaseSettings):
    app_name: str = Field(os.getenv("APP_NAME", "Data Governance Mapping Service"), env="APP_NAME")
    version: str = Field(os.getenv("APP_VERSION", "0.1.0"), env="APP_VERSION")
    debug: bool = Field(os.getenv("DEBUG", "False").lower() in ('true', 't', 'yes', 'y', '1'), env="DEBUG")
    environment: str = Field(os.getenv("ENVIRONMENT", "development"), env="ENVIRONMENT")
    allowed_hosts: List[str] = Field(eval(os.getenv("ALLOWED_HOSTS", '["*"]')), env="ALLOWED_HOSTS")
    azure: AzureSettings = AzureSettings()
    elasticsearch: ElasticsearchSettings = ElasticsearchSettings()
    security: SecuritySettings = SecuritySettings()
    logging: LoggingSettings = LoggingSettings()

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

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
        _settings = Settings()
    return _settings