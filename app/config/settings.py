"""
Application settings configuration.
"""

import os
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from pydantic_settings import BaseSettings

class AzureSettings(BaseModel):
    tenant_id: str = Field(..., env="AZURE_TENANT_ID")
    client_id: str = Field(..., env="AZURE_CLIENT_ID")
    client_secret: str = Field(..., env="AZURE_CLIENT_SECRET")
    openai_endpoint: str = Field(..., env="AZURE_OPENAI_ENDPOINT")
    embedding_model: str = Field("text-embedding-3-large", env="AZURE_EMBEDDING_MODEL")
    deployment_name: str = Field("text-embedding-3-large", env="AZURE_EMBEDDING_DEPLOYMENT") 
    llm_model: str = Field("gpt-4o-mini", env="AZURE_LLM_MODEL")
    llm_deployment: str = Field("gpt-4o-mini", env="AZURE_LLM_DEPLOYMENT")

class ElasticsearchSettings(BaseModel):
    hosts: List[str] = Field(["http://localhost:9200"], env="ELASTICSEARCH_HOSTS")
    index_name: str = Field("business_terms", env="ELASTICSEARCH_INDEX_NAME")
    username: Optional[str] = Field(None, env="ELASTICSEARCH_USERNAME")
    password: Optional[str] = Field(None, env="ELASTICSEARCH_PASSWORD")

class SecuritySettings(BaseModel):
    secret_key: str = Field("your-secret-key-here", env="SECRET_KEY")
    algorithm: str = Field("HS256", env="JWT_ALGORITHM")
    access_token_expire_minutes: int = Field(30, env="ACCESS_TOKEN_EXPIRE_MINUTES")

class LoggingSettings(BaseModel):
    log_level: str = Field("INFO", env="LOG_LEVEL")
    log_format: str = Field("detailed", env="LOG_FORMAT")
    log_to_file: bool = Field(True, env="LOG_TO_FILE")
    log_file: str = Field("logs/app.log", env="LOG_FILE")

class Settings(BaseSettings):
    app_name: str = Field("Data Governance Mapping Service", env="APP_NAME")
    version: str = Field("0.1.0", env="APP_VERSION")
    debug: bool = Field(False, env="DEBUG")
    environment: str = Field("development", env="ENVIRONMENT")
    allowed_hosts: List[str] = Field(["*"], env="ALLOWED_HOSTS")
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