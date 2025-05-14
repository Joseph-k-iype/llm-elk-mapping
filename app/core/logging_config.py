"""
Advanced logging configuration for the AI Tagging Service API.
Supports different output formats, log rotation, and log levels.
"""

import os
import sys
import logging
import logging.handlers
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path

# Default logging level
DEFAULT_LOG_LEVEL = "INFO"

# Available logging levels
LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL
}

# Log format strings
LOG_FORMATS = {
    "simple": "%(asctime)s [%(levelname)s] %(message)s",
    "detailed": "%(asctime)s [%(levelname)s] [%(name)s:%(lineno)d] %(message)s",
    "json": '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "line": %(lineno)d, "message": "%(message)s"}',
}

# Default logging configuration
DEFAULT_CONFIG = {
    "log_level": DEFAULT_LOG_LEVEL,
    "log_format": "detailed",
    "log_to_file": True,
    "log_file": "logs/app.log",
    "log_file_max_size": 10 * 1024 * 1024,  # 10 MB
    "log_file_backup_count": 5,
    "log_to_console": True,
    "log_to_json": False
}

def configure_logging(config: Optional[Dict[str, Any]] = None) -> None:
    """
    Configure the logging system with the provided configuration.
    
    Args:
        config: Logging configuration dictionary (optional)
    """
    if config is None:
        config = DEFAULT_CONFIG.copy()
    else:
        # Merge with defaults
        merged_config = DEFAULT_CONFIG.copy()
        merged_config.update(config)
        config = merged_config
    
    # Get log level from environment or config
    log_level_name = os.environ.get("LOG_LEVEL", config["log_level"]).upper()
    log_level = LOG_LEVELS.get(log_level_name, logging.INFO)
    
    # Get log format from config
    log_format_name = config["log_format"]
    log_format = LOG_FORMATS.get(log_format_name, LOG_FORMATS["detailed"])
    
    # Create formatter
    formatter = logging.Formatter(log_format)
    
    # Create root logger and set level
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add console handler if enabled
    if config["log_to_console"]:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # Add file handler if enabled
    if config["log_to_file"]:
        # Ensure log directory exists
        log_file = config["log_file"]
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        
        # Create rotating file handler
        file_handler = logging.handlers.RotatingFileHandler(
            filename=log_file,
            maxBytes=config["log_file_max_size"],
            backupCount=config["log_file_backup_count"]
        )
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Set specific levels for noisy libraries
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    
    # Log configuration
    logging.info(f"Logging configured with level={log_level_name}, format={log_format_name}")
    
    # For JSON logging, use a different approach - create a second handler with JSON formatter
    if config["log_to_json"]:
        json_formatter = logging.Formatter(LOG_FORMATS["json"])
        json_log_file = config["log_file"].replace(".log", "_json.log")
        
        json_handler = logging.handlers.RotatingFileHandler(
            filename=json_log_file,
            maxBytes=config["log_file_max_size"],
            backupCount=config["log_file_backup_count"]
        )
        json_handler.setFormatter(json_formatter)
        root_logger.addHandler(json_handler)
        logging.info(f"JSON logging enabled to {json_log_file}")

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)

def log_request(method: str, url: str, status_code: int, duration_ms: float) -> None:
    """
    Log an HTTP request.
    
    Args:
        method: HTTP method
        url: Request URL
        status_code: Response status code
        duration_ms: Request duration in milliseconds
    """
    logger = logging.getLogger("api.request")
    level = logging.INFO if status_code < 400 else logging.ERROR
    logger.log(level, f"{method} {url} {status_code} ({duration_ms:.2f}ms)")

def log_exception(exc: Exception, context: Optional[Dict[str, Any]] = None) -> None:
    """
    Log an exception with additional context.
    
    Args:
        exc: Exception to log
        context: Additional context information
    """
    logger = logging.getLogger("api.exception")
    logger.exception(f"Exception: {exc}", extra={"context": context or {}})

# Helper function to get request ID from context
def get_request_id() -> str:
    """
    Get the request ID from the current context.
    
    Returns:
        str: Request ID
    """
    import contextvars
    request_id_var = contextvars.ContextVar("request_id", default="")
    return request_id_var.get()

# Initialize logging with default configuration
configure_logging()