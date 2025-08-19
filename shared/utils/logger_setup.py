"""
Logging setup for the Credit Intelligence Platform.
Provides structured logging configuration for all components.
"""

import logging
import logging.handlers
import sys
import os
from datetime import datetime
from typing import Optional

def setup_logging(level: str = "INFO", log_file: Optional[str] = None, 
                 log_dir: str = "logs", max_bytes: int = 10*1024*1024, 
                 backup_count: int = 5):
    """
    Setup logging configuration for the application.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file name
        log_dir: Directory for log files
        max_bytes: Maximum size of log file before rotation
        backup_count: Number of backup files to keep
    """
    
    # Create logs directory if it doesn't exist
    if log_file and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    
    # Configure logging level
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_path = os.path.join(log_dir, log_file)
        file_handler = logging.handlers.RotatingFileHandler(
            log_path, 
            maxBytes=max_bytes, 
            backupCount=backup_count
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Set specific logger levels for external libraries
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('asyncpg').setLevel(logging.WARNING)
    logging.getLogger('aiohttp').setLevel(logging.WARNING)
    
    logging.info(f"Logging configured - Level: {level}, File: {log_file or 'Console only'}")

def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for a specific module."""
    return logging.getLogger(name)

class StructuredLogger:
    """Structured logger for consistent logging across the platform."""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
    
    def info(self, message: str, **kwargs):
        """Log info message with structured data."""
        extra_info = " | ".join([f"{k}={v}" for k, v in kwargs.items()])
        full_message = f"{message} | {extra_info}" if extra_info else message
        self.logger.info(full_message)
    
    def error(self, message: str, error: Exception = None, **kwargs):
        """Log error message with structured data."""
        extra_info = " | ".join([f"{k}={v}" for k, v in kwargs.items()])
        full_message = f"{message} | {extra_info}" if extra_info else message
        
        if error:
            full_message += f" | error={str(error)}"
        
        self.logger.error(full_message)
    
    def warning(self, message: str, **kwargs):
        """Log warning message with structured data."""
        extra_info = " | ".join([f"{k}={v}" for k, v in kwargs.items()])
        full_message = f"{message} | {extra_info}" if extra_info else message
        self.logger.warning(full_message)
    
    def debug(self, message: str, **kwargs):
        """Log debug message with structured data."""
        extra_info = " | ".join([f"{k}={v}" for k, v in kwargs.items()])
        full_message = f"{message} | {extra_info}" if extra_info else message
        self.logger.debug(full_message)
