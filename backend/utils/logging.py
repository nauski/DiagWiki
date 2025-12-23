import os
import logging
from pathlib import Path
from logging.handlers import RotatingFileHandler


def setup_logging():
    """Configure logging for the application"""
    # Create logs directory if it doesn't exist
    # base dir is project root
    base_dir = Path(__file__).resolve().parent.parent
    log_dir = base_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Get log level from environment variable, default to INFO
    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    log_level_value = getattr(logging, log_level)

    # Create formatters
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Create and configure console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level_value)
    console_handler.setFormatter(formatter)

    # Create and configure file handler with rotation
    file_handler = RotatingFileHandler(
        log_dir / "app.log",
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(log_level_value)
    file_handler.setFormatter(formatter)

    # Get root logger and configure it
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level_value)
    
    # Remove any existing handlers to avoid duplicates
    root_logger.handlers.clear()
    
    # Add our handlers
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    # Configure uvicorn loggers to use our handlers
    uvicorn_error_logger = logging.getLogger("uvicorn.error")
    uvicorn_error_logger.handlers.clear()
    uvicorn_error_logger.addHandler(console_handler)
    uvicorn_error_logger.addHandler(file_handler)
    uvicorn_error_logger.propagate = False

    # Set specific log levels for noisy libraries
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
