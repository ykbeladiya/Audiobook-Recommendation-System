"""
Logging configuration for the FastAPI application.
"""
import logging
import sys
import time
from pathlib import Path
from logging.handlers import RotatingFileHandler
from typing import Any, Callable
from fastapi import FastAPI, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from functools import partial

# Create logs directory if it doesn't exist
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

# Configure logging
logger = logging.getLogger("audiobook_api")
logger.setLevel(logging.INFO)

# Console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
console_handler.setFormatter(console_formatter)

# File handler (rotating)
file_handler = RotatingFileHandler(
    log_dir / "api.log",
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5,
    encoding='utf-8'
)
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
file_handler.setFormatter(file_formatter)

# Add handlers
logger.addHandler(console_handler)
logger.addHandler(file_handler)

class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging requests, responses, and timing."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process the request/response and log details."""
        # Start timer
        start_time = time.time()
        
        # Get request details
        path = request.url.path
        method = request.method
        client_host = request.client.host if request.client else "unknown"
        
        # Log request
        logger.info(
            f"Request - Method: {method} Path: {path} "
            f"Client: {client_host}"
        )
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Log response
            logger.info(
                f"Response - Method: {method} Path: {path} "
                f"Status: {response.status_code} Duration: {duration:.3f}s"
            )
            
            return response
            
        except Exception as e:
            # Log error
            logger.error(
                f"Error - Method: {method} Path: {path} "
                f"Error: {str(e)}",
                exc_info=True
            )
            raise

def setup_logging(app: FastAPI) -> None:
    """Set up logging for the FastAPI application."""
    # Add logging middleware
    app.add_middleware(LoggingMiddleware)
    
    # Log application startup
    @app.on_event("startup")
    async def startup_event():
        logger.info("Application starting up")
        
    # Log application shutdown
    @app.on_event("shutdown")
    async def shutdown_event():
        logger.info("Application shutting down") 