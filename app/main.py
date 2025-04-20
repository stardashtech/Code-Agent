import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Callable

import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator
from starlette.middleware.base import BaseHTTPMiddleware

from app.api.api import api_router
from app.schemas import HealthResponse

# Setup logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
    ]
)

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

logger = logging.getLogger(__name__)

# Create Prometheus instrumentator
instrumentator = Instrumentator()

# Middleware for request ID
class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = str(uuid.uuid4())
        # Add request ID to request state
        request.state.request_id = request_id
        
        # Process the request
        response = await call_next(request)
        
        # Add request ID to response headers
        response.headers["X-Request-ID"] = request_id
        return response

# Middleware for request logging
class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Get request details
        start_time = time.time()
        request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
        
        logger.info(
            f"Request started: {request.method} {request.url.path} "
            f"(ID: {request_id})"
        )
        
        # Process the request
        try:
            response = await call_next(request)
            process_time = time.time() - start_time
            
            # Log request completion
            logger.info(
                f"Request completed: {request.method} {request.url.path} "
                f"- Status: {response.status_code} - Duration: {process_time:.4f}s "
                f"(ID: {request_id})"
            )
            
            return response
        except Exception as e:
            # Log exceptions
            process_time = time.time() - start_time
            logger.error(
                f"Request failed: {request.method} {request.url.path} "
                f"- Error: {str(e)} - Duration: {process_time:.4f}s "
                f"(ID: {request_id})",
                exc_info=True
            )
            raise

# Startup and shutdown event handlers
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Create required directories
    Path("logs").mkdir(exist_ok=True)
    Path("data").mkdir(exist_ok=True)
    Path("temp").mkdir(exist_ok=True)
    
    logger.info("API Server starting up")
    
    yield
    
    # Shutdown: Clean up resources
    logger.info("API Server shutting down")

# Create FastAPI app
app = FastAPI(
    title="Code Agent API",
    description="API for code analysis, search, and automated fixes",
    version="1.0.0",
    lifespan=lifespan,
    docs_url=None,  # Disable default docs
    redoc_url=None,  # Disable default redoc
)

# Add middlewares
app.add_middleware(RequestIDMiddleware)
app.add_middleware(LoggingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins in development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Setup metrics
instrumentator.instrument(app).expose(app, endpoint="/metrics")

# Include API router
app.include_router(api_router, prefix="/api")

# Custom exception handler
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"},
    )

# Health check endpoint
@app.get("/healthz", response_model=HealthResponse, tags=["health"])
async def health_check():
    return {
        "status": "ok",
        "version": "1.0.0",
        "components": {
            "api": "healthy",
            "database": "connected",
        }
    }

# Custom Swagger UI
@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=f"{app.title} - API Documentation",
        swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@4/swagger-ui-bundle.js",
        swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@4/swagger-ui.css",
    )

# Main entrypoint for development
if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    ) 