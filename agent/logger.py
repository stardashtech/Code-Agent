import logging
import requests
from config import settings # Import settings object

def get_logger(name):
    logger = logging.getLogger(name)
    # Basic setup if handlers are not already configured (e.g., by FastAPI/Uvicorn)
    if not logger.hasHandlers() or len(logger.handlers) == 0:
        logger.setLevel(logging.DEBUG) # Set level (consider making this configurable via settings)
        ch = logging.StreamHandler()
        formatter = logging.Formatter('[%(asctime)s] %(levelname)s:%(name)s: %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        logger.propagate = False # Avoid duplicate logs if root logger is configured
    return logger

def meilisearch_log(message: str, level: str = "INFO"):
    """
    Simple MeiliSearch log submission. Uses configuration from global settings.
    Error handling is basic; suppresses errors to avoid impacting main flow.
    """
    if not settings.MEILISEARCH_HOST or not settings.MEILISEARCH_INDEX:
        # Silently ignore if MeiliSearch is not configured
        return
        
    url = f"{settings.MEILISEARCH_HOST}/indexes/{settings.MEILISEARCH_INDEX}/documents"
    payload = {"level": level, "message": message}
    headers = {"Content-Type": "application/json"}
    # Add API key header only if it's set
    if settings.MEILISEARCH_API_KEY:
        headers["Authorization"] = f"Bearer {settings.MEILISEARCH_API_KEY}" # Standard Bearer token
        # headers["X-Meili-API-Key"] = settings.MEILISEARCH_API_KEY # Old header, use Authorization
        
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=2)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
    except requests.exceptions.RequestException as e:
        # Get the root logger to avoid circular dependency if this logger fails
        logging.getLogger().warning(f"Failed to send log to MeiliSearch: {e}", exc_info=False)
    except Exception as e:
        logging.getLogger().warning(f"Unexpected error sending log to MeiliSearch: {e}", exc_info=False) 