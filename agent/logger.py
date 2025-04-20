import logging
import requests
from config import MEILISEARCH_HOST, MEILISEARCH_API_KEY, MEILISEARCH_INDEX

def get_logger(name):
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        # Handler for console output
        ch = logging.StreamHandler()
        formatter = logging.Formatter('[%(asctime)s] %(levelname)s:%(name)s: %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger

def meilisearch_log(message: str, level: str = "INFO"):
    """
    Simple MeiliSearch log submission. Additional error handling should be added in a real project.
    """
    url = f"{MEILISEARCH_HOST}/indexes/{MEILISEARCH_INDEX}/documents"
    payload = {"level": level, "message": message}
    headers = {"X-Meili-API-Key": MEILISEARCH_API_KEY, "Content-Type": "application/json"}
    try:
        requests.post(url, json=payload, headers=headers, timeout=2)
    except Exception:
        # Suppress logging errors so the main flow isn't affected
        pass 