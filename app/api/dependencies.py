import logging
from fastapi import Depends
from functools import lru_cache

from app.services.code_agent import CodeAgent

logger = logging.getLogger(__name__)

# Singleton pattern for services
_code_agent = None

@lru_cache()
def get_agent() -> CodeAgent:
    """
    Get or create a singleton instance of CodeAgent
    """
    global _code_agent
    if not _code_agent:
        logger.info("Initializing CodeAgent singleton...")
        try:
            _code_agent = CodeAgent()
            logger.info("CodeAgent singleton initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing CodeAgent: {str(e)}", exc_info=True)
            raise
    return _code_agent 