import logging
from typing import Any, Dict, List, Optional
import aiohttp
import json

from app.config import settings # Assuming settings are here
from app.tools.base_provider import CodeSearchProvider

logger = logging.getLogger(__name__)

class StackOverflowSearchProvider(CodeSearchProvider):
    """Stack Overflow search provider"""
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key # Assumes key is passed, CodeAgent __init__ does this
        if not self.api_key:
            logger.warning("Stack Overflow API key not provided. Search may be rate-limited or fail.")
        
    async def search(self, query: str, language: Optional[str] = None, **kwargs) -> List[Dict[str, Any]]:
        """Search Stack Overflow using the advanced search endpoint.
        
        TODO: Implement robust error handling and pagination.
        """
        params = {
            "site": "stackoverflow",
            "key": self.api_key,
            "tagged": language if language else None,
            "q": query,
            "pagesize": 50 # Request a moderate number of results
        }
        # Remove None parameters
        filtered_params = {k: v for k, v in params.items() if v is not None}
        
        logger.debug(f"Stack Overflow Search - Params: {filtered_params}")

        # Basic implementation without error handling or pagination (as before)
        # Needs significant enhancement for TOOL-003
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    "https://api.stackexchange.com/2.3/search/advanced",
                    params=filtered_params
                ) as response:
                    if response.status == 200:
                        try:
                            result = await response.json()
                            # TODO: Check 'quota_remaining', 'has_more' for pagination/rate limits
                            # TODO: Handle API errors within the JSON response (e.g., error_id)
                            return result.get("items", [])
                        except json.JSONDecodeError as json_err:
                             logger.error(f"Stack Overflow search failed: Invalid JSON response. Error: {json_err}")
                             return []
                    else:
                        error_text = await response.text()
                        logger.error(f"Stack Overflow search failed with status {response.status}. Response: {error_text[:200]}...")
                        return []
        except aiohttp.ClientError as client_err:
            logger.error(f"Stack Overflow search failed: Network error. Error: {client_err}")
            return []
        except Exception as e:
            logger.error(f"Stack Overflow search failed: Unexpected error. Error: {e}", exc_info=True)
            return [] 