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

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    "https://api.stackexchange.com/2.3/search/advanced",
                    params=filtered_params
                ) as response:
                    response_body_text = await response.text() # Read body once
                    logger.debug(f"Stack Overflow API Response Status: {response.status}")

                    if response.status == 200:
                        try:
                            result = json.loads(response_body_text)
                            
                            # Check for API errors within the JSON response
                            if 'error_id' in result:
                                error_name = result.get('error_name', 'UnknownError')
                                error_message = result.get('error_message', 'No details provided.')
                                logger.error(
                                    f"Stack Overflow API returned an error: "
                                    f"ID={result['error_id']}, Name={error_name}, Message='{error_message}'"
                                )
                                return [] # Return empty on API error

                            items = result.get("items", [])
                            has_more = result.get("has_more", False)
                            quota_remaining = result.get("quota_remaining", "N/A")
                            quota_max = result.get("quota_max", "N/A")
                            
                            logger.info(
                                f"Stack Overflow search successful. Found {len(items)} items. "
                                f"Quota: {quota_remaining}/{quota_max}. Has More: {has_more}"
                            )

                            if has_more:
                                logger.info("More Stack Overflow results available. Pagination not implemented.")
                                
                            return items
                            
                        except json.JSONDecodeError as json_err:
                             logger.error(f"Stack Overflow search failed: Invalid JSON response. Status: {response.status}. Error: {json_err}. Response Body: {response_body_text[:500]}...")
                             return []
                        except Exception as parse_err:
                             logger.error(f"Stack Overflow search failed: Error processing successful response. Status: {response.status}. Error: {parse_err}")
                             return []
                    else:
                        logger.error(f"Stack Overflow search failed with status {response.status}. Response: {response_body_text[:500]}...")
                        return []
        except aiohttp.ClientError as client_err:
            logger.error(f"Stack Overflow search failed: Network error. Error: {client_err}")
            return []
        except Exception as e:
            logger.error(f"Stack Overflow search failed: Unexpected error. Error: {e}", exc_info=True)
            return [] 