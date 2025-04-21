import logging
import json
from typing import Any, Dict, List, Optional
import aiohttp

from app.config import settings # Assuming settings are here
from app.tools.base_provider import CodeSearchProvider

logger = logging.getLogger(__name__)

class GitHubSearchProvider(CodeSearchProvider):
    """GitHub code search provider with enhanced error handling and pagination."""
    def __init__(self, access_token: Optional[str] = None):
        # Explicitly use token from settings
        self.access_token = access_token or settings.github_token 
        self.headers = {
            "Authorization": f"token {self.access_token}",
            "Accept": "application/vnd.github.v3+json"
        } if self.access_token else {
            "Accept": "application/vnd.github.v3+json"
        }
        if not self.access_token:
            logger.warning("GitHub token not provided. GitHub search may be rate-limited.")
        
    async def search(self, query: str, language: Optional[str] = None, per_page: int = 30, **kwargs) -> List[Dict[str, Any]]:
        """Search GitHub for code, handling errors and pagination.

        Args:
            query: The search query.
            language: Optional language filter.
            per_page: Results per page (max 100, default 30).
            **kwargs: Absorbs potential extra arguments from base class call.

        Returns:
            A list of search results (items), or an empty list on error.
        """
        search_query = f"{query} language:{language}" if language else query
        params = {"q": search_query, "per_page": min(per_page, 100)} # Enforce max 100
        
        logger.debug(f"GitHub Search - Query: '{search_query}', Params: {params}")
        
        # Using a single session for potential retries (though not implemented here)
        async with aiohttp.ClientSession(headers=self.headers) as session:
            try:
                async with session.get("https://api.github.com/search/code", params=params) as response:
                    # Check status code
                    if response.status == 200:
                        try:
                            result = await response.json()
                            items = result.get("items", [])
                            logger.info(f"GitHub search successful. Found {len(items)} items for query: '{query}'")
                            # TODO: Implement further pagination if result['total_count'] > len(items)
                            return items
                        except json.JSONDecodeError as json_err:
                            logger.error(f"GitHub search failed: Invalid JSON response for query '{query}'. Error: {json_err}")
                            return []
                        except Exception as parse_err: # Catch other potential issues processing response
                             logger.error(f"GitHub search failed: Error processing response for query '{query}'. Error: {parse_err}")
                             return []
                    # Handle specific error codes
                    elif response.status == 403:
                        rate_limit_info = response.headers.get('X-RateLimit-Remaining', 'N/A')
                        reset_time = response.headers.get('X-RateLimit-Reset', 'N/A')
                        logger.warning(f"GitHub search failed (403 Forbidden): Rate limit likely exceeded or invalid token. Remaining: {rate_limit_info}, Reset: {reset_time}")
                        # Optionally parse reset time and wait/retry
                        return []
                    elif response.status == 422:
                        error_details = await response.text() # Get error details if possible
                        logger.error(f"GitHub search failed (422 Unprocessable Entity): Invalid query '{search_query}'. Details: {error_details[:200]}...")
                        return []
                    else:
                        # General error for other statuses
                        error_text = await response.text()
                        logger.error(f"GitHub search failed with status {response.status} for query '{query}'. Response: {error_text[:200]}...")
                        return []
            except aiohttp.ClientError as client_err:
                logger.error(f"GitHub search failed: Network error for query '{query}'. Error: {client_err}")
                return []
            except Exception as e:
                # Catch-all for unexpected errors
                logger.error(f"GitHub search failed: Unexpected error for query '{query}'. Error: {e}", exc_info=True)
                return [] 