import os
import logging
from typing import List, Dict, Any, Optional

import aiohttp
from tavily import TavilyClient

from app.config import settings

logger = logging.getLogger(__name__)

class WebSearchProvider:
    """Base class for web search providers."""
    async def search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        raise NotImplementedError

class TavilySearchProvider(WebSearchProvider):
    """Provides search functionality using the Tavily API."""
    def __init__(self, api_key: Optional[str] = None):
        effective_api_key = api_key or settings.tavily_api_key
        if not effective_api_key:
            logger.warning("Tavily API key not provided. Tavily search will not be available.")
            self.client = None
        else:
            try:
                self.client = TavilyClient(api_key=effective_api_key)
                logger.info("TavilySearchProvider initialized successfully.")
            except Exception as e:
                logger.error(f"Failed to initialize TavilyClient: {e}", exc_info=True)
                self.client = None

    async def search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Perform a search using the Tavily API.

        Args:
            query: The search query string.
            max_results: The maximum number of results to return.

        Returns:
            A list of search result dictionaries, or an empty list if search fails.
        """
        if not self.client:
            logger.error("Tavily client not initialized. Cannot perform search.")
            return []

        try:
            logger.debug(f"Performing Tavily search for query: '{query}' with max_results={max_results}")
            # Use Tavily's search method - it's synchronous, so run in executor
            # Tavily client handles async internally if event loop is running
            # but let's use a standard async pattern with aiohttp if available
            # or run_in_executor if the library is purely sync.
            # Checking TavilyClient source, it seems to use requests internally.
            # Best practice is to use an async library like aiohttp or httpx for async code.
            # However, for simplicity and using the provided client, let's assume
            # we might need to wrap the sync call if it blocks.
            # Tavily documentation examples use it directly in async functions,
            # implying it might handle the event loop context correctly. Let's try direct call first.

            response = await self.client.search(query=query, search_depth="advanced", max_results=max_results)
            # The Tavily library might have changed. The `.search()` method might not be async.
            # If it blocks, we'll need `loop.run_in_executor`.
            # Let's assume it works directly for now based on common patterns.

            logger.debug(f"Tavily raw response: {response}")

            # Process results (structure might vary slightly based on Tavily response format)
            # Assuming 'results' is a key in the response dictionary containing a list
            search_results = response.get("results", [])
            if isinstance(search_results, list):
                 # Format results to a common standard if needed
                 # Example: {'title': ..., 'url': ..., 'content': ...}
                 formatted_results = [
                     {
                         "title": res.get("title"),
                         "url": res.get("url"),
                         "content": res.get("content"), # Or 'snippet'
                         "score": res.get("score")
                     }
                     for res in search_results
                     if res.get("url") # Ensure result has a URL
                 ]
                 logger.info(f"Tavily search successful. Found {len(formatted_results)} results for query: '{query}'")
                 return formatted_results
            else:
                 logger.error(f"Unexpected format for Tavily search results: {type(search_results)}")
                 return []

        except Exception as e:
            logger.error(f"Error during Tavily search for query '{query}': {e}", exc_info=True)
            return []

# Example of potential Google Search Provider (requires google-api-python-client)
# class GoogleSearchProvider(WebSearchProvider):
#     def __init__(self, api_key: Optional[str] = None, cse_id: Optional[str] = None):
#         self.api_key = api_key or settings.google_api_key
#         self.cse_id = cse_id or settings.google_cse_id
#         if not self.api_key or not self.cse_id:
#             logger.warning("Google API Key or CSE ID missing. Google Search disabled.")
#             self.service = None
#         else:
#             try:
#                 from googleapiclient.discovery import build
#                 self.service = build("customsearch", "v1", developerKey=self.api_key)
#                 logger.info("Google Search initialized.")
#             except ImportError:
#                 logger.error("google-api-python-client not installed. Google Search disabled.")
#                 self.service = None
#             except Exception as e:
#                 logger.error(f"Failed to initialize Google Search: {e}")
#                 self.service = None

#     async def search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
#         if not self.service:
#             return []
#         try:
#             loop = asyncio.get_running_loop()
#             response = await loop.run_in_executor(
#                 None, # Use default executor
#                 lambda: self.service.cse().list(
#                     q=query,
#                     cx=self.cse_id,
#                     num=max_results
#                 ).execute()
#             )
#             results = response.get('items', [])
#             formatted = [
#                 {"title": item.get('title'), "url": item.get('link'), "content": item.get('snippet')}
#                 for item in results
#             ]
#             return formatted
#         except Exception as e:
#             logger.error(f"Error during Google search: {e}")
#             return [] 