import os
import logging
from typing import List, Dict, Any, Optional

import aiohttp
from tavily import TavilyClient
import httpx
import asyncio

from app.config import settings

logger = logging.getLogger(__name__)

# --- Yeniden Deneme ve Rate Limit Yapılandırması ---
TAVILY_MAX_RETRIES = 3
TAVILY_INITIAL_DELAY = 1.5 # saniye
TAVILY_BACKOFF_FACTOR = 2

class WebSearchProvider:
    """Base class for web search providers."""
    async def search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        raise NotImplementedError

class TavilySearchProvider(WebSearchProvider):
    """Provides search functionality using the Tavily API."""
    def __init__(self, api_key: Optional[str] = None):
        """
        Initializes the Tavily Search Provider.

        Args:
            api_key: Tavily API key. If None, attempts to read from settings.
        """
        self.api_key = api_key or settings.tavily_api_key
        if not self.api_key:
            logger.error("Tavily API key is not configured. Web search will fail.")
            # Raise error or disable? Disabling is safer for agent startup.
            self.client = None
            # raise ValueError("Tavily API key is missing.")
        else:
            try:
                # Initialize TavilyClient asynchronously if possible, otherwise sync
                # As of tavily-python 0.3.3, client seems synchronous internally
                # but we use async structure for consistency.
                self.client = TavilyClient(api_key=self.api_key)
                # Test connection immediately? TavilyClient doesn't have a direct ping.
                # We'll rely on the first search call to validate.
                logger.info("TavilySearchProvider initialized successfully.")
            except Exception as e:
                 logger.error(f"Failed to initialize TavilyClient: {e}", exc_info=True)
                 self.client = None

    async def search(self, query: str, search_depth: str = "basic", max_results: int = 5, 
                     include_domains: Optional[List[str]] = None, 
                     exclude_domains: Optional[List[str]] = None) -> List[Dict]:
        """
        Performs a web search using the Tavily API.

        Args:
            query: The search query.
            search_depth: "basic" or "advanced". Advanced is more comprehensive but uses more credits.
            max_results: Maximum number of results to return.
            include_domains: Optional list of domains to include.
            exclude_domains: Optional list of domains to exclude.

        Returns:
            A list of search result dictionaries or an empty list on error.
        """
        if not self.client:
            logger.error("Tavily client not initialized. Cannot perform search.")
            return [{"error": "Tavily client not configured."}]
            
        if not query:
            logger.warning("Tavily search called with empty query.")
            return []

        current_delay = TAVILY_INITIAL_DELAY
        for attempt in range(TAVILY_MAX_RETRIES + 1):
            try:
                logger.info(f"Performing Tavily search (Attempt {attempt + 1}): '{query}'")
                # TavilyClient.search seems synchronous, run in executor for async context
                # This might change in future versions of tavily-python
                loop = asyncio.get_running_loop()
                response = await loop.run_in_executor(
                    None, # Use default executor
                    self.client.search,
                    query,
                    search_depth,
                    max_results,
                    include_domains,
                    exclude_domains,
                    None # include_answer - keep it simple for now
                )
                
                # Tavily returns results directly in a list of dicts (or similar structure)
                # Example structure (verify with actual Tavily response):
                # [{'title': '...', 'url': '...', 'content': '...', 'score': ...}]
                logger.info(f"Tavily search successful. Received {len(response.get('results', []))} results.")
                return response.get("results", []) # Return the list of results

            except httpx.HTTPStatusError as e: # Catch specific HTTP errors if tavily uses httpx
                 # Check for Rate Limit Error (429)
                 if e.response.status_code == 429:
                     logger.warning(f"Tavily API rate limit hit (Attempt {attempt + 1}). Retrying after {current_delay:.1f}s...")
                     if attempt < TAVILY_MAX_RETRIES:
                         await asyncio.sleep(current_delay)
                         current_delay *= TAVILY_BACKOFF_FACTOR
                         continue # Retry
                     else:
                         logger.error("Max retries reached for Tavily rate limit.")
                         return [{"error": "Tavily API rate limit exceeded after retries."}]
                 # Handle other HTTP errors
                 else:
                     logger.error(f"Tavily API HTTP error: {e.response.status_code} - {e.response.text}", exc_info=True)
                     return [{"error": f"Tavily API HTTP error: {e.response.status_code}"}]
                     
            except httpx.RequestError as e: # Catch network-related errors
                 logger.warning(f"Tavily network error (Attempt {attempt + 1}): {e}. Retrying after {current_delay:.1f}s...")
                 if attempt < TAVILY_MAX_RETRIES:
                      await asyncio.sleep(current_delay)
                      current_delay *= TAVILY_BACKOFF_FACTOR
                      continue # Retry
                 else:
                      logger.error("Max retries reached for Tavily network error.")
                      return [{"error": "Tavily API network error after retries."}]
                      
            except Exception as e: # Catch any other unexpected errors from TavilyClient
                logger.error(f"Unexpected error during Tavily search: {e}", exc_info=True)
                # Stop retrying on unexpected errors
                return [{"error": f"Unexpected error during Tavily search: {e}"}]
                
        # Should only be reached if loop finishes without returning (e.g., max retries hit)
        logger.error("Tavily search failed after all retries.")
        return [{"error": "Tavily search failed after retries."}]

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