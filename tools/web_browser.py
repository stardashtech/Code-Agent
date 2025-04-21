import requests
import os
import logging
import time
import json
from config import settings # Import settings object

logger = logging.getLogger(__name__)

# Google Custom Search API default limits (adjust if needed)
# Free tier usually has 100 queries/day. Paid tiers have higher QPS.
# Simple exponential backoff for retries.
MAX_RETRIES = 2
INITIAL_RETRY_DELAY = 1 # seconds

class WebBrowserTool:
    """
    Tool for web search using Google Custom Search API.
    Includes basic retry logic, improved error handling, and pagination support.
    Reads configuration from the global settings object.
    """
    def __init__(self):
        # Use settings object, Pydantic handles None
        self.api_key = settings.GOOGLE_API_KEY 
        self.cse_id = settings.GOOGLE_CSE_ID
        if not self.api_key or not self.cse_id:
            logger.warning("GOOGLE_API_KEY or GOOGLE_CSE_ID not set in config/env. Web search tool will be unavailable.")
            self.api_key = None
            self.cse_id = None

    def search(self, query: str, start_index: int = 1, num_results: int = 10, max_retries=MAX_RETRIES) -> str:
        """
        Performs a web search using Google Custom Search API.

        Args:
            query: The search query string.
            start_index: The index of the first result to return (1-based). Max 100.
            num_results: Number of search results to return (1-10).
            max_retries: Max number of retries for transient errors.

        Returns:
            A string containing search result snippets or an error message.
        """
        if not self.api_key or not self.cse_id:
            return "Error: Web search tool is not configured (GOOGLE_API_KEY or GOOGLE_CSE_ID missing)."
            
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": self.api_key,
            "cx": self.cse_id,
            "q": query,
            "start": max(1, start_index), # Ensure start is at least 1
            "num": min(max(1, num_results), 10) # Ensure num is between 1 and 10
        }

        retry_delay = INITIAL_RETRY_DELAY
        for attempt in range(max_retries + 1):
            try:
                response = requests.get(url, params=params, timeout=10) # Increased timeout

                if response.status_code == 200:
                    results = response.json().get("items", [])
                    if not results:
                        return f"Web search for '{query}' returned no results."
                        
                    snippets = [item.get("snippet", "").replace('\n', ' ').strip() for item in results if item.get("snippet")]
                    return f"Web search results (page starting at {params['start']}): {'; '.join(snippets)}"
                    
                # Handle specific HTTP error codes
                elif response.status_code == 429: # Rate limit exceeded
                    error_msg = f"Rate Limit Exceeded (HTTP {response.status_code})"
                    # Attempt to parse error details from Google API response
                    try:
                        error_details = response.json().get('error', {}).get('message', '')
                        if error_details:
                            error_msg += f": {error_details}"
                    except json.JSONDecodeError:
                         pass # Ignore if response is not JSON
                    logger.warning(f"{error_msg}. Retrying in {retry_delay}s...")
                elif 400 <= response.status_code < 500:
                    # Client-side error (bad request, invalid key/cx, etc.) - likely not retryable
                    error_msg = f"Client Error (HTTP {response.status_code})"
                    try:
                        error_details = response.json().get('error', {}).get('message', '')
                        if error_details:
                            error_msg += f": {error_details}"
                    except json.JSONDecodeError:
                         pass 
                    logger.error(f"Web search failed: {error_msg}")
                    return f"Error: Web search failed ({error_msg}). Please check configuration or query."
                else: # Server-side errors (5xx) or unexpected codes - potentially retryable
                     error_msg = f"Server Error or Unexpected Status (HTTP {response.status_code})"
                     logger.warning(f"{error_msg}. Retrying in {retry_delay}s...")

                # If we are here, it means an error occurred that might be retryable
                if attempt == max_retries:
                    logger.error(f"Web search failed after {max_retries} retries: {error_msg}")
                    return f"Error: Web search failed after retries ({error_msg})"
                
                time.sleep(retry_delay)
                retry_delay *= 2 # Exponential backoff
                continue # Go to next attempt

            except requests.exceptions.Timeout as e:
                logger.warning(f"Web search timed out: {e}. Retrying in {retry_delay}s...")
                if attempt == max_retries:
                    return f"Error: Web search timed out after retries."
                time.sleep(retry_delay)
                retry_delay *= 2
            except requests.exceptions.RequestException as e:
                # Catch other connection errors, etc.
                logger.error(f"Web search connection error: {e}")
                # Usually not retryable unless transient network issue, but we retry anyway
                if attempt == max_retries:
                    return f"Error: Web search connection error after retries: {e}"
                time.sleep(retry_delay)
                retry_delay *= 2
            except Exception as e:
                # Catch unexpected errors during the request/processing
                logger.exception("Unexpected error during web search: %s", e)
                return f"Error: Unexpected web search error: {str(e)}" # Don't retry unexpected errors

        # Should not be reached if logic is correct
        return "Error: Web search failed after exhausting retries." 