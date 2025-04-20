import requests
import os
import logging

logger = logging.getLogger(__name__)

class WebBrowserTool:
    """
    Tool for web search using Google Custom Search API.
    Environment variables: GOOGLE_API_KEY, GOOGLE_CSE_ID
    """
    def __init__(self):
        self.api_key = os.getenv("GOOGLE_API_KEY")
        self.cse_id = os.getenv("GOOGLE_CSE_ID")
        if not self.api_key or not self.cse_id:
            logger.warning("GOOGLE_API_KEY or GOOGLE_CSE_ID not set. Web search will not work.")

    def search(self, query: str) -> str:
        if not self.api_key or not self.cse_id:
            return "Error: Google API Key or CSE ID not defined."
        try:
            url = "https://www.googleapis.com/customsearch/v1"
            params = {
                "key": self.api_key,
                "cx": self.cse_id,
                "q": query
            }
            response = requests.get(url, params=params, timeout=5)
            if response.status_code == 200:
                results = response.json().get("items", [])
                snippets = [item.get("snippet", "") for item in results]
                return f"Web search results: {'; '.join(snippets)}"
            else:
                return f"Error: Web search failed. Status code: {response.status_code}"
        except Exception as e:
            logger.exception("Error during web search: %s", e)
            return f"Error: Web search error: {str(e)}" 