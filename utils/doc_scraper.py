import requests
from typing import Optional
from urllib.parse import urlparse
import time # Added for retry logic
import random # Added for jitter

# Optional: Consider adding BeautifulSoup4 for more robust HTML parsing
# from bs4 import BeautifulSoup

# Consider adding logging

class DocumentationScraper:
    """
    Utility to fetch content from documentation URLs.
    Provides basic fetching and includes retry logic.
    Potential future HTML parsing/cleaning.
    """
    MAX_RETRIES = 3
    INITIAL_BACKOFF = 0.5 # seconds

    def __init__(self):
        self.headers = {
            # Be a good citizen, identify your bot
            "User-Agent": "CodeAgent/1.0 (Documentation Scraper)", 
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Referer": "https://www.google.com/", # Common referer
            "DNT": "1", # Do Not Track
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1"
        }
        # Using a session object is generally better for connection reuse and header persistence
        self.session = requests.Session()
        self.session.headers.update(self.headers)

    def _is_valid_url(self, url: str) -> bool:
        """Basic check if the URL seems valid."""
        try:
            result = urlparse(url)
            return all([result.scheme in ['http', 'https'], result.netloc])
        except ValueError:
            return False

    def fetch_content(self, url: str) -> Optional[str]:
        """
        Fetches the raw HTML or text content from a given documentation URL with retry logic.

        Args:
            url: The URL to fetch content from.

        Returns:
            The raw content (likely HTML) as a string, or None on failure.
        """
        if not self._is_valid_url(url):
             print(f"Error: Invalid URL provided to scraper: {url}")
             return None

        retries = 0
        while retries < self.MAX_RETRIES:
            try:
                # Use GET request with session and timeout
                response = self.session.get(url, timeout=10)
                
                # Check for retryable status codes before calling raise_for_status
                if response.status_code == 429: # Too Many Requests
                     print(f"Scraper rate limit hit (429) for {url}. Attempt {retries + 1}/{self.MAX_RETRIES}")
                elif response.status_code >= 500: # Server errors
                     print(f"Scraper server error ({response.status_code}) for {url}. Attempt {retries + 1}/{self.MAX_RETRIES}")
                else:
                    response.raise_for_status() # Check for other HTTP errors (4xx)
                    content_type = response.headers.get('content-type', '').lower()
                    if 'html' in content_type or 'text' in content_type:
                        return response.text
                    else:
                         print(f"Warning: Fetched content from {url} is not HTML or text (type: {content_type}). Skipping.")
                         return None

            except requests.exceptions.Timeout as e:
                 print(f"Error: Timeout while fetching documentation from {url}. Attempt {retries + 1}/{self.MAX_RETRIES}")
            except requests.exceptions.ConnectionError as e:
                 print(f"Error fetching documentation (connection error) from {url}: {e}. Attempt {retries + 1}/{self.MAX_RETRIES}")
            except requests.exceptions.HTTPError as e:
                 status_code = e.response.status_code
                 if status_code < 500 and status_code != 429:
                      print(f"Scraper HTTP error ({status_code}) for {url}: {e}. Not retrying.")
                      return None 
                 print(f"Scraper HTTP error ({status_code}) occurred for {url}. Attempt {retries + 1}/{self.MAX_RETRIES}")
            except requests.exceptions.RequestException as e:
                 print(f"Error fetching documentation from {url}: {e}. Attempt {retries + 1}/{self.MAX_RETRIES}")
            except Exception as e:
                 print(f"An unexpected error occurred while scraping {url}: {e}. Not retrying.")
                 return None

            # Calculate backoff and sleep if retry is needed
            wait_time = self.INITIAL_BACKOFF * (2 ** retries) + random.uniform(0, 0.5)
            print(f"Retrying in {wait_time:.2f} seconds...")
            time.sleep(wait_time)
            retries += 1

        print(f"Scraper failed to fetch content from {url} after {self.MAX_RETRIES} retries.")
        return None

# Example Usage (for testing purposes)
if __name__ == '__main__':
    scraper = DocumentationScraper()
    # Example URL (replace with actual one for testing)
    # test_url = "https://requests.readthedocs.io/en/latest/"
    test_url = "https://docs.python.org/3/library/abc.html" 
    
    print(f"Attempting to fetch content from: {test_url}")
    content = scraper.fetch_content(test_url)

    if content:
        print(f"Successfully fetched content (first 500 chars):\n{content[:500]}...")
        # Example: Further processing with LLM could happen here
    else:
        print("Failed to fetch content.") 