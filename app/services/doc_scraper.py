import logging
import aiohttp
import asyncio
from typing import Optional, Dict, Any, List
from bs4 import BeautifulSoup, Comment # Import BeautifulSoup

logger = logging.getLogger(__name__)

class DocScraperError(Exception):
    """Custom exception for DocumentationScraper errors."""
    pass

class DocumentationScraper:
    """
    An asynchronous service to scrape and extract content from documentation web pages.
    """

    def __init__(self, session: Optional[aiohttp.ClientSession] = None):
        """
        Initializes the DocumentationScraper.

        Args:
            session: An optional existing aiohttp.ClientSession. If None, a new one is created.
        """
        self._session = session
        self._created_session = False
        if self._session is None:
            try:
                # Add standard headers to mimic a browser request
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.9',
                }
                self._session = aiohttp.ClientSession(headers=headers)
                self._created_session = True
                logger.info("DocumentationScraper created its own aiohttp session.")
            except Exception as e:
                logger.error(f"Failed to create internal aiohttp session for DocumentationScraper: {e}", exc_info=True)
                self._session = None
                raise DocScraperError(f"Failed to create aiohttp session: {e}") from e

    async def close_session(self):
        """Closes the aiohttp session if it was created internally by this client."""
        if self._created_session and self._session and not self._session.closed:
            await self._session.close()
            logger.info("DocumentationScraper closed its internally created aiohttp session.")
        self._created_session = False

    async def _fetch_html(self, url: str) -> Optional[str]:
        """Fetches HTML content from a URL."""
        if not self._session or self._session.closed:
             if self._created_session: # If we created it, try recreating
                 logger.warning("DocumentationScraper session was closed unexpectedly. Recreating.")
                 try:
                      # Recreate with headers
                      headers = {
                          'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                          'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
                          'Accept-Language': 'en-US,en;q=0.9',
                      }
                      self._session = aiohttp.ClientSession(headers=headers)
                 except Exception as e:
                      logger.error(f"Failed to recreate internal aiohttp session for DocumentationScraper: {e}", exc_info=True)
                      self._session = None
                      raise DocScraperError("aiohttp session is closed and could not be recreated.")
             else: # If externally managed, we can't recreate it
                 raise DocScraperError("aiohttp session is closed or not available.")

        logger.debug(f"Fetching HTML from URL: {url}")
        try:
            # Allow redirects and handle potential SSL errors more gracefully
            async with self._session.get(url, timeout=20, allow_redirects=True, ssl=False) as response: # Increased timeout, allow redirects, ignore basic SSL verification issues
                if response.status == 200:
                    try:
                        # Check content type before reading
                        content_type = response.headers.get('Content-Type', '').lower()
                        if 'html' not in content_type:
                             logger.warning(f"URL {url} returned non-HTML content type: {content_type}. Skipping scrape.")
                             return None
                        html_content = await response.text()
                        logger.debug(f"Successfully fetched HTML from {url} (Length: {len(html_content)})" )
                        return html_content
                    except Exception as read_err:
                         logger.error(f"Error reading response text from {url}: {read_err}", exc_info=True)
                         raise DocScraperError(f"Error reading response from {url}: {read_err}") from read_err
                else:
                    logger.warning(f"Failed to fetch {url}. Status: {response.status}")
                    return None # Return None for non-200 status

        except aiohttp.ClientError as e:
            logger.error(f"Network or client error fetching {url}: {e}", exc_info=True)
            raise DocScraperError(f"Network error accessing {url}: {e}") from e
        except asyncio.TimeoutError:
             logger.error(f"Timeout fetching {url}")
             raise DocScraperError(f"Request timed out for {url}")
        except Exception as e:
             logger.error(f"Unexpected error fetching {url}: {e}", exc_info=True)
             raise DocScraperError(f"Unexpected error accessing {url}: {e}")


    def _extract_text_from_html(self, html_content: str) -> str:
        """Extracts meaningful text content from HTML using BeautifulSoup."""
        if not html_content:
            return ""
        try:
            soup = BeautifulSoup(html_content, 'lxml') # Use lxml for better performance and robustness

            # Remove script, style, header, footer, nav, and comment tags
            tags_to_remove = ['script', 'style', 'header', 'footer', 'nav', 'aside', 'form', 'button', 'select', 'textarea', 'iframe']
            for tag in soup(tags_to_remove):
                tag.decompose()

            # Remove comments
            for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
                comment.extract()

            # Attempt to find the main content area (common tags/attributes)
            # This requires heuristics and might need adjustment per site type
            main_content = soup.find('main') or \
                           soup.find('article') or \
                           soup.find(id='content') or \
                           soup.find(class_='content') or \
                           soup.find(role='main')
                           # Add more potential selectors

            target_element = main_content if main_content else soup.body # Fallback to body if main content not found

            if not target_element:
                 logger.warning("Could not find <body> tag in HTML, returning empty string.")
                 return ""

            # Get text, try to preserve some structure with separators
            text_parts = []
            # Use .stripped_strings for better whitespace handling
            for string in target_element.stripped_strings:
                 text_parts.append(string)

            # Join parts, potentially add separators based on block elements later if needed
            full_text = "\n".join(text_parts)

            # Further clean up multiple newlines
            cleaned_text = "\n".join([line.strip() for line in full_text.splitlines() if line.strip()])

            logger.debug(f"Extracted text length: {len(cleaned_text)}")
            return cleaned_text

        except Exception as e:
            logger.error(f"Error parsing HTML with BeautifulSoup: {e}", exc_info=True)
            # Return empty string or raise? Returning empty is safer for the flow.
            return ""


    async def scrape_url(self, url: str) -> Optional[str]:
        """
        Fetches HTML from a URL and extracts the main textual content.

        Args:
            url: The URL of the documentation page to scrape.

        Returns:
            The extracted text content as a string, or None if fetching/parsing fails.
        """
        if not url or not (url.startswith('http://') or url.startswith('https://')):
            logger.warning(f"Invalid or missing URL provided to scrape_url: {url}")
            return None

        logger.info(f"Attempting to scrape documentation content from: {url}")
        try:
            html_content = await self._fetch_html(url)
            if html_content:
                extracted_text = self._extract_text_from_html(html_content)
                if not extracted_text:
                     logger.warning(f"Could not extract meaningful text from {url}")
                     return None # Return None if extraction yields nothing
                logger.info(f"Successfully scraped and extracted text from {url}")
                return extracted_text
            else:
                # _fetch_html already logged the warning/error
                return None
        except DocScraperError as e:
            logger.warning(f"Could not scrape content from {url}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error during scrape_url for {url}: {e}", exc_info=True)
            return None

# Example Usage Section
async def _test_scraper():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    logger.info("Testing DocumentationScraper...")
    # Example URL - use a relatively simple page for testing
    test_url = "https://docs.python.org/3/library/asyncio-task.html"
    scraper = DocumentationScraper()
    scraped_content = None
    try:
        logger.info(f"Scraping URL: {test_url}")
        scraped_content = await scraper.scrape_url(test_url)

        if scraped_content:
            logger.info(f"Successfully scraped content from {test_url}.")
            print("-" * 20 + " Scraped Content (First 500 chars) " + "-" * 20)
            print(scraped_content[:500] + "...")
            print("-" * (44 + len(" Scraped Content (First 500 chars) ")))
            assert len(scraped_content) > 100 # Basic check that we got *something*
        else:
            logger.error(f"Failed to scrape content from {test_url}.")

        # Test with a bad URL
        bad_url = "https://invalid-domain-that-does-not-exist-12345.xyz/"
        logger.info(f"Attempting to scrape invalid URL: {bad_url}")
        bad_content = await scraper.scrape_url(bad_url)
        if bad_content is None:
            logger.info(f"Successfully handled invalid URL (returned None).")
        else:
            logger.error("Failed to handle invalid URL correctly.")

        # Test with a non-HTML URL (e.g., PDF - should return None)
        pdf_url = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"
        logger.info(f"Attempting to scrape non-HTML URL: {pdf_url}")
        pdf_content = await scraper.scrape_url(pdf_url)
        if pdf_content is None:
            logger.info(f"Successfully handled non-HTML URL (returned None).")
        else:
             logger.error("Failed to handle non-HTML URL correctly.")


    except DocScraperError as e:
        logger.error(f"A DocScraperError occurred during test: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during DocumentationScraper test: {e}", exc_info=True)
    finally:
        if scraper:
            await scraper.close_session()
        logger.info("DocumentationScraper test finished.")


if __name__ == "__main__":
    # To run this test script directly: python -m app.services.doc_scraper
    # Ensure beautifulsoup4 and lxml are installed: pip install beautifulsoup4 lxml
    asyncio.run(_test_scraper()) 