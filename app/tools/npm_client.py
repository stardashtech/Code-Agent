import logging
import aiohttp
import asyncio
import urllib.parse
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)

class NpmClientError(Exception):
    """Custom exception for NpmClient errors."""
    pass

class NpmClient:
    """
    An asynchronous client to interact with the npm registry API.
    Provides methods to search for packages and retrieve package details.
    """
    BASE_URL = "https://registry.npmjs.org"
    SEARCH_URL = f"{BASE_URL}/-/v1/search"

    def __init__(self, session: Optional[aiohttp.ClientSession] = None):
        """
        Initializes the NpmClient.

        Args:
            session: An optional existing aiohttp.ClientSession. If None, a new one is created.
        """
        self._session = session
        self._created_session = False
        if self._session is None:
            try:
                self._session = aiohttp.ClientSession()
                self._created_session = True
                logger.info("NpmClient created its own aiohttp session.")
            except Exception as e:
                logger.error(f"Failed to create internal aiohttp session for NpmClient: {e}", exc_info=True)
                self._session = None
                raise NpmClientError(f"Failed to create aiohttp session: {e}") from e

    async def close_session(self):
        """Closes the aiohttp session if it was created internally by this client."""
        if self._created_session and self._session and not self._session.closed:
            await self._session.close()
            logger.info("NpmClient closed its internally created aiohttp session.")
        self._created_session = False

    async def _request(self, url: str, params: Optional[Dict] = None) -> Optional[Dict[str, Any]]:
        """
        Performs an asynchronous GET request and returns the JSON response.

        Args:
            url: The URL to request.
            params: Optional query parameters.

        Returns:
            The JSON response as a dictionary, or None if not found (404).

        Raises:
            NpmClientError: If the request fails with non-404 status or other errors occur.
        """
        if not self._session or self._session.closed:
            if self._created_session:
                logger.warning("NpmClient session was closed unexpectedly. Recreating.")
                try:
                     self._session = aiohttp.ClientSession()
                except Exception as e:
                     logger.error(f"Failed to recreate internal aiohttp session for NpmClient: {e}", exc_info=True)
                     self._session = None
                     raise NpmClientError("aiohttp session is closed and could not be recreated.")
            else:
                raise NpmClientError("aiohttp session is closed or not available.")

        logger.debug(f"Making npm request: GET {url} with params: {params}")
        try:
            async with self._session.get(url, params=params, timeout=15) as response:
                if response.status == 200:
                    try:
                        data = await response.json(content_type=None)
                        logger.debug(f"npm request successful for {url}")
                        return data
                    except (aiohttp.ContentTypeError, ValueError, TypeError) as json_err:
                        try: error_text = await response.text()
                        except Exception: error_text = "(Could not read response text)"
                        logger.error(f"Failed to decode JSON response from {url}. Status: {response.status}, Content-Type: {response.headers.get('Content-Type')}, Response: {error_text[:200]}...", exc_info=True)
                        raise NpmClientError(f"Failed to decode JSON from {url}: {json_err}") from json_err
                elif response.status == 404:
                     logger.warning(f"npm resource not found (404): {url}")
                     return None
                else:
                    error_text = await response.text()
                    logger.error(f"npm request failed for {url}. Status: {response.status}, Response: {error_text[:200]}...")
                    raise NpmClientError(f"npm request failed for {url} with status {response.status}")

        except aiohttp.ClientError as e:
            logger.error(f"Network or client error during npm request to {url}: {e}", exc_info=True)
            raise NpmClientError(f"Network error accessing {url}: {e}") from e
        except asyncio.TimeoutError:
             logger.error(f"Timeout during npm request to {url}")
             raise NpmClientError(f"Request timed out for {url}")
        except Exception as e:
             logger.error(f"Unexpected error during npm request to {url}: {e}", exc_info=True)
             raise NpmClientError(f"Unexpected error accessing {url}: {e}")

    async def get_package_details(self, package_name: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves details for a specific npm package.

        Args:
            package_name: The exact name of the package (can include scope like @angular/core).

        Returns:
            A dictionary containing package details, or None if not found or error occurs.
        """
        if not package_name:
             logger.warning("get_package_details called with empty package_name.")
             return None

        # URL encode the package name, especially important for scoped packages like @org/package
        package_name_safe = urllib.parse.quote(package_name.strip(), safe='')
        url = f"{self.BASE_URL}/{package_name_safe}"
        log_msg = f"Fetching details for npm package: {package_name}"

        logger.info(log_msg)
        try:
             details = await self._request(url)
             return details
        except NpmClientError as e:
             logger.warning(f"Could not fetch details for npm package {package_name}: {e}")
             return None
        except Exception as e:
             logger.error(f"Unexpected error in get_package_details for npm package {package_name}: {e}", exc_info=True)
             return None

    async def search_packages(self, query: str, size: int = 20, from_result: int = 0) -> Optional[List[Dict[str, Any]]]:
        """
        Searches for npm packages matching the query.

        Args:
            query: The search term.
            size: Number of results to return (default: 20).
            from_result: Offset for pagination (default: 0).

        Returns:
            A list of package objects found, or None if an error occurs.
            Each package object contains details like name, version, description, score, etc.
        """
        if not query:
            logger.warning("search_packages called with empty query.")
            return [] # Return empty list for empty query

        params = {"text": query, "size": size, "from": from_result}
        log_msg = f"Searching npm packages for query: '{query}' (size={size}, from={from_result})"
        logger.info(log_msg)

        try:
            results = await self._request(self.SEARCH_URL, params=params)
            if results and 'objects' in results and isinstance(results['objects'], list):
                # Extract the list of package objects
                packages = [item.get('package', {}) for item in results['objects']]
                logger.info(f"Found {len(packages)} npm packages matching '{query}'.")
                return packages
            elif results is None: # Handle 404 or other cases where _request returns None
                 logger.warning(f"npm search for '{query}' returned no results (or 404).")
                 return []
            else:
                 logger.error(f"npm search for '{query}' returned unexpected format: {results}")
                 return None # Indicate an error occurred
        except NpmClientError as e:
            logger.warning(f"Could not search npm packages for '{query}': {e}")
            return None # Indicate an error occurred
        except Exception as e:
            logger.error(f"Unexpected error in search_packages for '{query}': {e}", exc_info=True)
            return None # Indicate an error occurred

# Example Usage Section
async def _test_client():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    logger.info("Testing NpmClient...")
    client = NpmClient()
    results = {}
    try:
        # Test getting details for a known package
        package_name = "react"
        logger.info(f"Getting details for '{package_name}'...")
        details = await client.get_package_details(package_name)
        if details and isinstance(details, dict) and details.get('name') == package_name:
            logger.info(f"Success: Got details for '{package_name}'. Latest version: {details.get('dist-tags', {}).get('latest')}")
            results['get_success'] = True
        else:
            logger.error(f"Failure: Could not get details for '{package_name}' or details format invalid.")
            results['get_success'] = False

        # Test getting details for a scoped package
        scoped_package_name = "@angular/core"
        logger.info(f"Getting details for scoped package '{scoped_package_name}'...")
        scoped_details = await client.get_package_details(scoped_package_name)
        if scoped_details and isinstance(scoped_details, dict) and scoped_details.get('name') == scoped_package_name:
            logger.info(f"Success: Got details for '{scoped_package_name}'.")
            results['get_scoped_success'] = True
        else:
            logger.error(f"Failure: Could not get details for '{scoped_package_name}'.")
            results['get_scoped_success'] = False

        # Test getting details for a non-existent package
        non_existent_package = "this-npm-package-does-not-exist-12345xyz"
        logger.info(f"Getting details for non-existent package '{non_existent_package}'...")
        non_existent_details = await client.get_package_details(non_existent_package)
        if non_existent_details is None:
            logger.info(f"Success: Correctly handled non-existent npm package '{non_existent_package}' (returned None).")
            results['get_nonexistent_success'] = True
        else:
            logger.error(f"Failure: Incorrectly handled non-existent npm package '{non_existent_package}'. Details: {type(non_existent_details)}")
            results['get_nonexistent_success'] = False

        # Test searching for packages
        search_query = "request promise"
        logger.info(f"Searching for packages matching '{search_query}'...")
        search_results = await client.search_packages(search_query, size=5)
        if search_results is not None and isinstance(search_results, list):
            logger.info(f"Success: Found {len(search_results)} packages for '{search_query}'. First result: {search_results[0].get('name') if search_results else 'N/A'}")
            results['search_success'] = True
        else:
            logger.error(f"Failure: Search failed or returned invalid format for '{search_query}'. Result: {search_results}")
            results['search_success'] = False

        # Test search with empty query
        logger.info(f"Searching for packages matching empty query...")
        empty_search_results = await client.search_packages("")
        if empty_search_results is not None and isinstance(empty_search_results, list) and len(empty_search_results) == 0:
            logger.info(f"Success: Correctly handled empty search query (returned empty list).")
            results['search_empty_success'] = True
        else:
             logger.error(f"Failure: Empty search query not handled correctly. Result: {empty_search_results}")
             results['search_empty_success'] = False


    except NpmClientError as e:
        logger.error(f"An NpmClientError occurred during test: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during NpmClient test: {e}", exc_info=True)
    finally:
        if client:
             await client.close_session()
        logger.info("NpmClient test finished.")
        logger.info(f"Test Results Summary: {results}")


if __name__ == "__main__":
    # To run this test script directly: python -m app.tools.npm_client
    asyncio.run(_test_client()) 