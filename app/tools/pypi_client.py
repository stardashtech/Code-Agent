import logging
import aiohttp
import asyncio
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)

class PyPiClientError(Exception):
    """Custom exception for PyPiClient errors."""
    pass

class PyPiClient:
    """
    An asynchronous client to interact with the PyPI (Python Package Index) JSON API.
    Provides methods to retrieve package details.
    """
    BASE_URL = "https://pypi.org/pypi"

    def __init__(self, session: Optional[aiohttp.ClientSession] = None):
        """
        Initializes the PyPiClient.

        Args:
            session: An optional existing aiohttp.ClientSession. If None, a new one is created.
                     It's recommended to pass a session from the calling application
                     for better resource management (e.g., connection pooling).
        """
        self._session = session
        self._created_session = False
        if self._session is None:
            # If no session is provided, create one internally.
            # Note: The application using this client is responsible for closing
            # this session by calling client.close_session() if it was created here.
            try:
                 self._session = aiohttp.ClientSession()
                 self._created_session = True
                 logger.info("PyPiClient created its own aiohttp session.")
            except Exception as e:
                 logger.error(f"Failed to create internal aiohttp session for PyPiClient: {e}", exc_info=True)
                 self._session = None # Ensure session is None if creation failed
                 raise PyPiClientError(f"Failed to create aiohttp session: {e}") from e


    async def close_session(self):
        """Closes the aiohttp session if it was created internally by this client."""
        if self._created_session and self._session and not self._session.closed:
            await self._session.close()
            logger.info("PyPiClient closed its internally created aiohttp session.")
        # Reset flags even if closing failed or session didn't exist
        self._created_session = False
        # self._session = None # Optionally set session to None after closing

    async def _request(self, url: str) -> Optional[Dict[str, Any]]:
        """
        Performs an asynchronous GET request and returns the JSON response.

        Args:
            url: The URL to request.

        Returns:
            The JSON response as a dictionary, or None if not found (404).

        Raises:
            PyPiClientError: If the request fails with non-404 status or other errors occur.
        """
        if not self._session or self._session.closed:
            # Handle case where session wasn't created or was closed prematurely
             if self._created_session: # If we created it, try recreating
                 logger.warning("PyPiClient session was closed unexpectedly. Recreating.")
                 try:
                      self._session = aiohttp.ClientSession()
                 except Exception as e:
                      logger.error(f"Failed to recreate internal aiohttp session for PyPiClient: {e}", exc_info=True)
                      self._session = None # Ensure session is None if recreation failed
                      raise PyPiClientError("aiohttp session is closed and could not be recreated.")
             else: # If externally managed, we can't recreate it
                 raise PyPiClientError("aiohttp session is closed or not available.")

        logger.debug(f"Making PyPI request: GET {url}")
        try:
            async with self._session.get(url, timeout=15) as response: # Added timeout
                if response.status == 200:
                    try:
                        data = await response.json(content_type=None) # Allow any content type for flexibility
                        logger.debug(f"PyPI request successful for {url}")
                        return data
                    except (aiohttp.ContentTypeError, ValueError, TypeError) as json_err: # Catch more JSON errors
                         # Attempt to read text even if JSON fails
                         try:
                              error_text = await response.text()
                         except Exception:
                              error_text = "(Could not read response text)"
                         logger.error(f"Failed to decode JSON response from {url}. Status: {response.status}, Content-Type: {response.headers.get('Content-Type')}, Response: {error_text[:200]}...", exc_info=True)
                         raise PyPiClientError(f"Failed to decode JSON from {url}: {json_err}") from json_err
                elif response.status == 404:
                     logger.warning(f"PyPI resource not found (404): {url}")
                     return None # Return None for 404s
                else:
                    error_text = await response.text()
                    logger.error(f"PyPI request failed for {url}. Status: {response.status}, Response: {error_text[:200]}...")
                    # Raise specific error for non-200/404 status
                    raise PyPiClientError(f"PyPI request failed for {url} with status {response.status}")

        except aiohttp.ClientError as e:
            logger.error(f"Network or client error during PyPI request to {url}: {e}", exc_info=True)
            raise PyPiClientError(f"Network error accessing {url}: {e}") from e
        except asyncio.TimeoutError:
             logger.error(f"Timeout during PyPI request to {url}")
             raise PyPiClientError(f"Request timed out for {url}")
        except Exception as e:
             # Catch any other unexpected errors
             logger.error(f"Unexpected error during PyPI request to {url}: {e}", exc_info=True)
             raise PyPiClientError(f"Unexpected error accessing {url}: {e}")


    async def get_package_details(self, package_name: str, version: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Retrieves details for a specific package, optionally for a specific version.

        Args:
            package_name: The exact name of the package (case-insensitive matching is handled by PyPI).
            version: The specific version to retrieve (e.g., "1.0.0"). If None, fetches latest stable.

        Returns:
            A dictionary containing package details (like info, releases), or None if not found or error occurs.
        """
        if not package_name:
             logger.warning("get_package_details called with empty package_name.")
             return None

        package_name_safe = package_name.strip() # Basic sanitization

        if version:
            version_safe = version.strip()
            url = f"{self.BASE_URL}/{package_name_safe}/{version_safe}/json"
            log_msg = f"Fetching details for PyPI package: {package_name_safe} version: {version_safe}"
        else:
            url = f"{self.BASE_URL}/{package_name_safe}/json" # Fetches latest by default
            log_msg = f"Fetching details for PyPI package: {package_name_safe} (latest)"

        logger.info(log_msg)
        try:
             details = await self._request(url)
             # _request returns None for 404, so we just return it
             return details
        except PyPiClientError as e:
             # Log specific errors from _request but return None to the caller
             logger.warning(f"Could not fetch details for {package_name_safe}: {e}")
             return None
        except Exception as e:
             # Catch any unexpected errors during the call
             logger.error(f"Unexpected error in get_package_details for {package_name_safe}: {e}", exc_info=True)
             return None

    # Note: PyPI does not provide a stable, documented JSON API for searching.
    # Implementing search would typically involve web scraping or using third-party tools/APIs.
    # For robustness, we focus on fetching details by exact name.
    # async def search_packages(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
    #     pass # Not implemented

# Example Usage Section (for direct testing of this module)
async def _test_client():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    logger.info("Testing PyPiClient...")
    # It's better to manage the session outside if used in a larger app,
    # but for standalone testing, letting the client create it is okay.
    client = PyPiClient()
    results = {}
    try:
        # Test getting details for a known package
        package_name = "fastapi"
        logger.info(f"Getting details for '{package_name}'...")
        details = await client.get_package_details(package_name)
        if details and isinstance(details, dict) and 'info' in details:
            logger.info(f"Success: Got details for '{package_name}'. Latest version: {details.get('info', {}).get('version')}")
            results['get_success'] = True
        else:
            logger.error(f"Failure: Could not get details for '{package_name}' or details format invalid.")
            results['get_success'] = False

        # Test getting details for a specific version
        version = "0.100.0"
        logger.info(f"Getting details for '{package_name}' version '{version}'...")
        version_details = await client.get_package_details(package_name, version)
        if version_details and isinstance(version_details, dict) and version_details.get('info', {}).get('version') == version:
            logger.info(f"Success: Got details for '{package_name}' version '{version}'.")
            results['get_version_success'] = True
        else:
            logger.error(f"Failure: Could not get details for '{package_name}' version '{version}' or version mismatch.")
            results['get_version_success'] = False

        # Test getting details for a non-existent package
        non_existent_package = "this-package-definitely-does-not-exist-12345xyz"
        logger.info(f"Getting details for non-existent package '{non_existent_package}'...")
        non_existent_details = await client.get_package_details(non_existent_package)
        if non_existent_details is None:
            logger.info(f"Success: Correctly handled non-existent package '{non_existent_package}' (returned None).")
            results['get_nonexistent_success'] = True
        else:
            logger.error(f"Failure: Incorrectly handled non-existent package '{non_existent_package}'. Details: {type(non_existent_details)}")
            results['get_nonexistent_success'] = False

        # Test with empty package name
        logger.info("Getting details for empty package name...")
        empty_details = await client.get_package_details("")
        if empty_details is None:
             logger.info(f"Success: Correctly handled empty package name (returned None).")
             results['get_empty_success'] = True
        else:
             logger.error(f"Failure: Incorrectly handled empty package name.")
             results['get_empty_success'] = False


    except PyPiClientError as e:
        logger.error(f"A PyPiClientError occurred during test: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during PyPiClient test: {e}", exc_info=True)
    finally:
        # Ensure the session created by the client is closed
        if client: # Check if client was successfully initialized
             await client.close_session()
        logger.info("PyPiClient test finished.")
        logger.info(f"Test Results Summary: {results}")


if __name__ == "__main__":
    # To run this test script directly: python -m app.tools.pypi_client
    # Note: Running top-level async code requires asyncio.run()
    asyncio.run(_test_client()) 