"""
Module that provides a client for interacting with the NuGet API.
"""
import logging
import aiohttp
import asyncio
import urllib.parse
from typing import Optional, Dict, Any, List, Set

logger = logging.getLogger(__name__)

class NuGetClientError(Exception):
    """Custom exception for NuGetClient errors."""
    pass

class NuGetClient:
    """
    An asynchronous client to interact with the NuGet API.
    Provides methods to search for packages and retrieve package details.
    """
    BASE_URL = "https://api.nuget.org/v3"
    SEARCH_URL = f"{BASE_URL}/search"
    QUERY_URL = f"{BASE_URL}/query"
    
    def __init__(self, api_url: str = "https://api.nuget.org/v3/index.json"):
        """
        Initializes the NuGetClient.
        
        Args:
            api_url: URL of the NuGet API, defaults to the official v3 endpoint
        """
        self.api_url = api_url
        self.session = None
        self.search_endpoint = None
        self.package_base_address = None
        self._create_session()
        logger.info(f"NuGetClient initialized with API URL: {api_url}")
    
    def _create_session(self):
        """Create an aiohttp session for making requests."""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
    
    async def close_session(self):
        """Closes the aiohttp session if it exists."""
        if self.session and not self.session.closed:
            await self.session.close()
            logger.info("NuGetClient session closed")
    
    async def initialize(self):
        """Initialize the client by discovering the service endpoints."""
        if self.search_endpoint and self.package_base_address:
            # Already initialized
            return
            
        try:
            # Get NuGet service index
            index_data = await self._request(self.api_url)
            resources = index_data.get("resources", [])
            
            # Find the search endpoint and package base address
            for resource in resources:
                if resource.get("@type") == "SearchQueryService":
                    self.search_endpoint = resource.get("@id")
                elif resource.get("@type") == "PackageBaseAddress/3.0.0":
                    self.package_base_address = resource.get("@id")
                    
            if not self.search_endpoint:
                logger.error("Could not find NuGet search endpoint in service index")
                raise NuGetClientError("NuGet search endpoint not found")
                
            if not self.package_base_address:
                logger.error("Could not find NuGet package base address in service index")
                raise NuGetClientError("NuGet package base address not found")
                
            logger.info(f"NuGetClient initialized with search endpoint: {self.search_endpoint}")
            logger.info(f"NuGetClient initialized with package base address: {self.package_base_address}")
            
        except Exception as e:
            logger.error(f"Error initializing NuGetClient: {str(e)}")
            raise NuGetClientError(f"Failed to initialize NuGetClient: {str(e)}")
    
    async def _request(self, url: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Performs an asynchronous GET request and returns the JSON response.
        
        Args:
            url: The URL to request.
            params: Optional query parameters.
            
        Returns:
            The JSON response as a dictionary.
            
        Raises:
            NuGetClientError: If the request fails with non-200 status or other errors occur.
        """
        self._create_session()
        
        logger.debug(f"Making NuGet request: GET {url} with params: {params}")
        try:
            async with self.session.get(url, params=params, timeout=15) as response:
                if response.status == 200:
                    try:
                        data = await response.json()
                        logger.debug(f"NuGet request successful for {url}")
                        return data
                    except (aiohttp.ContentTypeError, ValueError, TypeError) as json_err:
                        try: error_text = await response.text()
                        except Exception: error_text = "(Could not read response text)"
                        logger.error(f"Failed to decode JSON response from {url}. Status: {response.status}, Content-Type: {response.headers.get('Content-Type')}, Response: {error_text[:200]}...", exc_info=True)
                        raise NuGetClientError(f"Failed to decode JSON from {url}: {json_err}") from json_err
                elif response.status == 404:
                    logger.warning(f"NuGet resource not found (404): {url}")
                    return None
                else:
                    error_text = await response.text()
                    logger.error(f"NuGet request failed for {url}. Status: {response.status}, Response: {error_text[:200]}...")
                    raise NuGetClientError(f"NuGet request failed for {url} with status {response.status}")
                    
        except aiohttp.ClientError as e:
            logger.error(f"Network or client error during NuGet request to {url}: {e}", exc_info=True)
            raise NuGetClientError(f"Network error accessing {url}: {e}") from e
        except asyncio.TimeoutError:
            logger.error(f"Timeout during NuGet request to {url}")
            raise NuGetClientError(f"Request timed out for {url}")
        except Exception as e:
            logger.error(f"Unexpected error during NuGet request to {url}: {e}", exc_info=True)
            raise NuGetClientError(f"Unexpected error accessing {url}: {e}")
            
    async def search_packages(self, query: str, prerelease: bool = False, 
                             take: int = 20, skip: int = 0) -> Dict[str, Any]:
        """
        Searches for NuGet packages matching the query.
        
        Args:
            query: The search term.
            prerelease: Whether to include prerelease packages. Defaults to False.
            take: Number of results to return (max 1000).
            skip: Number of results to skip for pagination.
            
        Returns:
            A dictionary containing search results.
        """
        await self.initialize()
        
        params = {
            "q": query,
            "prerelease": str(prerelease).lower(),
            "take": min(take, 1000),  # Max is 1000
            "skip": max(0, skip)
        }
        
        log_msg = f"Searching NuGet packages for query: '{query}' (take={take}, skip={skip}, prerelease={prerelease})"
        logger.info(log_msg)
        
        try:
            result = await self._request(self.search_endpoint, params)
            return result
        except NuGetClientError as e:
            logger.warning(f"Could not search NuGet packages for '{query}': {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error in search_packages for '{query}': {e}", exc_info=True)
            return None
            
    async def get_package_versions(self, package_id: str, include_prerelease: bool = False) -> List[str]:
        """
        Get all versions of a package.
        
        Args:
            package_id: The package ID
            include_prerelease: Whether to include prerelease versions
            
        Returns:
            List of version strings
        """
        await self.initialize()
        
        # URL encode the package ID
        safe_package_id = urllib.parse.quote(package_id.lower())
        
        # Get all versions
        url = f"{self.package_base_address}/{safe_package_id}/index.json"
        
        try:
            result = await self._request(url)
            versions = result.get("versions", [])
            
            if not include_prerelease:
                # Filter out prerelease versions
                versions = [v for v in versions if "-" not in v]
                
            return versions
        except Exception as e:
            logger.error(f"Error getting versions for package {package_id}: {str(e)}")
            raise NuGetClientError(f"Failed to get versions for package {package_id}: {str(e)}")
    
    async def get_package_metadata(self, package_id: str, version: Optional[str] = None) -> Dict[str, Any]:
        """
        Get metadata for a package.
        
        Args:
            package_id: The package ID
            version: Optional specific version, defaults to latest
            
        Returns:
            Dictionary with package metadata
        """
        await self.initialize()
        
        # If version is not specified, get the latest version
        if not version:
            versions = await self.get_package_versions(package_id)
            if not versions:
                raise NuGetClientError(f"No versions found for package {package_id}")
            version = versions[-1]  # Assume versions are sorted
            
        # URL encode the package ID
        safe_package_id = urllib.parse.quote(package_id.lower())
        safe_version = urllib.parse.quote(version.lower())
        
        # Get package metadata
        url = f"{self.package_base_address}/{safe_package_id}/{safe_version}/{safe_package_id}.nuspec"
        
        try:
            # This isn't JSON but XML - we'd need to parse it
            # For simplicity, we'll just return the URL and info we already have
            return {
                "id": package_id,
                "version": version,
                "nuspec_url": url,
                "metadata_available": True
            }
        except Exception as e:
            logger.error(f"Error getting metadata for package {package_id} {version}: {str(e)}")
            raise NuGetClientError(f"Failed to get metadata for package {package_id} {version}: {str(e)}")
    
    async def get_package_dependencies(self, package_id: str, version: Optional[str] = None) -> List[Dict[str, str]]:
        """
        Get dependencies for a package.
        
        Note: This is a simplified implementation. In a real implementation,
        you would need to parse the nuspec XML to extract dependencies.
        
        Args:
            package_id: The package ID
            version: Optional specific version, defaults to latest
            
        Returns:
            List of dependencies (simplified)
        """
        metadata = await self.get_package_metadata(package_id, version)
        
        # This is a placeholder - in a real implementation, you would
        # need to download and parse the nuspec file to extract dependencies
        return [
            {"message": "Dependencies information requires parsing the nuspec XML file."},
            {"note": "This is a simplified implementation."},
            {"nuspec_url": metadata.get("nuspec_url", "")}
        ]

    async def get_package_download_url(self, package_id: str, version: Optional[str] = None) -> str:
        """
        Get download URL for a package.
        
        Args:
            package_id: The package ID
            version: Optional specific version, defaults to latest
            
        Returns:
            URL to download the package
        """
        await self.initialize()
        
        # If version is not specified, get the latest version
        if not version:
            versions = await self.get_package_versions(package_id)
            if not versions:
                raise NuGetClientError(f"No versions found for package {package_id}")
            version = versions[-1]  # Assume versions are sorted
            
        # URL encode the package ID
        safe_package_id = urllib.parse.quote(package_id.lower())
        safe_version = urllib.parse.quote(version.lower())
        
        # Construct download URL
        download_url = f"{self.package_base_address}/{safe_package_id}/{safe_version}/{safe_package_id}.{safe_version}.nupkg"
        
        return download_url

# Example Usage
async def _test_client():
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    logger.info("Testing NuGetClient...")
    
    # Example package to test with
    test_package = "Newtonsoft.Json"
    
    client = NuGetClient()
    try:
        # Initialize the client
        await client.initialize()
        
        # Search for packages
        search_results = await client.search_packages("json", take=5)
        if search_results and "data" in search_results and isinstance(search_results["data"], list):
            packages = search_results["data"]
            logger.info(f"Search results: Found {len(packages)} packages")
            for idx, pkg in enumerate(packages):
                logger.info(f"Result {idx+1}: {pkg.get('id')} - {pkg.get('version')}")
                
        # Get package versions
        print("\nVersions of Newtonsoft.Json:")
        versions = await client.get_package_versions("Newtonsoft.Json", include_prerelease=False)
        print(f"Total versions: {len(versions)}")
        print(f"Latest few versions: {versions[-5:] if len(versions) >= 5 else versions}")
        
        # Get package metadata and download URL
        print("\nMetadata for latest Newtonsoft.Json:")
        metadata = await client.get_package_metadata("Newtonsoft.Json")
        print(f"Package ID: {metadata.get('id')}")
        print(f"Version: {metadata.get('version')}")
        
        download_url = await client.get_package_download_url("Newtonsoft.Json")
        print(f"\nDownload URL: {download_url}")
    except Exception as e:
        logger.error(f"Test failed with error: {e}", exc_info=True)
    finally:
        await client.close_session()
        logger.info("Test completed.")
        
if __name__ == "__main__":
    # To run this test script directly
    asyncio.run(_test_client()) 