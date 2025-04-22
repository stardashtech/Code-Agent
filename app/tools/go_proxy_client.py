"""
Module that provides a client for interacting with Go module proxy.
"""
import logging
import aiohttp
import asyncio
import json
import re
from typing import Optional, Dict, Any, List, Tuple

logger = logging.getLogger(__name__)

class GoProxyError(Exception):
    """Exception raised for errors in GoProxyClient."""
    pass

class GoProxyClient:
    """
    Client for interacting with the Go module proxy API (proxy.golang.org).
    
    This client provides functionality to search for Go modules and
    retrieve their details and dependencies.
    """
    
    def __init__(self, proxy_url: str = "https://proxy.golang.org"):
        """
        Initialize the GoProxyClient.
        
        Args:
            proxy_url: URL of the Go proxy, defaults to proxy.golang.org
        """
        self.proxy_url = proxy_url
        self.session = None
        self._create_session()
        logger.info(f"GoProxyClient initialized with proxy URL: {proxy_url}")
        
    def _create_session(self):
        """Create an aiohttp session for making requests."""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
            
    async def close_session(self):
        """Close the aiohttp session if it exists."""
        if self.session and not self.session.closed:
            await self.session.close()
            logger.info("GoProxyClient session closed")
    
    async def _request(self, url: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Make an HTTP GET request to the specified URL.
        
        Args:
            url: The URL to request
            params: Optional query parameters
            
        Returns:
            Response data as dictionary
            
        Raises:
            GoProxyError: If the request fails
        """
        self._create_session()
        
        try:
            async with self.session.get(url, params=params) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise GoProxyError(f"Request failed with status {response.status}: {error_text}")
                
                content_type = response.headers.get('Content-Type', '')
                if 'application/json' in content_type:
                    return await response.json()
                else:
                    text = await response.text()
                    try:
                        # Try to parse as JSON anyway in case Content-Type is wrong
                        return json.loads(text)
                    except json.JSONDecodeError:
                        # Return text content as a dictionary
                        return {"content": text}
                        
        except aiohttp.ClientError as e:
            raise GoProxyError(f"Request error: {str(e)}")
        except Exception as e:
            raise GoProxyError(f"Unexpected error: {str(e)}")
    
    async def get_module_info(self, module_path: str, version: Optional[str] = None) -> Dict[str, Any]:
        """
        Retrieve information about a specific Go module.
        
        Args:
            module_path: The import path of the module (e.g., "github.com/gorilla/mux")
            version: Optional specific version, defaults to latest
            
        Returns:
            Dictionary with module information
        """
        if not version:
            # Get the latest version
            latest_url = f"{self.proxy_url}/{module_path}/@latest"
            latest_info = await self._request(latest_url)
            version = latest_info.get('Version')
            
            if not version:
                raise GoProxyError(f"Could not determine latest version for module: {module_path}")
        
        # Normalize version if needed (if it doesn't start with v)
        if not version.startswith('v'):
            version = f"v{version}"
        
        # Get the .info file for the module
        info_url = f"{self.proxy_url}/{module_path}/@v/{version}.info"
        return await self._request(info_url)
    
    async def get_module_dependencies(self, module_path: str, version: Optional[str] = None) -> List[Dict[str, str]]:
        """
        Retrieve dependencies for a specific Go module.
        
        Args:
            module_path: The import path of the module
            version: Optional specific version, defaults to latest
            
        Returns:
            List of dependencies
        """
        if not version:
            # Get the latest version
            latest_info = await self.get_module_info(module_path)
            version = latest_info.get('Version')
            
            if not version:
                raise GoProxyError(f"Could not determine latest version for module: {module_path}")
        
        # Normalize version if needed
        if not version.startswith('v'):
            version = f"v{version}"
        
        # Get the module's go.mod file
        mod_url = f"{self.proxy_url}/{module_path}/@v/{version}.mod"
        mod_data = await self._request(mod_url)
        
        if isinstance(mod_data, dict) and "content" in mod_data:
            # Parse the go.mod file content to extract dependencies
            dependencies = []
            mod_content = mod_data["content"]
            
            # Regular expression to match require statements
            require_blocks = re.findall(r'require\s+\((.*?)\)', mod_content, re.DOTALL)
            
            # Process require blocks
            for block in require_blocks:
                for line in block.strip().split('\n'):
                    line = line.strip()
                    if line and not line.startswith('//'):
                        parts = line.split()
                        if len(parts) >= 2:
                            dependencies.append({
                                "path": parts[0],
                                "version": parts[1]
                            })
            
            # Also look for single require statements
            single_requires = re.findall(r'require\s+([^\s]+)\s+([^\s]+)', mod_content)
            for path, version in single_requires:
                dependencies.append({
                    "path": path,
                    "version": version
                })
                
            return dependencies
        else:
            raise GoProxyError(f"Failed to parse go.mod file for {module_path}@{version}")
    
    async def search_modules(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for Go modules matching the query.
        
        Note: This is a simplified implementation since proxy.golang.org doesn't 
        provide a direct search API. In a real-world scenario, you might want to 
        use the Go Module Index or pkg.go.dev API for more comprehensive search.
        
        Args:
            query: Search term
            limit: Maximum number of results to return
            
        Returns:
            List of matching modules
        """
        # Since the Go Module Proxy doesn't provide a direct search API,
        # we'll implement a very simple search against a list of popular modules
        # In a real implementation, you would use a more comprehensive data source
        popular_modules = [
            "github.com/gorilla/mux",
            "github.com/gin-gonic/gin",
            "github.com/go-chi/chi",
            "github.com/stretchr/testify",
            "github.com/spf13/cobra",
            "github.com/spf13/viper",
            "github.com/pkg/errors",
            "golang.org/x/crypto",
            "golang.org/x/text",
            "golang.org/x/net",
            "github.com/prometheus/client_golang",
            "github.com/sirupsen/logrus",
            "github.com/dgrijalva/jwt-go",
            "github.com/gorilla/websocket",
            "go.uber.org/zap",
            "google.golang.org/grpc",
            "github.com/golang/protobuf",
            "github.com/go-sql-driver/mysql",
            "github.com/lib/pq",
            "github.com/jinzhu/gorm",
            "gorm.io/gorm",
        ]
        
        # Filter modules by query
        matching_modules = [
            module for module in popular_modules 
            if query.lower() in module.lower()
        ][:limit]
        
        # Collect information for each matching module
        results = []
        for module_path in matching_modules:
            try:
                module_info = await self.get_module_info(module_path)
                results.append({
                    "path": module_path,
                    "version": module_info.get("Version", ""),
                    "time": module_info.get("Time", ""),
                })
            except GoProxyError as e:
                logger.warning(f"Error fetching info for {module_path}: {e}")
        
        return results


async def _test_client():
    """Example usage of the GoProxyClient."""
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    client = GoProxyClient()
    
    try:
        # Search for modules
        print("Searching for 'mux' modules:")
        modules = await client.search_modules("mux", limit=5)
        for module in modules:
            print(f"- {module['path']} ({module['version']})")
        
        # Get module info
        print("\nGetting module info for gorilla/mux:")
        module_info = await client.get_module_info("github.com/gorilla/mux")
        print(f"Latest version: {module_info.get('Version')}")
        print(f"Published: {module_info.get('Time')}")
        
        # Get dependencies
        print("\nGetting dependencies for gorilla/mux:")
        dependencies = await client.get_module_dependencies("github.com/gorilla/mux")
        for dep in dependencies:
            print(f"- {dep['path']} ({dep['version']})")
            
    finally:
        await client.close_session()

if __name__ == "__main__":
    asyncio.run(_test_client()) 