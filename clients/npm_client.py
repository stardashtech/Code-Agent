import requests
import json
from typing import List, Dict, Any, Optional
import time # Added for rate limiting backoff
import random # Added for jitter

from interfaces.api_client import ExternalApiClient

# Consider adding logging

class NpmClient(ExternalApiClient):
    """
    Client for interacting with the npm Registry API (registry.npmjs.org).
    Implements the ExternalApiClient interface for Node.js/JS/TS packages.
    Includes basic rate limiting retry mechanism.
    """
    BASE_URL = "https://registry.npmjs.org"
    MAX_RETRIES = 3
    INITIAL_BACKOFF = 0.5 # seconds

    def __init__(self):
        # npm registry API generally doesn't require authentication for public package info
        self.headers = {
            "Accept": "application/json",
            "User-Agent": "CodeAgent/1.0 (Language=NodeJS)" # Identify client
        }

    def _make_request(self, method: str, endpoint: str, params: Optional[Dict] = None) -> Optional[Any]:
        """Helper method to make requests to the npm Registry API with retry logic."""
        url = f"{self.BASE_URL}{endpoint}"
        retries = 0
        while retries < self.MAX_RETRIES:
            try:
                response = requests.request(method, url, headers=self.headers, params=params, timeout=10) # Add timeout
                
                if response.status_code == 429:
                    print(f"npm Registry API rate limit hit (429) for {url}. Attempt {retries + 1}/{self.MAX_RETRIES}")
                elif response.status_code >= 500:
                     print(f"npm Registry API server error ({response.status_code}) for {url}. Attempt {retries + 1}/{self.MAX_RETRIES}")
                else:
                    response.raise_for_status() # Raise an exception for other bad status codes (4xx)
                    if response.status_code == 204: # No content
                        return None
                    return response.json()

            except requests.exceptions.Timeout as e:
                 print(f"npm Registry API request timed out for {url}: {e}. Attempt {retries + 1}/{self.MAX_RETRIES}")
            except requests.exceptions.ConnectionError as e:
                 print(f"npm Registry API connection error for {url}: {e}. Attempt {retries + 1}/{self.MAX_RETRIES}")
            except requests.exceptions.HTTPError as e:
                 status_code = e.response.status_code
                 if status_code < 500 and status_code != 429:
                      print(f"npm Registry API HTTP error ({status_code}) for {url}: {e}. Not retrying.")
                      return None 
                 print(f"npm Registry API HTTP error ({status_code}) occurred for {url}. Attempt {retries + 1}/{self.MAX_RETRIES}")
            except requests.exceptions.RequestException as e:
                 print(f"npm Registry API request failed for {url}: {e}. Attempt {retries + 1}/{self.MAX_RETRIES}")
            except json.JSONDecodeError as e:
                 print(f"Failed to decode JSON response from {url}: {e}. Not retrying.")
                 return None
            except Exception as e:
                 print(f"An unexpected error occurred during npm Registry API request for {url}: {e}. Not retrying.")
                 return None

            # Calculate backoff and sleep if retry is needed
            wait_time = self.INITIAL_BACKOFF * (2 ** retries) + random.uniform(0, 0.5)
            print(f"Retrying in {wait_time:.2f} seconds...")
            time.sleep(wait_time)
            retries += 1
            
        print(f"npm Registry API request failed after {self.MAX_RETRIES} retries for {url}.")
        return None

    def search_repositories(self, query: str, language: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Uses the npm search endpoint, but focuses on package info, not repositories directly.
        The results contain links to repositories if available in package metadata.
        """
        # npm search API v2: /-/v1/search
        endpoint = "/-/v1/search"
        params = {"text": query, "size": 10} # Limit results for now
        # Language filter isn't directly supported in the same way as GitHub search
        
        result = self._make_request("GET", endpoint, params=params)
        packages = []
        if result and "objects" in result:
            for item in result.get("objects", []):
                pkg_info = item.get("package", {})
                # Try to extract relevant info similar to GitHub search result structure
                packages.append({
                    "id": pkg_info.get("name"), # Use package name as ID
                    "name": pkg_info.get("name"),
                    "description": pkg_info.get("description"),
                    "version": pkg_info.get("version"),
                    "homepage": pkg_info.get("links", {}).get("homepage"),
                    "repository_url": pkg_info.get("links", {}).get("repository"),
                    "npm_url": pkg_info.get("links", {}).get("npm"),
                    "score": item.get("score", {}).get("final")
                })
        return packages

    def get_latest_version(self, package_name: str) -> Optional[str]:
        """
        Gets the latest version from the 'dist-tags' field.
        Handles scoped packages (e.g., @scope/package).
        """
        # Ensure package name is properly handled for URL path (scoped packages)
        endpoint = f"/{package_name.replace('/', '%2F')}" 
        result = self._make_request("GET", endpoint)
        return result.get("dist-tags", {}).get("latest") if result else None

    def fetch_file_content(self, repo_url: str, file_path: str, revision: Optional[str] = None) -> Optional[str]:
        """
        npm registry does not host file content. Requires finding the source repository.
        """
        print("Warning: fetch_file_content requires a source repository URL, not directly applicable to npm registry.")
        return None

    def fetch_documentation_url(self, package_name: str, version: Optional[str] = None) -> Optional[str]:
        """
        Attempts to find documentation URL from package metadata.
        Checks 'homepage' or repository URL.
        Version parameter is currently ignored.
        Handles scoped packages.
        """
        endpoint = f"/{package_name.replace('/', '%2F')}" 
        result = self._make_request("GET", endpoint)
        if not result:
            return None

        # Check homepage field first
        homepage = result.get("homepage")
        if homepage and isinstance(homepage, str) and homepage.startswith('http'):
            return homepage

        # Fallback to repository URL
        repo_info = result.get("repository")
        if isinstance(repo_info, dict):
            repo_url = repo_info.get("url")
            # Clean up potential git+ prefix
            if repo_url and isinstance(repo_url, str) and repo_url.startswith("git+"):
                 repo_url = repo_url[4:]
            # Convert ssh urls? For now, just return http(s)
            if repo_url and isinstance(repo_url, str) and repo_url.startswith('http'):
                 return repo_url
        elif isinstance(repo_info, str): # Sometimes it's just a string
             if repo_info.startswith("git+"):
                 repo_info = repo_info[4:]
             if repo_info.startswith('http'):
                 return repo_info

        # Final fallback to the npm package page itself
        return result.get("versions", {}).get(result.get('dist-tags',{}).get('latest'), {}).get('dist',{}).get('tarball')

    def find_code_examples(self, library_name: str, function_name: Optional[str] = None, class_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        npm registry does not host code examples directly.
        Requires finding the source repository first.
        """
        print("Warning: find_code_examples requires searching a source repository, not directly applicable to npm registry.")
        # Potential enhancement: Find repo URL via fetch_documentation_url and call GitHub client
        return []

    def check_vulnerabilities(self, package_name: str, version: str) -> List[Dict[str, Any]]:
        """
        Placeholder: Checking vulnerabilities typically requires the 'npm audit' command 
        or integration with services like Snyk/GitHub Advisory Database.
        The standard registry API doesn't expose this directly in a simple way.
        """
        # TODO: Implement by either running 'npm audit' (requires Node environment) 
        # or using an external vulnerability database API.
        print(f"Warning: Vulnerability check for npm package {package_name} v{version} via registry API not directly implemented.")
        return [] 