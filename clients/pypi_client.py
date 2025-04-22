import requests
import json
from typing import List, Dict, Any, Optional
import time # Added for rate limiting backoff
import random # Added for jitter

from interfaces.api_client import ExternalApiClient

# Consider adding logging

class PyPiClient(ExternalApiClient):
    """
    Client for interacting with the PyPI (Python Package Index) JSON API.
    Implements the ExternalApiClient interface for Python packages.
    Includes basic rate limiting retry mechanism.
    """
    BASE_URL = "https://pypi.org/pypi"
    MAX_RETRIES = 3
    INITIAL_BACKOFF = 0.5 # seconds (PyPI might be less strict)

    def __init__(self):
        # PyPI API generally doesn't require authentication for read operations
        self.headers = {
            "Accept": "application/json",
            "User-Agent": "CodeAgent/1.0 (Language=Python)" # Good practice to identify your client
        }

    def _make_request(self, method: str, endpoint: str, params: Optional[Dict] = None) -> Optional[Any]:
        """Helper method to make requests to the PyPI API with retry logic."""
        url = f"{self.BASE_URL}{endpoint}"
        retries = 0
        while retries < self.MAX_RETRIES:
            try:
                response = requests.request(method, url, headers=self.headers, params=params, timeout=10) # Add timeout
                
                # Check for common rate limiting / server error status codes
                if response.status_code == 429: # Too Many Requests
                    print(f"PyPI API rate limit hit (429) for {url}. Attempt {retries + 1}/{self.MAX_RETRIES}")
                    # Use calculated backoff directly
                elif response.status_code >= 500: # Server errors (500, 502, 503, 504, etc.)
                     print(f"PyPI API server error ({response.status_code}) for {url}. Attempt {retries + 1}/{self.MAX_RETRIES}")
                     # Retry on server errors
                else:
                    response.raise_for_status() # Raise an exception for other bad status codes (4xx)
                    if response.status_code == 204: # No content
                        return None
                    return response.json()

            except requests.exceptions.Timeout as e:
                 print(f"PyPI API request timed out for {url}: {e}. Attempt {retries + 1}/{self.MAX_RETRIES}")
            except requests.exceptions.ConnectionError as e:
                 print(f"PyPI API connection error for {url}: {e}. Attempt {retries + 1}/{self.MAX_RETRIES}")
            except requests.exceptions.HTTPError as e:
                 # Should have been caught by specific status code checks above or raise_for_status
                 # If we reach here for a 4xx/5xx, it means raise_for_status triggered it
                 status_code = e.response.status_code
                 if status_code < 500 and status_code != 429: # Don't retry client errors other than 429
                      print(f"PyPI API HTTP error ({status_code}) for {url}: {e}. Not retrying.")
                      return None 
                 # If it's a 5xx or 429 that wasn't caught above, it will proceed to retry logic
                 print(f"PyPI API HTTP error ({status_code}) occurred for {url}. Attempt {retries + 1}/{self.MAX_RETRIES}")
            except requests.exceptions.RequestException as e:
                 print(f"PyPI API request failed for {url}: {e}. Attempt {retries + 1}/{self.MAX_RETRIES}")
            except json.JSONDecodeError as e:
                 print(f"Failed to decode JSON response from {url}: {e}. Not retrying.")
                 return None
            except Exception as e:
                 print(f"An unexpected error occurred during PyPI API request for {url}: {e}. Not retrying.")
                 return None

            # Calculate backoff and sleep if retry is needed
            wait_time = self.INITIAL_BACKOFF * (2 ** retries) + random.uniform(0, 0.5) # Exponential backoff with jitter
            print(f"Retrying in {wait_time:.2f} seconds...")
            time.sleep(wait_time)
            retries += 1
            
        print(f"PyPI API request failed after {self.MAX_RETRIES} retries for {url}.")
        return None

    def search_repositories(self, query: str, language: Optional[str] = None) -> List[Dict[str, Any]]:
        """ 
        PyPI doesn't have a direct repository search API like GitHub.
        This method is not applicable for PyPI. Returning empty list.
        """
        print("Warning: search_repositories is not applicable for PyPI.")
        return []

    def get_latest_version(self, package_name: str) -> Optional[str]:
        endpoint = f"/{package_name}/json"
        result = self._make_request("GET", endpoint)
        # The primary key 'version' usually holds the latest stable release
        return result.get("info", {}).get("version") if result else None

    def fetch_file_content(self, repo_url: str, file_path: str, revision: Optional[str] = None) -> Optional[str]:
        """
        PyPI does not host file content directly. Need to get repo URL first.
        This method is not applicable for direct PyPI calls.
        Use fetch_documentation_url or repo info to find the source repo first.
        """
        print("Warning: fetch_file_content requires a source repository URL, not directly applicable to PyPI.")
        return None

    def fetch_documentation_url(self, package_name: str, version: Optional[str] = None) -> Optional[str]:
        """
        Attempts to find documentation URL from package metadata.
        Checks 'Documentation', 'Homepage', or 'Project URLs' fields.
        Version parameter is currently ignored.
        """
        endpoint = f"/{package_name}/json"
        result = self._make_request("GET", endpoint)
        if not result:
            return None

        info = result.get("info", {})
        project_urls = info.get("project_urls", {}) or {}

        # Prioritize specific documentation URL
        doc_url = project_urls.get('Documentation') or project_urls.get('documentation')
        if doc_url:
            return doc_url

        # Fallback to Homepage
        homepage = info.get("home_page") or project_urls.get('Homepage') or project_urls.get('homepage')
        if homepage:
            return homepage
            
        # Fallback to general Project URL
        project_url = info.get("project_url")
        if project_url:
             return project_url

        # Maybe check other project_urls like 'Source', 'Repository'? 

        return None

    def find_code_examples(self, library_name: str, function_name: Optional[str] = None, class_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        PyPI does not directly host code examples.
        This method should ideally find the source repository first (e.g., via fetch_documentation_url) 
        and then delegate to a GitHub client or similar.
        Returning empty list for now.
        """
        print("Warning: find_code_examples requires searching a source repository, not directly applicable to PyPI.")
        # Potential enhancement: Find repo URL via fetch_documentation_url and call GitHub client
        return []

    def check_vulnerabilities(self, package_name: str, version: str) -> List[Dict[str, Any]]:
        """
        Checks vulnerabilities using the PyPI JSON API's 'vulnerabilities' field (if available).
        """
        endpoint = f"/{package_name}/{version}/json" # Check specific version endpoint
        result = self._make_request("GET", endpoint)
        
        if result and "vulnerabilities" in result:
             # Structure: [{'id': 'PYSEC-XXXX', 'link': '...', 'source': 'osv', 'details': '...', 'aliases': ['CVE-XXXX'], 'fixed_in': ['1.2.4']}] 
             # Standardize the output slightly
             vulns = []
             for vuln in result["vulnerabilities"]:
                 vulns.append({
                     "id": vuln.get("id"),
                     "source": vuln.get("source"),
                     "details": vuln.get("details"),
                     "aliases": vuln.get("aliases", []), 
                     "fixed_in": vuln.get("fixed_in", []), 
                     "url": vuln.get("link")
                 })
             return vulns
        # Also check the main package endpoint as a fallback, might list general vulns
        endpoint_main = f"/{package_name}/json"
        result_main = self._make_request("GET", endpoint_main)
        if result_main and "vulnerabilities" in result_main:
             # Process similarly to above
             vulns = []
             for vuln in result_main["vulnerabilities"]:
                 # Filter out vulns fixed in versions older than or equal to the requested version?
                 # This might require version comparison logic.
                 vulns.append({
                     "id": vuln.get("id"),
                     "source": vuln.get("source"),
                     "details": vuln.get("details"),
                     "aliases": vuln.get("aliases", []), 
                     "fixed_in": vuln.get("fixed_in", []), 
                     "url": vuln.get("link")
                 })
             # TODO: Filter vulns based on the requested version and fixed_in field
             return vulns

        return [] 