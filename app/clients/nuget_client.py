import requests
import json
from typing import List, Dict, Any, Optional
import time # Added for rate limiting backoff
import random # Added for jitter

from interfaces.api_client import ExternalApiClient

# Consider adding logging

class NuGetClient(ExternalApiClient):
    """
    Client for interacting with the NuGet V3 API.
    Implements the ExternalApiClient interface for .NET packages.
    Includes basic rate limiting retry mechanism.
    See: https://learn.microsoft.com/en-us/nuget/api/overview
    """
    SERVICE_INDEX_URL = "https://api.nuget.org/v3/index.json"
    MAX_RETRIES = 3
    INITIAL_BACKOFF = 0.5 # seconds

    def __init__(self):
        self.headers = {
            "Accept": "application/json",
            "User-Agent": "CodeAgent/1.0 (Language=CSharp)"
        }
        self.service_urls = self._fetch_service_urls()
        if not self.service_urls: 
             print("Error: Failed to initialize NuGetClient due to index fetching errors.")

    def _fetch_service_urls(self) -> Dict[str, str]:
        """Fetches the necessary V3 service endpoint URLs from the index."""
        service_urls = {}
        try:
            # Add timeout to index fetching as well
            response = requests.get(self.SERVICE_INDEX_URL, headers=self.headers, timeout=10) 
            response.raise_for_status()
            index_data = response.json()
            resources = index_data.get("resources", [])
            for resource in resources:
                resource_type = resource.get("@type", "")
                resource_id = resource.get("@id")
                if resource_id:
                    # Prioritize known/preferred versions if multiple exist
                    if "SearchQueryService/3.5.0" in resource_type or (not service_urls.get('SearchQueryService') and "SearchQueryService" in resource_type):
                         service_urls['SearchQueryService'] = resource_id
                    elif "PackageBaseAddress/3.0.0" in resource_type:
                        service_urls['PackageBaseAddress'] = resource_id
                    elif "RegistrationsBaseUrl/3.6.0" in resource_type or (not service_urls.get('RegistrationsBaseUrl') and "RegistrationsBaseUrl" in resource_type):
                         service_urls['RegistrationsBaseUrl'] = resource_id
                         
            required_keys = ['SearchQueryService', 'PackageBaseAddress', 'RegistrationsBaseUrl']
            missing_keys = [key for key in required_keys if key not in service_urls]
            if missing_keys:
                 print(f"Warning: Could not retrieve required NuGet service URLs from index: {missing_keys}")
                 
        except requests.exceptions.RequestException as e:
            print(f"Failed to fetch NuGet service index: {e}")
        except json.JSONDecodeError as e:
             print(f"Failed to decode NuGet service index JSON: {e}")
        except Exception as e:
             print(f"An unexpected error occurred fetching NuGet service index: {e}")
        return service_urls

    def _make_request(self, base_url_key: str, endpoint_suffix: str = "", params: Optional[Dict] = None) -> Optional[Any]:
        """Helper method to make requests to a specific NuGet service endpoint with retry logic."""
        if base_url_key not in self.service_urls:
            print(f"Error: NuGet service URL for '{base_url_key}' not found or client not initialized properly.")
            return None
        
        base_url = self.service_urls[base_url_key]
        # Ensure base_url ends with / before appending suffix if needed
        if not base_url.endswith('/'):
            base_url += '/'
        # Ensure suffix doesn't start with / if base has it
        if endpoint_suffix.startswith('/'):
             endpoint_suffix = endpoint_suffix[1:]
             
        url = f"{base_url}{endpoint_suffix}"
        retries = 0
        while retries < self.MAX_RETRIES:
            try:
                response = requests.get(url, headers=self.headers, params=params, timeout=10) # Use GET for NuGet V3

                if response.status_code == 429:
                    print(f"NuGet API rate limit hit (429) for {url}. Attempt {retries + 1}/{self.MAX_RETRIES}")
                elif response.status_code >= 500:
                     print(f"NuGet API server error ({response.status_code}) for {url}. Attempt {retries + 1}/{self.MAX_RETRIES}")
                else:
                    response.raise_for_status() # Handles 4xx client errors (like 404 Not Found)
                    if response.status_code == 204:
                        return None
                    # Check if response is empty before trying to decode JSON
                    if not response.content:
                         return None # Treat empty response as None
                    return response.json()

            except requests.exceptions.Timeout as e:
                 print(f"NuGet API request timed out for {url}: {e}. Attempt {retries + 1}/{self.MAX_RETRIES}")
            except requests.exceptions.ConnectionError as e:
                 print(f"NuGet API connection error for {url}: {e}. Attempt {retries + 1}/{self.MAX_RETRIES}")
            except requests.exceptions.HTTPError as e:
                 status_code = e.response.status_code
                 if status_code < 500 and status_code != 429:
                      print(f"NuGet API HTTP error ({status_code}) for {url}: {e}. Not retrying.")
                      return None 
                 print(f"NuGet API HTTP error ({status_code}) occurred for {url}. Attempt {retries + 1}/{self.MAX_RETRIES}")
            except requests.exceptions.RequestException as e:
                 print(f"NuGet API request failed for {url}: {e}. Attempt {retries + 1}/{self.MAX_RETRIES}")
            except json.JSONDecodeError as e:
                 # Handle cases where response is not JSON, even if status code is OK (might happen)
                 print(f"Failed to decode JSON response from {url}: {e}. Response text: {response.text[:100]}... Not retrying.")
                 return None
            except Exception as e:
                 print(f"An unexpected error occurred during NuGet API request for {url}: {e}. Not retrying.")
                 return None

            # Calculate backoff and sleep only if a retryable condition was met
            wait_time = self.INITIAL_BACKOFF * (2 ** retries) + random.uniform(0, 0.5)
            print(f"Retrying in {wait_time:.2f} seconds...")
            time.sleep(wait_time)
            retries += 1
            
        print(f"NuGet API request failed after {self.MAX_RETRIES} retries for {url}.")
        return None

    def search_repositories(self, query: str, language: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Uses the NuGet SearchQueryService.
        """
        params = {"q": query, "prerelease": "false", "take": 10}
        result = self._make_request("SearchQueryService", params=params)
        
        packages = []
        if result and "data" in result:
            for item in result.get("data", []):
                 packages.append({
                    "id": item.get("id"),
                    "name": item.get("id"),
                    "description": item.get("description"),
                    "version": item.get("version"),
                    "homepage_url": item.get("projectUrl"),
                    "authors": item.get("authors"),
                    "tags": item.get("tags"),
                    "total_downloads": item.get("totalDownloads"),
                    "nuget_url": f"https://www.nuget.org/packages/{item.get('id')}/"
                 })
        return packages

    def get_latest_version(self, package_name: str) -> Optional[str]:
        """
        Gets the latest stable version using the PackageBaseAddress service.
        """
        endpoint_suffix = f"{package_name.lower()}/index.json"
        result = self._make_request("PackageBaseAddress", endpoint_suffix=endpoint_suffix)

        if result and "versions" in result and isinstance(result["versions"], list):
            stable_versions = [v for v in result["versions"] if '-' not in v]
            if stable_versions:
                 return stable_versions[-1]
            elif result["versions"]:
                 return result["versions"][-1]
        return None

    def fetch_file_content(self, repo_url: str, file_path: str, revision: Optional[str] = None) -> Optional[str]:
        """
        NuGet does not host source file content. Requires finding the source repository.
        """
        print("Warning: fetch_file_content requires a source repository URL, not directly applicable to NuGet API.")
        return None

    def fetch_documentation_url(self, package_name: str, version: Optional[str] = None) -> Optional[str]:
        """
        Attempts to find documentation URL from package metadata using RegistrationsBaseUrl.
        Checks 'projectUrl' field.
        Uses latest version if not specified.
        """
        latest_version = version or self.get_latest_version(package_name)
        if not latest_version:
            return None
                
        endpoint_suffix = f"{package_name.lower()}/{latest_version}.json"
        result = self._make_request("RegistrationsBaseUrl", endpoint_suffix=endpoint_suffix)
        
        if result and result.get("projectUrl"):
            return result["projectUrl"]
        
        if result and "repository" in result: 
             repo_info = result.get("repository")
             if isinstance(repo_info, dict) and repo_info.get("type") == "git":
                  repo_url = repo_info.get("url")
                  if repo_url and repo_url.startswith("http"):
                       return repo_url 

        search_results = self.search_repositories(package_name)
        if search_results:
            match = next((p for p in search_results if p['id'].lower() == package_name.lower()), None)
            if match and match.get("homepage_url"):
                 return match["homepage_url"]
            elif not match and search_results[0].get("homepage_url"):
                 return search_results[0].get("homepage_url")

        return None

    def find_code_examples(self, library_name: str, function_name: Optional[str] = None, class_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        NuGet API does not host code examples. Requires finding source repository.
        """
        print("Warning: find_code_examples requires searching a source repository, not directly applicable to NuGet API.")
        return []

    def check_vulnerabilities(self, package_name: str, version: str) -> List[Dict[str, Any]]:
        """
        Checks vulnerabilities using the registration endpoint (includes basic vulnerability info).
        """
        endpoint_suffix = f"{package_name.lower()}/{version}.json"
        result = self._make_request("RegistrationsBaseUrl", endpoint_suffix=endpoint_suffix)
        
        vulns = []
        # Check both potential keys based on observation/docs
        vuln_list = result.get("vulnerabilities") if result else None
        if vuln_list is None and result:
            vuln_list = result.get("vulnerability", []) 
        elif vuln_list is None:
             vuln_list = []
             
        if isinstance(vuln_list, list):
             for vuln_ref in vuln_list:
                 if not isinstance(vuln_ref, dict): continue # Skip invalid entries
                 # Structure: {'@id': 'reg_vuln_url', '@type': 'PackageVulnerability', 'advisoryUrl': 'gh_advisory_url', 'severity': '2'}
                 vulns.append({
                      "id": vuln_ref.get("advisoryUrl"), 
                      "source": "NuGet Gallery / GitHub Advisories", 
                      "details": f"Severity: {vuln_ref.get('severity')}. Check advisory URL for details.", 
                      "aliases": [], 
                      "fixed_in": [], 
                      "url": vuln_ref.get("advisoryUrl"),
                      "severity": vuln_ref.get("severity")
                 })
        return vulns 