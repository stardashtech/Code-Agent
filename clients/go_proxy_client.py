import requests
import json
from typing import List, Dict, Any, Optional
import re
import time # Added for rate limiting backoff
import random # Added for jitter
import os # Added for GOPROXY env var

from interfaces.api_client import ExternalApiClient

# Consider adding logging

class GoProxyClient(ExternalApiClient):
    """
    Client for interacting with Go Module Proxy services (like proxy.golang.org).
    Implements the ExternalApiClient interface for Go modules.
    Includes basic rate limiting retry mechanism.
    See: https://go.dev/ref/mod#goproxy-protocol
    """
    DEFAULT_BASE_URL = "https://proxy.golang.org"
    MAX_RETRIES = 3
    INITIAL_BACKOFF = 0.5 # seconds

    def __init__(self):
        self.headers = {
            "User-Agent": "CodeAgent/1.0 (Language=Go)"
        }
        proxy_url_env = os.environ.get("GOPROXY", self.DEFAULT_BASE_URL)
        self.base_url = None
        if proxy_url_env and proxy_url_env.lower() != "direct":
             urls = [url.strip() for url in proxy_url_env.split(',') if url.strip().startswith(('http', 'https'))]
             if urls:
                  self.base_url = urls[0] # Use the first valid proxy URL
             else:
                  print(f"Warning: GOPROXY environment variable ({proxy_url_env}) contains no valid HTTP/HTTPS URLs. GoProxyClient may not function.")
        else:
             print("Warning: GOPROXY is not set or is 'direct'. GoProxyClient may not function.")

    def _make_request(self, method: str, endpoint: str, params: Optional[Dict] = None, decode_json: bool = True) -> Optional[Any]:
        """Helper method to make requests to the Go Proxy with retry logic."""
        if not self.base_url:
             print("Error: Go Proxy URL is not configured or usable.")
             return None
             
        path_parts = endpoint.split('/')
        encoded_path_parts = []
        for i, part in enumerate(path_parts):
            if i > 0 and part:
                encoded_part = "".join([f"!{c.lower()}" if 'A' <= c <= 'Z' else c for c in part])
                encoded_path_parts.append(encoded_part)
            else:
                encoded_path_parts.append(part)
        encoded_endpoint = "/".join(encoded_path_parts)

        url = f"{self.base_url}{encoded_endpoint}"
        retries = 0
        while retries < self.MAX_RETRIES:
            try:
                response = requests.request(method, url, headers=self.headers, params=params, timeout=10) # Add timeout

                if response.status_code == 429:
                    print(f"Go Proxy rate limit hit (429) for {url}. Attempt {retries + 1}/{self.MAX_RETRIES}")
                elif response.status_code >= 500:
                     print(f"Go Proxy server error ({response.status_code}) for {url}. Attempt {retries + 1}/{self.MAX_RETRIES}")
                else:
                    # Go proxy might return 404 or 410 for not found/gone, treat as non-retryable client error
                    response.raise_for_status()
                    if response.status_code == 204:
                        return None
                    if decode_json:
                        if endpoint.endswith(".info"):
                            lines = response.text.strip().split('\n')
                            return json.loads(lines[-1]) if lines else None
                        else: # Assume standard JSON (though proxy doesn't really use it elsewhere)
                             return response.json()
                    else: # Handle non-JSON like .mod files
                        return response.text

            except requests.exceptions.Timeout as e:
                 print(f"Go Proxy request timed out for {url}: {e}. Attempt {retries + 1}/{self.MAX_RETRIES}")
            except requests.exceptions.ConnectionError as e:
                 print(f"Go Proxy connection error for {url}: {e}. Attempt {retries + 1}/{self.MAX_RETRIES}")
            except requests.exceptions.HTTPError as e:
                 status_code = e.response.status_code
                 if status_code < 500 and status_code != 429:
                      # Includes 404 (Not Found), 410 (Gone) which are valid responses for non-existent modules/versions
                      if status_code in [404, 410]:
                          print(f"Go Proxy returned {status_code} for {url}. Module or version likely not found.")
                      else:
                          print(f"Go Proxy HTTP error ({status_code}) for {url}: {e}. Not retrying.")
                      return None 
                 print(f"Go Proxy HTTP error ({status_code}) occurred for {url}. Attempt {retries + 1}/{self.MAX_RETRIES}")
            except requests.exceptions.RequestException as e:
                 print(f"Go Proxy request failed for {url}: {e}. Attempt {retries + 1}/{self.MAX_RETRIES}")
            except json.JSONDecodeError as e:
                 print(f"Failed to decode JSON response from {url}: {e}. Not retrying.") # Should only apply if decode_json=True
                 return None
            except Exception as e:
                 print(f"An unexpected error occurred during Go Proxy request for {url}: {e}. Not retrying.")
                 return None

            wait_time = self.INITIAL_BACKOFF * (2 ** retries) + random.uniform(0, 0.5)
            print(f"Retrying in {wait_time:.2f} seconds...")
            time.sleep(wait_time)
            retries += 1
            
        print(f"Go Proxy request failed after {self.MAX_RETRIES} retries for {url}.")
        return None

    def search_repositories(self, query: str, language: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Go proxy protocol does not have a standardized search endpoint.
        Returning empty list.
        """
        print("Warning: search_repositories is not applicable for Go Proxy Protocol.")
        return []

    def get_latest_version(self, package_name: str) -> Optional[str]:
        """
        Gets the latest version from the /<module_path>/@latest endpoint (.info file).
        """
        endpoint = f"/{package_name}/@latest.info"
        result = self._make_request("GET", endpoint, decode_json=True)
        return result.get("Version") if result else None

    def fetch_file_content(self, repo_url: str, file_path: str, revision: Optional[str] = None) -> Optional[str]:
        """
        Can fetch go.mod file content for a specific revision.
        Other files are not directly supported via proxy.
        """
        if file_path == 'go.mod' and revision:
             module_path = repo_url # Assume repo_url is the module path for go
             endpoint = f"/{module_path}/@{revision}.mod"
             return self._make_request("GET", endpoint, decode_json=False)
             
        print(f"Warning: fetch_file_content for '{file_path}' is not directly supported via Go Proxy. Can fetch go.mod if revision provided.")
        return None

    def _get_source_repo_from_mod_content(self, mod_content: str) -> Optional[str]:
        """ Parses go.mod content to find the source repository URL (heuristic). """
        match = re.search(r"^module\s+([\S]+)", mod_content, re.MULTILINE)
        if match:
            module_path = match.group(1)
            if module_path.startswith("github.com/"):
                return f"https://{module_path}"
        return None
        
    def fetch_documentation_url(self, package_name: str, version: Optional[str] = None) -> Optional[str]:
        """
        Attempts to find documentation by checking pkg.go.dev or finding the source repo.
        Uses the latest version if not specified.
        """
        latest_version = version or self.get_latest_version(package_name)
        if not latest_version:
            return None

        pkg_go_dev_url = f"https://pkg.go.dev/{package_name}@{latest_version}"

        mod_content = self.fetch_file_content(package_name, "go.mod", latest_version)
        if mod_content:
            repo_url = self._get_source_repo_from_mod_content(mod_content)
            if repo_url:
                return repo_url

        return pkg_go_dev_url

    def find_code_examples(self, library_name: str, function_name: Optional[str] = None, class_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Go proxy doesn't host code examples. Requires finding source repository.
        """
        print("Warning: find_code_examples requires searching a source repository, not directly applicable to Go Proxy.")
        return []

    def check_vulnerabilities(self, package_name: str, version: str) -> List[Dict[str, Any]]:
        """
        Placeholder: Checking Go vulnerabilities typically uses the govulncheck tool
        or integration with databases like the Go Vulnerability Database.
        """
        print(f"Warning: Vulnerability check for Go module {package_name} v{version} via proxy protocol not implemented.")
        return [] 