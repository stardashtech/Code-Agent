import requests
import os
import base64
import json
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse, urlunparse
import time # Added for rate limiting backoff
import random # Added for jitter

from interfaces.api_client import ExternalApiClient

# Consider adding logging

class GitHubApiClient(ExternalApiClient):
    """
    Client for interacting with the GitHub REST API.
    Implements the ExternalApiClient interface.
    Requires a GITHUB_TOKEN environment variable for authentication.
    Includes basic rate limiting retry mechanism.
    """
    BASE_URL = "https://api.github.com"
    MAX_RETRIES = 3
    INITIAL_BACKOFF = 1 # seconds

    def __init__(self):
        self.token = os.environ.get("GITHUB_TOKEN")
        if not self.token:
            # Consider raising a specific configuration error or logging a warning
            print("Warning: GITHUB_TOKEN environment variable not set. API requests may be rate-limited or fail.")
            self.headers = {
                "Accept": "application/vnd.github.v3+json",
                "User-Agent": "CodeAgent/1.0 (GitHubClient)" # Add User-Agent
            }
        else:
            self.headers = {
                "Accept": "application/vnd.github.v3+json",
                "Authorization": f"token {self.token}",
                "User-Agent": "CodeAgent/1.0 (GitHubClient)"
            }

    def _make_request(self, method: str, endpoint: str, params: Optional[Dict] = None, data: Optional[Dict] = None) -> Optional[Any]:
        """Helper method to make requests to the GitHub API with retry logic."""
        url = f"{self.BASE_URL}{endpoint}"
        retries = 0
        while retries < self.MAX_RETRIES:
            try:
                response = requests.request(method, url, headers=self.headers, params=params, json=data, timeout=15) # Add timeout
                
                # Check for rate limiting specifically (though raise_for_status handles 4xx/5xx)
                if response.status_code == 403 and 'X-RateLimit-Remaining' in response.headers and int(response.headers['X-RateLimit-Remaining']) == 0:
                     # Rate limit exceeded, extract reset time if possible
                     reset_time = response.headers.get('X-RateLimit-Reset')
                     wait_time = self.INITIAL_BACKOFF * (2 ** retries) + random.uniform(0, 1)
                     if reset_time:
                         try:
                              wait_time = max(wait_time, int(reset_time) - time.time() + 1) # Add buffer
                         except ValueError:
                              pass # Use calculated backoff if reset time isn't a valid timestamp
                     print(f"GitHub API rate limit exceeded. Retrying in {wait_time:.2f} seconds...")
                     time.sleep(wait_time)
                     retries += 1
                     continue # Go to next retry iteration
                     
                response.raise_for_status() # Raise an exception for other bad status codes (4xx or 5xx)
                
                if response.status_code == 204: # No content
                    return None
                return response.json()

            except requests.exceptions.Timeout as e:
                 print(f"GitHub API request timed out for {url}: {e}. Attempt {retries + 1}/{self.MAX_RETRIES}")
                 # Retry on timeout
            except requests.exceptions.ConnectionError as e:
                 print(f"GitHub API connection error for {url}: {e}. Attempt {retries + 1}/{self.MAX_RETRIES}")
                 # Retry on connection error
            except requests.exceptions.HTTPError as e:
                 # Handle specific HTTP errors for retry (e.g., 5xx server errors)
                 status_code = e.response.status_code
                 if status_code in [500, 502, 503, 504]: # Retry on server errors
                      print(f"GitHub API server error ({status_code}) for {url}. Attempt {retries + 1}/{self.MAX_RETRIES}")
                 elif status_code == 429: # Explicit Too Many Requests (though 403 with headers is more common for GitHub rate limits)
                      print(f"GitHub API returned 429 Too Many Requests for {url}. Attempt {retries + 1}/{self.MAX_RETRIES}")
                 else:
                      print(f"GitHub API HTTP error ({status_code}) for {url}: {e}. Not retrying.")
                      return None # Don't retry for other client errors (4xx) like Not Found, Unauthorized etc.
            except requests.exceptions.RequestException as e:
                 # Catch other potential request exceptions
                 print(f"GitHub API request failed for {url}: {e}. Attempt {retries + 1}/{self.MAX_RETRIES}")
            except json.JSONDecodeError as e:
                 print(f"Failed to decode JSON response from {url}: {e}. Not retrying.")
                 return None # Don't retry if response is not valid JSON
            except Exception as e:
                 print(f"An unexpected error occurred during GitHub API request for {url}: {e}. Not retrying.")
                 return None # Don't retry for unexpected errors

            # If exception occurred and is retryable, calculate backoff and sleep
            wait_time = self.INITIAL_BACKOFF * (2 ** retries) + random.uniform(0, 1) # Exponential backoff with jitter
            print(f"Retrying in {wait_time:.2f} seconds...")
            time.sleep(wait_time)
            retries += 1
            
        print(f"GitHub API request failed after {self.MAX_RETRIES} retries for {url}.")
        return None # Return None after all retries fail

    def _parse_repo_url(self, repo_url: str) -> Optional[tuple[str, str]]:
        """Parses owner/repo from various GitHub URL formats."""
        try:
            parsed = urlparse(repo_url)
            if parsed.netloc == 'github.com':
                path_parts = [part for part in parsed.path.strip('/').split('/') if part]
                if len(path_parts) >= 2:
                    owner, repo = path_parts[:2]
                    # Remove potential .git suffix
                    if repo.endswith('.git'):
                        repo = repo[:-4]
                    return owner, repo
            # Handle owner/repo format directly
            elif '/' in repo_url and not repo_url.startswith(('http', '/')):
                 path_parts = [part for part in repo_url.strip('/').split('/') if part]
                 if len(path_parts) == 2:
                     owner, repo = path_parts
                     if repo.endswith('.git'):
                         repo = repo[:-4]
                     return owner, repo
        except Exception as e:
            print(f"Error parsing repository URL '{repo_url}': {e}")
        return None

    def search_repositories(self, query: str, language: Optional[str] = None) -> List[Dict[str, Any]]:
        endpoint = "/search/repositories"
        params = {"q": query}
        if language:
            params["q"] += f" language:{language}"
        
        result = self._make_request("GET", endpoint, params=params)
        return result.get("items", []) if result else []

    def get_latest_version(self, package_name: str) -> Optional[str]:
        """
        Interprets 'latest version' for a GitHub repo as the latest tag.
        Requires package_name in 'owner/repo' format.
        """
        parsed_repo = self._parse_repo_url(package_name)
        if not parsed_repo:
            print(f"Could not parse owner/repo from '{package_name}'")
            return None
        owner, repo = parsed_repo
        
        endpoint = f"/repos/{owner}/{repo}/tags"
        params = {"per_page": 1} # Fetch only the most recent tag
        
        result = self._make_request("GET", endpoint, params=params)
        
        if result and isinstance(result, list) and len(result) > 0:
            # Tags are usually sorted newest first by default API behavior, but explicit sorting might be safer if needed.
            return result[0].get("name")
        return None

    def fetch_file_content(self, repo_url: str, file_path: str, revision: Optional[str] = None) -> Optional[str]:
        parsed_repo = self._parse_repo_url(repo_url)
        if not parsed_repo:
            print(f"Could not parse owner/repo from '{repo_url}'")
            return None
        owner, repo = parsed_repo
        
        endpoint = f"/repos/{owner}/{repo}/contents/{file_path}"
        params = {}
        if revision:
            params["ref"] = revision

        result = self._make_request("GET", endpoint, params=params)

        if result and result.get("encoding") == "base64" and "content" in result:
            try:
                # Remove newlines that might be inserted in the base64 content
                content_b64 = result["content"].replace("\n", "")
                decoded_bytes = base64.b64decode(content_b64)
                return decoded_bytes.decode('utf-8')
            except (base64.binascii.Error, UnicodeDecodeError, TypeError) as e:
                print(f"Error decoding file content from {repo_url}/{file_path}: {e}")
                return None
        elif result and 'content' not in result and result.get('type') == 'file':
             # File might be too large, requiring Git Data API usage (more complex)
             print(f"Warning: File content not directly available for {repo_url}/{file_path}. May be too large.")
             # TODO: Implement fallback using Git Data API if needed
             return None
        return None

    def fetch_documentation_url(self, package_name: str, version: Optional[str] = None) -> Optional[str]:
        """
        Attempts to find a documentation URL. 
        First checks the 'homepage' field of the repo info.
        Requires package_name in 'owner/repo' format.
        Version parameter is currently ignored for GitHub repo directly.
        """
        parsed_repo = self._parse_repo_url(package_name)
        if not parsed_repo:
            print(f"Could not parse owner/repo from '{package_name}'")
            return None
        owner, repo = parsed_repo

        endpoint = f"/repos/{owner}/{repo}"
        result = self._make_request("GET", endpoint)
        
        if result and result.get("homepage"):
            # Ensure homepage is a valid URL (basic check)
            if isinstance(result["homepage"], str) and result["homepage"].startswith('http'):
                 return result["homepage"]
        # Fallback to repo URL if no homepage specified or invalid
        if result and result.get("html_url"):
            return result["html_url"]
        return None

    def find_code_examples(self, library_name: str, function_name: Optional[str] = None, class_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Basic implementation using GitHub Code Search API.
        Searches within repositories identified by the library_name (assumed owner/repo).
        Further refinement of query and result processing needed for better examples.
        """
        parsed_repo = self._parse_repo_url(library_name) # Assuming library_name is owner/repo for now
        if not parsed_repo:
             print(f"Warning: Could not parse owner/repo from library_name '{library_name}' for code search. Searching globally.")
             query = library_name # Search globally if parsing fails
        else:
            owner, repo = parsed_repo
            query = f"repo:{owner}/{repo}"
        
        search_term = function_name or class_name or library_name # Prioritize specific terms
        query += f" \"{search_term}\""
        # Could add language qualifiers etc.

        endpoint = "/search/code"
        params = {"q": query, "per_page": 5} # Limit results for now

        result = self._make_request("GET", endpoint, params=params)
        
        examples = []
        if result and "items" in result:
            for item in result.get("items", []):
                # Basic example structure, needs refinement
                examples.append({
                    "source_url": item.get("html_url"),
                    "repository": item.get("repository", {}).get("full_name"),
                    "file_path": item.get("path"),
                    # Snippet fetching would require another API call using fetch_file_content
                    # or potentially using text_matches from the search result if sufficient
                    "snippet": "(Snippet fetching not yet implemented)", 
                    "description": f"Code match in {item.get('path')}"
                })
        return examples

    def check_vulnerabilities(self, package_name: str, version: str) -> List[Dict[str, Any]]:
        """
        Placeholder: Checking vulnerabilities directly via standard GitHub REST API is limited.
        Requires integration with GitHub Security Advisories API or external databases.
        Requires package_name in 'owner/repo' format (for potential future Security Advisory lookup).
        """
        # TODO: Implement using GitHub Security Advisories GraphQL API or other sources.
        print(f"Warning: Vulnerability check for GitHub repo {package_name} v{version} not implemented yet.")
        return [] 