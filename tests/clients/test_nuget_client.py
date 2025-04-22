import unittest
from unittest.mock import patch, MagicMock
import requests

from clients.nuget_client import NuGetClient

# Sample structure for NuGet service index
SAMPLE_INDEX = {
    "version": "3.0.0",
    "resources": [
        {"@id": "https://api.nuget.org/v3/registration5-semver1/", "@type": "RegistrationsBaseUrl/3.6.0"},
        {"@id": "https://api.nuget.org/v3/search1/", "@type": "SearchQueryService"},
        {"@id": "https://api.nuget.org/v3-flatcontainer/", "@type": "PackageBaseAddress/3.0.0"}
        # ... other resources
    ]
}

class TestNuGetClient(unittest.TestCase):

    @patch('clients.nuget_client.requests.get')
    def setUp(self, mock_get_index): # Mock index fetch during setup
        mock_index_response = MagicMock()
        mock_index_response.status_code = 200
        mock_index_response.json.return_value = SAMPLE_INDEX
        mock_get_index.return_value = mock_index_response
        self.client = NuGetClient()
        # Verify index was fetched during init
        mock_get_index.assert_called_once_with(self.client.SERVICE_INDEX_URL, headers=self.client.headers)
        # Reset mock for subsequent tests within the class if needed, or use per-test patching
        # self.mock_get = mock_get_index # Store for later use? Or patch per method

    @patch('clients.nuget_client.requests.get') # Patch requests.get again for this specific test
    def test_search_repositories_success(self, mock_get_search):
        """Test successful package search."""
        # We need to mock the call made by _make_request to the SearchQueryService
        mock_search_response = MagicMock()
        mock_search_response.status_code = 200
        mock_search_response.json.return_value = {
            "totalHits": 1,
            "data": [
                {"id": "Newtonsoft.Json", "version": "13.0.1", "description": "...", "projectUrl": "..."}
            ]
        }
        mock_get_search.return_value = mock_search_response
        
        # Need to re-initialize client or ensure the mock is correctly applied to the _make_request call
        # Re-patching setUp mock to avoid interference
        with patch('clients.nuget_client.requests.get') as mock_get_for_search:
            mock_get_for_search.return_value = mock_search_response
            # Need to handle the index call mock within this patch scope if re-initializing
            # Simplified: Assume self.client uses the correct mocked _make_request
            client = self.client # Use client initialized in setUp if mock setup is sufficient
            results = client.search_repositories("Newtonsoft.Json")

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["name"], "Newtonsoft.Json")
        expected_search_url = SAMPLE_INDEX['resources'][1]['@id'] # SearchQueryService URL
        mock_get_search.assert_called_once_with(
            expected_search_url, 
            headers=client.headers, 
            params={"q": "Newtonsoft.Json", "prerelease": "false", "take": 10}
        )        

    @patch('clients.nuget_client.requests.get')
    def test_get_latest_version_success(self, mock_get_versions):
        """Test getting latest stable version."""
        mock_version_response = MagicMock()
        mock_version_response.status_code = 200
        mock_version_response.json.return_value = {"versions": ["12.0.1", "13.0.1-beta", "13.0.1"]}
        mock_get_versions.return_value = mock_version_response

        with patch('clients.nuget_client.requests.get') as mock_get_for_version:
             mock_get_for_version.return_value = mock_version_response
             client = self.client
             version = client.get_latest_version("Newtonsoft.Json")

        self.assertEqual(version, "13.0.1")
        expected_base_url = SAMPLE_INDEX['resources'][2]['@id'] # PackageBaseAddress URL
        mock_get_versions.assert_called_once_with(
            f"{expected_base_url}newtonsoft.json/index.json", # Lowercase package ID
            headers=client.headers,
            params=None
        )

    # Placeholder tests for fetch_documentation_url, check_vulnerabilities
    @patch('clients.nuget_client.requests.get')
    def test_fetch_documentation_url(self, mock_get):
        pass
        
    @patch('clients.nuget_client.requests.get')
    def test_check_vulnerabilities(self, mock_get):
         # Mock the call to RegistrationsBaseUrl/{pkg}/{ver}.json
         pass

    # Tests for inapplicable methods
    def test_fetch_file_content_not_applicable(self):
        self.assertIsNone(self.client.fetch_file_content("repo", "file"))

    def test_find_code_examples_not_applicable(self):
        self.assertEqual(self.client.find_code_examples("lib"), [])

if __name__ == '__main__':
    unittest.main() 