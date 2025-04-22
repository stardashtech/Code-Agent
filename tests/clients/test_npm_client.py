import unittest
from unittest.mock import patch, MagicMock
import requests

from clients.npm_client import NpmClient

class TestNpmClient(unittest.TestCase):

    @patch('clients.npm_client.requests.request')
    def test_search_repositories_success(self, mock_request):
        """Test successful package search (npm uses package search)."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "objects": [
                {"package": {"name": "react", "version": "18.2.0", "description": "...", "links": {"npm": "...", "repository": "https://github.com/facebook/react"}}}
            ],
            "total": 1
        }
        mock_request.return_value = mock_response

        client = NpmClient()
        results = client.search_repositories("react")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["name"], "react")
        self.assertEqual(results[0]["repository_url"], "https://github.com/facebook/react")
        mock_request.assert_called_once_with(
            "GET", 
            f"{client.BASE_URL}/-/v1/search", 
            headers=client.headers, 
            params={"text": "react", "size": 10}
        )
        
    @patch('clients.npm_client.requests.request')
    def test_get_latest_version_success(self, mock_request):
        """Test getting latest version successfully."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "name": "react", 
            "dist-tags": {"latest": "18.2.0", "next": "18.3.0-alpha"},
            "versions": { ... } # Omitted
        }
        mock_request.return_value = mock_response
        
        client = NpmClient()
        version = client.get_latest_version("react")
        self.assertEqual(version, "18.2.0")
        # Check scoped package name encoding
        client.get_latest_version("@angular/core")
        mock_request.assert_called_with(
             "GET", 
             f"{client.BASE_URL}/%40angular%2Fcore", # %2F is url encoded /
             headers=client.headers, 
             params=None
        )

    @patch('clients.npm_client.requests.request')
    def test_fetch_documentation_url_success(self, mock_request):
        """Test fetching documentation URL (homepage)."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "name": "react", 
            "homepage": "https://reactjs.org/",
            "repository": {"type": "git", "url": "git+https://github.com/facebook/react.git"}
        }
        mock_request.return_value = mock_response
        client = NpmClient()
        url = client.fetch_documentation_url("react")
        self.assertEqual(url, "https://reactjs.org/")
        
    # Tests for inapplicable methods
    def test_fetch_file_content_not_applicable(self):
        client = NpmClient()
        self.assertIsNone(client.fetch_file_content("repo", "file"))

    def test_find_code_examples_not_applicable(self):
        client = NpmClient()
        self.assertEqual(client.find_code_examples("lib"), [])
        
    def test_check_vulnerabilities_not_applicable(self):
        client = NpmClient()
        self.assertEqual(client.check_vulnerabilities("pkg", "1.0.0"), [])

if __name__ == '__main__':
    unittest.main() 