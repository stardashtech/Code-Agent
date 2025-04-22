import unittest
from unittest.mock import patch, MagicMock
import requests

from clients.pypi_client import PyPiClient

class TestPyPiClient(unittest.TestCase):

    @patch('clients.pypi_client.requests.request')
    def test_get_latest_version_success(self, mock_request):
        """Test getting latest version successfully."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "info": {"name": "requests", "version": "2.28.1"},
            "releases": { ... } # Omitted for brevity
        }
        mock_request.return_value = mock_response

        client = PyPiClient()
        version = client.get_latest_version("requests")

        self.assertEqual(version, "2.28.1")
        mock_request.assert_called_once_with(
            "GET",
            f"{client.BASE_URL}/requests/json",
            headers=client.headers,
            params=None
        )

    @patch('clients.pypi_client.requests.request')
    def test_get_latest_version_failure(self, mock_request):
        """Test failure when getting latest version."""
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = requests.exceptions.RequestException("API Error")
        mock_request.return_value = mock_response

        client = PyPiClient()
        version = client.get_latest_version("nonexistentpackage")
        self.assertIsNone(version)

    @patch('clients.pypi_client.requests.request')
    def test_fetch_documentation_url_success(self, mock_request):
        """Test fetching documentation URL successfully."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "info": {
                "name": "requests", 
                "version": "2.28.1",
                "home_page": "https://requests.readthedocs.io",
                "project_urls": {"Documentation": "https://requests.readthedocs.io/en/latest/"}
             }
        }
        mock_request.return_value = mock_response
        client = PyPiClient()
        url = client.fetch_documentation_url("requests")
        self.assertEqual(url, "https://requests.readthedocs.io/en/latest/") # Prefers Documentation URL

    @patch('clients.pypi_client.requests.request')
    def test_check_vulnerabilities_success(self, mock_request):
        """Test checking vulnerabilities when present."""
        mock_version_response = MagicMock()
        mock_version_response.status_code = 200
        mock_version_response.json.return_value = {
            "info": {"name": "requests", "version": "2.25.0"},
            "vulnerabilities": [
                {"id": "PYSEC-XXXX-YYYY", "details": "Example vuln", "link": "http://example.com/vuln", "aliases": ["CVE-1234-5678"], "fixed_in": ["2.25.1"]}
            ]
        }
        # Assume two calls: one for version endpoint, one for main package endpoint if version doesn't have vulns
        mock_request.return_value = mock_version_response 
        
        client = PyPiClient()
        vulns = client.check_vulnerabilities("requests", "2.25.0")
        self.assertEqual(len(vulns), 1)
        self.assertEqual(vulns[0]["id"], "PYSEC-XXXX-YYYY")
        self.assertEqual(vulns[0]["aliases"], ["CVE-1234-5678"]) 

    def test_search_repositories_not_applicable(self):
        client = PyPiClient()
        self.assertEqual(client.search_repositories("query"), [])

    def test_fetch_file_content_not_applicable(self):
        client = PyPiClient()
        self.assertIsNone(client.fetch_file_content("repo", "file"))

    def test_find_code_examples_not_applicable(self):
        client = PyPiClient()
        self.assertEqual(client.find_code_examples("lib"), [])

if __name__ == '__main__':
    unittest.main() 