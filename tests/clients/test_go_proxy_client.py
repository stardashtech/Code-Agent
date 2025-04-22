import unittest
from unittest.mock import patch, MagicMock
import requests
import os

from clients.go_proxy_client import GoProxyClient

class TestGoProxyClient(unittest.TestCase):

    @patch('clients.go_proxy_client.os.environ.get') # Mock environ access
    @patch('clients.go_proxy_client.requests.request')
    def test_get_latest_version_success(self, mock_request, mock_env_get):
        """Test getting latest Go module version successfully."""
        mock_env_get.return_value = GoProxyClient.BASE_URL # Use default proxy
        mock_response = MagicMock()
        mock_response.status_code = 200
        # Simulate .info response (JSON lines)
        mock_response.text = '{"Version":"v1.8.0","Time":"..."}\n{"Version":"v1.9.0","Time":"..."}'
        mock_request.return_value = mock_response

        client = GoProxyClient()
        # Test case encoding
        version = client.get_latest_version("github.com/gin-gonic/gin")
        self.assertEqual(version, "v1.9.0")
        mock_request.assert_called_once_with(
            "GET",
            f"{client.BASE_URL}/github.com/gin-gonic/gin/@latest.info",
            headers=client.headers,
            params=None
        )

    @patch('clients.go_proxy_client.os.environ.get') # Mock environ access
    @patch('clients.go_proxy_client.requests.request')
    def test_get_latest_version_case_encoding(self, mock_request, mock_env_get):
        """Test correct case encoding for module paths."""
        mock_env_get.return_value = GoProxyClient.BASE_URL
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = '{"Version":"v0.50.0","Time":"..."}'
        mock_request.return_value = mock_response

        client = GoProxyClient()
        client.get_latest_version("github.com/Azure/azure-sdk-for-go")
        mock_request.assert_called_once_with(
            "GET",
            # Expect Azure to be encoded as !azure
            f"{client.BASE_URL}/github.com/!azure/azure-sdk-for-go/@latest.info", 
            headers=client.headers,
            params=None
        )

    @patch('clients.go_proxy_client.os.environ.get') # Mock environ access
    @patch('clients.go_proxy_client.requests.request')
    def test_fetch_file_content_gomod_success(self, mock_request, mock_env_get):
        """Test fetching go.mod content successfully."""
        mock_env_get.return_value = GoProxyClient.BASE_URL
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "module github.com/gin-gonic/gin\n\ngo 1.18\n\nrequire (...)"
        mock_request.return_value = mock_response

        client = GoProxyClient()
        content = client.fetch_file_content("github.com/gin-gonic/gin", "go.mod", "v1.9.0")
        self.assertEqual(content, "module github.com/gin-gonic/gin\n\ngo 1.18\n\nrequire (...)")
        mock_request.assert_called_once_with(
            "GET",
            f"{client.BASE_URL}/github.com/gin-gonic/gin/@v1.9.0.mod",
            headers=client.headers,
            params=None
        )
        
    @patch('clients.go_proxy_client.os.environ.get') # Mock environ access
    def test_fetch_file_content_other_file(self, mock_env_get):
        """Test fetching non-go.mod file returns None."""
        mock_env_get.return_value = GoProxyClient.BASE_URL
        client = GoProxyClient()
        content = client.fetch_file_content("github.com/gin-gonic/gin", "main.go", "v1.9.0")
        self.assertIsNone(content)

    # Placeholder for fetch_documentation_url test
    @patch('clients.go_proxy_client.GoProxyClient.get_latest_version')
    def test_fetch_documentation_url(self, mock_get_latest):
        pass
        
    # Tests for inapplicable methods
    def test_search_repositories_not_applicable(self):
        client = GoProxyClient()
        self.assertEqual(client.search_repositories("query"), [])

    def test_find_code_examples_not_applicable(self):
        client = GoProxyClient()
        self.assertEqual(client.find_code_examples("lib"), [])
        
    def test_check_vulnerabilities_not_applicable(self):
        client = GoProxyClient()
        self.assertEqual(client.check_vulnerabilities("pkg", "v1.0.0"), [])

if __name__ == '__main__':
    unittest.main() 