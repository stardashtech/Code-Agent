import unittest
from unittest.mock import patch, MagicMock
import requests

from utils.doc_scraper import DocumentationScraper

class TestDocumentationScraper(unittest.TestCase):

    def test_is_valid_url(self):
        scraper = DocumentationScraper()
        self.assertTrue(scraper._is_valid_url("https://example.com"))
        self.assertTrue(scraper._is_valid_url("http://example.com/path"))
        self.assertFalse(scraper._is_valid_url("ftp://example.com"))
        self.assertFalse(scraper._is_valid_url("example.com"))
        self.assertFalse(scraper._is_valid_url("invalid-url"))

    @patch('utils.doc_scraper.requests.get')
    def test_fetch_content_success_html(self, mock_get):
        """Test successful fetching of HTML content."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "<html><head></head><body><p>Hello</p></body></html>"
        mock_response.headers = {'content-type': 'text/html; charset=utf-8'}
        mock_get.return_value = mock_response

        scraper = DocumentationScraper()
        content = scraper.fetch_content("https://example.com/doc")

        self.assertEqual(content, "<html><head></head><body><p>Hello</p></body></html>")
        mock_get.assert_called_once_with(
            "https://example.com/doc", 
            headers=scraper.headers,
            timeout=10
        )

    @patch('utils.doc_scraper.requests.get')
    def test_fetch_content_success_text(self, mock_get):
        """Test successful fetching of plain text content."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "This is plain text."
        mock_response.headers = {'content-type': 'text/plain'}
        mock_get.return_value = mock_response

        scraper = DocumentationScraper()
        content = scraper.fetch_content("https://example.com/file.txt")
        self.assertEqual(content, "This is plain text.")

    @patch('utils.doc_scraper.requests.get')
    def test_fetch_content_invalid_content_type(self, mock_get):
        """Test handling of non-text/html content types."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "Binary data..."
        mock_response.headers = {'content-type': 'application/octet-stream'}
        mock_get.return_value = mock_response

        scraper = DocumentationScraper()
        content = scraper.fetch_content("https://example.com/binary")
        self.assertIsNone(content)

    @patch('utils.doc_scraper.requests.get')
    def test_fetch_content_request_exception(self, mock_get):
        """Test handling of request exceptions."""
        mock_get.side_effect = requests.exceptions.RequestException("Connection error")
        scraper = DocumentationScraper()
        content = scraper.fetch_content("https://example.com/error")
        self.assertIsNone(content)

    @patch('utils.doc_scraper.requests.get')
    def test_fetch_content_timeout(self, mock_get):
        """Test handling of timeouts."""
        mock_get.side_effect = requests.exceptions.Timeout("Timeout error")
        scraper = DocumentationScraper()
        content = scraper.fetch_content("https://example.com/timeout")
        self.assertIsNone(content)
        
    def test_fetch_content_invalid_url(self):
         """Test handling of invalid URLs passed to fetch_content."""
         scraper = DocumentationScraper()
         content = scraper.fetch_content("invalid-url-format")
         self.assertIsNone(content)

if __name__ == '__main__':
    unittest.main() 