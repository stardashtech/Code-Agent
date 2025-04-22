import unittest
from unittest.mock import patch, MagicMock, call
import os
import tempfile

# Adjust import paths as needed
from services.version_comparer import VersionComparer, initialize_clients # Assuming initialize_clients is testable or mockable
from utils.dependency_parser import DependencyParser, DependencyParserError
from interfaces.api_client import ExternalApiClient

# Mock API Client Implementation
class MockApiClient(ExternalApiClient):
    def __init__(self, source_type: str):
        self.source_type = source_type
        # Simulate latest versions - use package names as keys
        self.latest_versions = {
            'pypi': {'requests': '2.29.0', 'flask': '2.1.0', 'numpy': '1.23.0'},
            'npm': {'express': '4.18.0', 'lodash': '4.17.21', 'jest': '28.0.0'},
            'go': {'github.com/gin-gonic/gin': 'v1.9.0', 'example.com/other': 'v1.2.3'},
            'nuget': {'Newtonsoft.Json': '13.0.2', 'Microsoft.Extensions.Logging': '7.0.0'}
        }
        # Simulate errors for specific packages
        self.error_packages = {
             'pypi': ['error-pkg'],
             'npm': ['npm-error']
        }
        self.not_found_packages = {
             'pypi': ['not-found']
        }

    def get_latest_version(self, package_name: str) -> Optional[str]:
        if package_name in self.error_packages.get(self.source_type, []):
             raise Exception(f"Simulated API error for {package_name}")
        if package_name in self.not_found_packages.get(self.source_type, []):
             return None # Simulate package not found
        return self.latest_versions.get(self.source_type, {}).get(package_name)

    # Implement other abstract methods if necessary for the interface, even if just `pass`
    def search_repositories(self, query: str, language: Optional[str] = None) -> List[Dict[str, Any]]: pass
    def fetch_file_content(self, repo_url: str, file_path: str, revision: Optional[str] = None) -> Optional[str]: pass
    def fetch_documentation_url(self, package_name: str, version: Optional[str] = None) -> Optional[str]: pass
    def find_code_examples(self, library_name: str, function_name: Optional[str] = None, class_name: Optional[str] = None) -> List[Dict[str, Any]]: pass
    def check_vulnerabilities(self, package_name: str, version: str) -> List[Dict[str, Any]]: pass

# Mock Parsed Dependencies
MOCK_REQS_DEPS = [
    {'name': 'requests', 'version_specifier': '==2.28.1'}, # Outdated
    {'name': 'flask', 'version_specifier': '>=2.2.0'},    # Can't compare range simply, check latest > 2.2.0?
    {'name': 'numpy', 'version_specifier': 'any'},         # Cannot compare 'any'
    {'name': 'not-found', 'version_specifier': '1.0.0'},   # API returns None
    {'name': 'error-pkg', 'version_specifier': '1.0.0'}    # API raises error
]
MOCK_PKG_JSON_DEPS = [
    {'name': 'express', 'version_specifier': '^4.17.1'}, # Outdated (4.17.1 vs 4.18.0)
    {'name': 'lodash', 'version_specifier': '~4.17.21'}, # Not outdated (4.17.21 vs 4.17.21)
    {'name': 'jest', 'version_specifier': '^28.0.0'}     # Not outdated (28.0.0 vs 28.0.0)
]
MOCK_GO_MOD_DEPS = [
    {'name': 'github.com/gin-gonic/gin', 'version_specifier': 'v1.8.1'}, # Outdated
    {'name': 'example.com/other', 'version_specifier': 'v1.2.3'}        # Not outdated
]
MOCK_CSPROJ_DEPS = [
    {'name': 'Newtonsoft.Json', 'version_specifier': '13.0.1'}, # Outdated
    {'name': 'Microsoft.Extensions.Logging', 'version_specifier': '6.0.0'} # Outdated
]


class TestVersionComparer(unittest.TestCase):

    @patch('utils.dependency_parser.DependencyParser.parse_dependencies')
    def test_compare_requirements_txt(self, mock_parse):
        \"\"\"Test comparing Python requirements.txt dependencies.\"\"\"
        mock_parse.return_value = MOCK_REQS_DEPS
        mock_clients = {'pypi': MockApiClient('pypi')}
        comparer = VersionComparer(mock_clients)
        
        results = comparer.compare_local_dependencies("dummy/path/requirements.txt")
        
        mock_parse.assert_called_once_with("dummy/path/requirements.txt")
        self.assertEqual(len(results), 5)
        
        # Check specific results
        requests_res = next(r for r in results if r['name'] == 'requests')
        self.assertEqual(requests_res['local_specifier'], '==2.28.1')
        self.assertEqual(requests_res['local_parsed'], '2.28.1')
        self.assertEqual(requests_res['latest_version'], '2.29.0')
        self.assertTrue(requests_res['is_outdated'])
        self.assertIsNone(requests_res['error'])
        
        flask_res = next(r for r in results if r['name'] == 'flask')
        self.assertIsNone(flask_res['is_outdated']) # Cannot compare range easily yet
        self.assertEqual(flask_res['latest_version'], '2.1.0')
        self.assertEqual(flask_res['local_parsed'], '2.2.0') # Parser extracts version from specifier

        numpy_res = next(r for r in results if r['name'] == 'numpy')
        self.assertIsNone(numpy_res['is_outdated']) # Cannot compare 'any'
        self.assertIsNone(numpy_res['local_parsed']) 
        self.assertEqual(numpy_res['latest_version'], '1.23.0')

        notfound_res = next(r for r in results if r['name'] == 'not-found')
        self.assertIsNone(notfound_res['is_outdated'])
        self.assertIsNone(notfound_res['latest_version'])
        self.assertIn("Could not fetch latest version", notfound_res['error']) 

        error_res = next(r for r in results if r['name'] == 'error-pkg')
        self.assertIsNone(error_res['is_outdated'])
        self.assertIsNone(error_res['latest_version'])
        self.assertIn("Simulated API error", error_res['error']) 

    @patch('utils.dependency_parser.DependencyParser.parse_dependencies')
    def test_compare_package_json(self, mock_parse):
        \"\"\"Test comparing Node.js package.json dependencies.\"\"\"
        mock_parse.return_value = MOCK_PKG_JSON_DEPS
        mock_clients = {'npm': MockApiClient('npm')}
        comparer = VersionComparer(mock_clients)
        
        results = comparer.compare_local_dependencies("dummy/path/package.json")
        mock_parse.assert_called_once_with("dummy/path/package.json")
        self.assertEqual(len(results), 3)
        
        express_res = next(r for r in results if r['name'] == 'express')
        self.assertTrue(express_res['is_outdated'])
        self.assertEqual(express_res['latest_version'], '4.18.0')
        self.assertEqual(express_res['local_parsed'], '4.17.1')

        lodash_res = next(r for r in results if r['name'] == 'lodash')
        self.assertFalse(lodash_res['is_outdated'])
        self.assertEqual(lodash_res['latest_version'], '4.17.21')
        self.assertEqual(lodash_res['local_parsed'], '4.17.21')
        
        jest_res = next(r for r in results if r['name'] == 'jest')
        self.assertFalse(jest_res['is_outdated'])
        self.assertEqual(jest_res['latest_version'], '28.0.0')
        self.assertEqual(jest_res['local_parsed'], '27.0.6') # Note: ^27.0.6 parsed as 27.0.6, latest is 28.0.0 -> outdated=True?
        # Revisit _parse_version for ranges like ^ or ~. Simple parsing is limited.
        # For now, we test that comparison happens based on parsed versions.
        # Actual outdated status depends on how specifiers like ^ are handled.
        # If we assume ^27.0.6 matches 28.0.0, then it should be False. 
        # If we compare base 27.0.6 vs 28.0.0, it's True.
        # Current simple parsing yields True, which might be desired (indicates newer major version available)
        self.assertTrue(jest_res['is_outdated'])
        
    @patch('utils.dependency_parser.DependencyParser.parse_dependencies')
    def test_compare_go_mod(self, mock_parse):
        \"\"\"Test comparing Go go.mod dependencies.\"\"\"
        mock_parse.return_value = MOCK_GO_MOD_DEPS
        mock_clients = {'go': MockApiClient('go')}
        comparer = VersionComparer(mock_clients)
        results = comparer.compare_local_dependencies("dummy/path/go.mod")
        self.assertEqual(len(results), 2)
        gin_res = next(r for r in results if r['name'] == 'github.com/gin-gonic/gin')
        self.assertTrue(gin_res['is_outdated'])
        self.assertEqual(gin_res['latest_version'], 'v1.9.0')
        self.assertEqual(gin_res['local_parsed'], '1.8.1')
        other_res = next(r for r in results if r['name'] == 'example.com/other')
        self.assertFalse(other_res['is_outdated'])
        self.assertEqual(other_res['latest_version'], 'v1.2.3')
        self.assertEqual(other_res['local_parsed'], '1.2.3')

    @patch('utils.dependency_parser.DependencyParser.parse_dependencies')
    def test_compare_csproj(self, mock_parse):
        \"\"\"Test comparing C# .csproj dependencies.\"\"\"
        mock_parse.return_value = MOCK_CSPROJ_DEPS
        mock_clients = {'nuget': MockApiClient('nuget')}
        comparer = VersionComparer(mock_clients)
        results = comparer.compare_local_dependencies("dummy/path/sample.csproj")
        self.assertEqual(len(results), 2)
        newton_res = next(r for r in results if r['name'] == 'Newtonsoft.Json')
        self.assertTrue(newton_res['is_outdated'])
        self.assertEqual(newton_res['latest_version'], '13.0.2')
        self.assertEqual(newton_res['local_parsed'], '13.0.1')
        log_res = next(r for r in results if r['name'] == 'Microsoft.Extensions.Logging')
        self.assertTrue(log_res['is_outdated'])
        self.assertEqual(log_res['latest_version'], '7.0.0')
        self.assertEqual(log_res['local_parsed'], '6.0.0')

    @patch('utils.dependency_parser.DependencyParser.parse_dependencies')
    def test_parser_error(self, mock_parse):
        \"\"\"Test handling errors during dependency parsing.\"\"\"
        mock_parse.side_effect = DependencyParserError("Bad format")
        mock_clients = {'pypi': MockApiClient('pypi')}
        comparer = VersionComparer(mock_clients)
        results = comparer.compare_local_dependencies("dummy/reqs.txt")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['name'], 'error')
        self.assertIn("Error parsing", results[0]['error'])
        self.assertIn("Bad format", results[0]['error'])

    def test_unsupported_file(self):
        \"\"\"Test handling unsupported file types.\"\"\"
        mock_clients = {'pypi': MockApiClient('pypi')}
        comparer = VersionComparer(mock_clients)
        results = comparer.compare_local_dependencies("dummy/file.yaml")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['name'], 'error')
        self.assertIn("Unsupported file or no client", results[0]['error'])

    def test_missing_client(self):
        \"\"\"Test handling missing client configuration.\"\"\"
        comparer = VersionComparer({}) # No clients provided
        # Need to patch the parser to avoid FileNotFoundError if dummy file doesn't exist
        with patch('utils.dependency_parser.DependencyParser.parse_dependencies') as mock_parse:
            mock_parse.return_value = MOCK_REQS_DEPS 
            results = comparer.compare_local_dependencies("dummy/requirements.txt")
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]['name'], 'error')
            self.assertIn("Unsupported file or no client", results[0]['error'])

if __name__ == '__main__':
    unittest.main() 