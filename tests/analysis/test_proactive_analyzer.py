import unittest
from unittest.mock import patch, MagicMock
import os
import tempfile
import shutil

# Adjust import paths as needed
from analysis.proactive_analyzer import ProactiveAnalyzer, ProactiveIssue
from services.version_comparer import VersionComparer

# Mock VersionComparer for testing
class MockVersionComparer:
    def __init__(self):
        self.comparison_results = {}

    def set_comparison_result(self, file_path, result):
        self.comparison_results[os.path.basename(file_path)] = result
        
    def compare_local_dependencies(self, file_path: str) -> List[Dict[str, Any]]:
        basename = os.path.basename(file_path)
        print(f"Mock compare_local_dependencies called for: {basename}") # Debug print
        # Simulate raising FileNotFoundError if necessary for testing error handling
        if basename == "nonexistent.txt":
             raise FileNotFoundError("Simulated not found")
             
        if basename == "parser_error.txt":
             # Need DependencyParserError, import or define dummy
             class DependencyParserError(Exception): pass
             from utils.dependency_parser import DependencyParserError
             raise DependencyParserError("Simulated parse error")
             
        return self.comparison_results.get(basename, [])

class TestProactiveAnalyzer(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.workspace_root = self.temp_dir.name
        
        self.mock_comparer = MockVersionComparer()
        self.analyzer = ProactiveAnalyzer(version_comparer=self.mock_comparer, 
                                        workspace_root=self.workspace_root)
        
        # Create dummy dependency files
        self.req_path = os.path.join(self.workspace_root, 'requirements.txt')
        with open(self.req_path, 'w') as f: f.write('requests==1.0.0\nold-package==0.1.0')
        
        self.pkg_json_path = os.path.join(self.workspace_root, 'subdir', 'package.json')
        os.makedirs(os.path.dirname(self.pkg_json_path)) # Create subdir
        with open(self.pkg_json_path, 'w') as f: f.write('{\"dependencies\": {\"some-lib\": \"1.2.3\"}}')
        
        # Create an excluded file
        self.venv_req_path = os.path.join(self.workspace_root, 'venv', 'requirements.txt')
        os.makedirs(os.path.dirname(self.venv_req_path))
        with open(self.venv_req_path, 'w') as f: f.write('excluded==1.0.0')

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_find_dependency_files(self):
        \"\"\"Test finding supported dependency files, excluding specific dirs.\"\"\"
        found_files = self.analyzer._find_dependency_files()
        
        # Use absolute paths for comparison if needed, or make relative
        expected_files = sorted([
            os.path.abspath(self.req_path),
            os.path.abspath(self.pkg_json_path)
            # Add .csproj and go.mod paths if created in setUp
        ])
        
        self.assertEqual(sorted([os.path.abspath(f) for f in found_files]), expected_files)
        self.assertNotIn(os.path.abspath(self.venv_req_path), [os.path.abspath(f) for f in found_files])

    def test_analyze_dependencies_success(self):
        \"\"\"Test successful analysis identifying outdated dependencies.\"\"\"
        # Setup mock comparison results
        req_results = [
            {'name': 'requests', 'local_specifier': '==1.0.0', 'local_parsed': '1.0.0', 'latest_version': '2.0.0', 'is_outdated': True, 'error': None},
            {'name': 'old-package', 'local_specifier': '==0.1.0', 'local_parsed': '0.1.0', 'latest_version': '0.1.0', 'is_outdated': False, 'error': None}
        ]
        pkg_json_results = [
            {'name': 'some-lib', 'local_specifier': '1.2.3', 'local_parsed': '1.2.3', 'latest_version': '1.3.0', 'is_outdated': True, 'error': None}
        ]
        self.mock_comparer.set_comparison_result('requirements.txt', req_results)
        self.mock_comparer.set_comparison_result('package.json', pkg_json_results)

        issues = self.analyzer.analyze_dependencies()
        
        self.assertEqual(len(issues), 2)
        
        req_issue = next((i for i in issues if i['item_name'] == 'requests'), None)
        self.assertIsNotNone(req_issue)
        self.assertEqual(req_issue['issue_type'], 'outdated_dependency')
        self.assertEqual(os.path.abspath(req_issue['file_path']), os.path.abspath(self.req_path))
        self.assertEqual(req_issue['severity'], 'medium')
        self.assertIn('Latest: \'2.0.0\'', req_issue['description'])
        self.assertEqual(req_issue['details']['latest_version'], '2.0.0')

        pkg_issue = next((i for i in issues if i['item_name'] == 'some-lib'), None)
        self.assertIsNotNone(pkg_issue)
        self.assertEqual(pkg_issue['issue_type'], 'outdated_dependency')
        self.assertEqual(os.path.abspath(pkg_issue['file_path']), os.path.abspath(self.pkg_json_path))
        self.assertEqual(pkg_issue['details']['latest_version'], '1.3.0')

    def test_analyze_dependencies_no_outdated(self):
        \"\"\"Test analysis when no dependencies are outdated.\"\"\"
        req_results = [
            {'name': 'requests', 'local_specifier': '==1.0.0', 'local_parsed': '1.0.0', 'latest_version': '1.0.0', 'is_outdated': False, 'error': None}
        ]
        self.mock_comparer.set_comparison_result('requirements.txt', req_results)
        # Assume package.json has no outdated deps either
        self.mock_comparer.set_comparison_result('package.json', []) 

        issues = self.analyzer.analyze_dependencies()
        self.assertEqual(len(issues), 0)

    def test_analyze_dependencies_comparer_error(self):
        \"\"\"Test handling errors reported by the VersionComparer.\"\"\"
        req_results = [
            {'name': 'error-lib', 'local_specifier': '1.0', 'local_parsed': '1.0', 'latest_version': None, 'is_outdated': None, 'error': 'API timeout'}
        ]
        self.mock_comparer.set_comparison_result('requirements.txt', req_results)
        self.mock_comparer.set_comparison_result('package.json', [])

        issues = self.analyzer.analyze_dependencies()
        # We expect no issues to be created, but errors logged (cannot test logs easily here)
        self.assertEqual(len(issues), 0)
        
    # Test find_files error handling?
    # Test analyze_dependencies with file not found or parser error?
    # Currently, compare_local_dependencies handles this in the mock

    def test_run_analysis_calls_sub_analyzers(self):
        \"\"\"Test that run_analysis calls different analysis methods.\"\"\"
        # Patch the individual analysis methods
        with patch.object(self.analyzer, 'analyze_dependencies', return_value=[{'issue_type': 'dep'}]) as mock_analyze_deps, \
             patch.object(self.analyzer, 'analyze_vulnerabilities', return_value=[{'issue_type': 'vuln'}]) as mock_analyze_vulns, \
             patch.object(self.analyzer, 'analyze_best_practices', return_value=[{'issue_type': 'bp'}]) as mock_analyze_bp:
            
            all_issues = self.analyzer.run_analysis()
            
            mock_analyze_deps.assert_called_once()
            mock_analyze_vulns.assert_called_once() # Will log warning as it's not implemented
            mock_analyze_bp.assert_called_once()  # Will log warning as it's not implemented
            
            # Check if results are aggregated (currently vuln/bp return empty)
            self.assertEqual(len(all_issues), 1) 
            self.assertEqual(all_issues[0]['issue_type'], 'dep')

if __name__ == '__main__':
    unittest.main() 