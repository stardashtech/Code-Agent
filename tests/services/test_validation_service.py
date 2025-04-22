import unittest
from unittest.mock import patch, MagicMock
import os
import tempfile
import subprocess

# Adjust import path as needed
from services.validation_service import ValidationService, ValidationResult

# Example config for tests
TEST_CONFIG = {
    'python': {
        'lint': 'flake8 --version', # Use commands that exist and exit quickly
        'test': 'pytest --version'
    },
    'javascript': {
        'lint': ['eslint --version', 'prettier --check .'], # List of commands
        'test': 'npm run test --if-present' # Command that might fail gracefully
    },
    'unconfigured': {
        'test': 'some-test-command' # Language might not have lint configured
    }
}

class TestValidationService(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.workspace_root = self.temp_dir.name
        self.validator = ValidationService(workspace_root=self.workspace_root, config=TEST_CONFIG)

    def tearDown(self):
        self.temp_dir.cleanup()

    @patch('subprocess.run')
    def test_run_command_success(self, mock_subprocess_run):
        \"\"\"Test running a command that succeeds.\"\"\"
        # Mock subprocess.run result for success
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.stdout = "Success output"
        mock_process.stderr = ""
        mock_subprocess_run.return_value = mock_process

        command = "echo Hello"
        tool_name = "test_echo"
        result = self.validator._run_command(command, tool_name)

        self.assertEqual(result['tool_name'], tool_name)
        self.assertEqual(result['command'], command)
        self.assertTrue(result['success'])
        self.assertEqual(result['exit_code'], 0)
        self.assertEqual(result['stdout'], "Success output")
        self.assertEqual(result['stderr'], "")
        self.assertIsNone(result['error_message'])
        mock_subprocess_run.assert_called_once_with(
            ['echo', 'Hello'], # shlex.split result
            cwd=self.workspace_root,
            capture_output=True,
            text=True,
            check=False,
            timeout=300
        )

    @patch('subprocess.run')
    def test_run_command_failure(self, mock_subprocess_run):
        \"\"\"Test running a command that fails (non-zero exit code).\"\"\"
        mock_process = MagicMock()
        mock_process.returncode = 1
        mock_process.stdout = ""
        mock_process.stderr = "Error details"
        mock_subprocess_run.return_value = mock_process

        command = "flake8 non_existent_file.py"
        tool_name = "test_flake8_fail"
        result = self.validator._run_command(command, tool_name)

        self.assertFalse(result['success'])
        self.assertEqual(result['exit_code'], 1)
        self.assertEqual(result['stderr'], "Error details")
        self.assertIsNone(result['error_message'])
        mock_subprocess_run.assert_called_once_with(
            ['flake8', 'non_existent_file.py'], 
            cwd=self.workspace_root, capture_output=True, text=True, check=False, timeout=300
        )
        
    @patch('subprocess.run')
    def test_run_command_not_found(self, mock_subprocess_run):
         \"\"\"Test running a command where the executable is not found.\"\"\"
         mock_subprocess_run.side_effect = FileNotFoundError("Command 'badcmd' not found")
         
         command = "badcmd --arg"
         tool_name = "test_bad_cmd"
         result = self.validator._run_command(command, tool_name)
         
         self.assertFalse(result['success'])
         self.assertEqual(result['exit_code'], -1) # Default exit code on execution error
         self.assertIn("Command not found", result['error_message'])
         self.assertIn("badcmd", result['error_message'])
         mock_subprocess_run.assert_called_once_with(
             ['badcmd', '--arg'], 
             cwd=self.workspace_root, capture_output=True, text=True, check=False, timeout=300
         )
         
    @patch('subprocess.run')
    def test_run_command_timeout(self, mock_subprocess_run):
         \"\"\"Test running a command that times out.\"\"\"
         mock_subprocess_run.side_effect = subprocess.TimeoutExpired(cmd="sleep 10", timeout=5)
         
         command = "sleep 10"
         tool_name = "test_timeout"
         result = self.validator._run_command(command, tool_name)
         
         self.assertFalse(result['success'])
         self.assertEqual(result['exit_code'], -1)
         self.assertIn("timed out", result['error_message'])
         mock_subprocess_run.assert_called_once_with(
             ['sleep', '10'], 
             cwd=self.workspace_root, capture_output=True, text=True, check=False, timeout=300
         )

    @patch('services.validation_service.ValidationService._run_command')
    def test_run_configured_validations_single_command(self, mock_run_command):
        \"\"\"Test running validation for a single configured command.\"\"\"
        mock_run_command.return_value = ValidationResult(tool_name='py_lint', command='flake8 --version', success=True, exit_code=0, stdout='4.0.0', stderr='', error_message=None)
        
        results = self.validator.run_configured_validations('python', 'lint')
        
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['tool_name'], 'python_lint')
        self.assertEqual(results[0]['stdout'], '4.0.0')
        mock_run_command.assert_called_once_with(TEST_CONFIG['python']['lint'], 'python_lint')

    @patch('services.validation_service.ValidationService._run_command')
    def test_run_configured_validations_multiple_commands(self, mock_run_command):
        \"\"\"Test running validation for multiple configured commands.\"\"\"
        # Define return values for the two commands
        mock_run_command.side_effect = [
            ValidationResult(tool_name='js_lint_0', command='eslint --version', success=True, exit_code=0, stdout='8.0.0', stderr='', error_message=None),
            ValidationResult(tool_name='js_lint_1', command='prettier --check .', success=False, exit_code=1, stdout='', stderr='Code style issues found', error_message=None)
        ]
        
        results = self.validator.run_configured_validations('javascript', 'lint')
        
        self.assertEqual(len(results), 2)
        self.assertEqual(mock_run_command.call_count, 2)
        mock_run_command.assert_has_calls([
            call(TEST_CONFIG['javascript']['lint'][0], 'javascript_lint_0'),
            call(TEST_CONFIG['javascript']['lint'][1], 'javascript_lint_1')
        ])
        self.assertEqual(results[0]['stdout'], '8.0.0')
        self.assertFalse(results[1]['success'])
        self.assertEqual(results[1]['stderr'], 'Code style issues found')

    @patch('services.validation_service.ValidationService._run_command')
    def test_run_configured_validations_not_configured(self, mock_run_command):
        \"\"\"Test running validation when the type is not configured for the language.\"\"\"
        results = self.validator.run_configured_validations('python', 'scan') # 'scan' is not in config
        
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['tool_name'], 'python_scan')
        self.assertEqual(results[0]['stdout'], 'Skipped')
        self.assertEqual(results[0]['error_message'], 'Not configured')
        mock_run_command.assert_not_called() # _run_command should not be called
        
    @patch('services.validation_service.ValidationService.run_configured_validations')
    def test_run_all_configured_validations(self, mock_run_configured):
         \"\"\"Test running all validations across configured languages.\"\"\"
         # Simulate return values for each configured language/type
         def side_effect_func(lang, type):
             if lang == 'python' and type == 'lint': return [ValidationResult(tool_name='py_lint', success=True)]
             if lang == 'python' and type == 'test': return [ValidationResult(tool_name='py_test', success=True)]
             if lang == 'javascript' and type == 'lint': return [ValidationResult(tool_name='js_lint_0', success=True), ValidationResult(tool_name='js_lint_1', success=False)]
             if lang == 'javascript' and type == 'test': return [ValidationResult(tool_name='js_test', success=True)]
             if lang == 'unconfigured' and type == 'test': return [ValidationResult(tool_name='unconf_test', success=True)]
             return [ValidationResult(tool_name=f'{lang}_{type}', success=True, stdout='Skipped')] # Default for unconfigured/other types
         mock_run_configured.side_effect = side_effect_func
         
         all_results = self.validator.run_all_configured_validations()
         
         self.assertIn('python', all_results)
         self.assertIn('javascript', all_results)
         self.assertIn('unconfigured', all_results)
         self.assertEqual(len(all_results['python']), 2) # lint + test
         self.assertEqual(len(all_results['javascript']), 3) # lint (2 commands) + test
         self.assertEqual(len(all_results['unconfigured']), 1) # test only
         
         # Check that run_configured_validations was called for each defined type
         expected_calls = [
             call('python', 'lint'), call('python', 'test'),
             call('javascript', 'lint'), call('javascript', 'test'),
             call('unconfigured', 'test')
         ]
         mock_run_configured.assert_has_calls(expected_calls, any_order=True)
         self.assertEqual(mock_run_configured.call_count, len(expected_calls))

if __name__ == '__main__':
    unittest.main() 