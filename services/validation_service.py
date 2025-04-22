import logging
import subprocess
import os
import json
import re
import shutil
from typing import List, Dict, Any, Optional, Tuple, TypedDict, Union
import shlex # For safely splitting command strings

logger = logging.getLogger(__name__)

class ValidationResult(TypedDict):
    tool_name: str
    command: str
    success: bool
    exit_code: int
    stdout: str
    stderr: str
    error_message: Optional[str] # For execution errors, not tool findings
    details: Optional[Dict[str, Any]] # For parsed tool-specific result details

class ValidationServiceError(Exception):
    """Custom exception for validation service errors."""
    pass

class ValidationService:
    """
    Runs various validation tools (linters, tests, security scanners) against 
    the codebase or specific changes.
    """

    def __init__(self, workspace_root: str = '.', config: Optional[Dict[str, Any]] = None):
        """
        Initializes the ValidationService.

        Args:
            workspace_root: The root directory where commands should be executed.
            config: Optional configuration dictionary. Might specify tools, commands,
                    or language-specific settings.
        """
        self.workspace_root = os.path.abspath(workspace_root)
        self.config = config or self._get_default_config()
        logger.info(f"ValidationService initialized for workspace: {self.workspace_root}")
        if not self.config:
             logger.warning("ValidationService config is empty. No tools configured.")
             
    def _get_default_config(self) -> Dict[str, Dict[str, Union[str, List[str]]]]:
        """
        Returns a default configuration based on commonly available tools.
        Will detect tools in the environment and provide appropriate commands.
        """
        config = {}
        
        # Python validation tools
        python_config = {}
        if shutil.which('flake8'):
            python_config['lint'] = 'flake8 .'
        elif shutil.which('pylint'):
            python_config['lint'] = 'pylint .'
            
        if shutil.which('pytest'):
            python_config['test'] = 'pytest'
        elif os.path.exists(os.path.join(self.workspace_root, 'unittest')):
            python_config['test'] = 'python -m unittest discover'
            
        if shutil.which('bandit'):
            python_config['security'] = 'bandit -r .'
        
        if python_config:
            config['python'] = python_config
            
        # JavaScript/TypeScript validation
        js_config = {}
        # Check for package.json to see if it's a Node.js project
        if os.path.exists(os.path.join(self.workspace_root, 'package.json')):
            # Read package.json to see if eslint, jest, etc. are configured
            try:
                with open(os.path.join(self.workspace_root, 'package.json'), 'r') as f:
                    pkg_data = json.load(f)
                
                scripts = pkg_data.get('scripts', {})
                dev_deps = pkg_data.get('devDependencies', {})
                deps = pkg_data.get('dependencies', {})
                
                # Check for lint script or eslint
                if 'lint' in scripts:
                    js_config['lint'] = 'npm run lint'
                elif 'eslint' in dev_deps or 'eslint' in deps:
                    if os.path.exists(os.path.join(self.workspace_root, 'node_modules', '.bin', 'eslint')):
                        js_config['lint'] = './node_modules/.bin/eslint .'
                    else:
                        js_config['lint'] = 'npx eslint .'
                        
                # Check for test script or jest/mocha
                if 'test' in scripts:
                    js_config['test'] = 'npm test'
                elif 'jest' in dev_deps or 'jest' in deps:
                    if os.path.exists(os.path.join(self.workspace_root, 'node_modules', '.bin', 'jest')):
                        js_config['test'] = './node_modules/.bin/jest'
                    else:
                        js_config['test'] = 'npx jest'
                elif 'mocha' in dev_deps or 'mocha' in deps:
                    if os.path.exists(os.path.join(self.workspace_root, 'node_modules', '.bin', 'mocha')):
                        js_config['test'] = './node_modules/.bin/mocha'
                    else:
                        js_config['test'] = 'npx mocha'
                
                # Security scanning
                if 'audit' in scripts:
                    js_config['security'] = 'npm audit'
                else:
                    js_config['security'] = 'npm audit'
                    
            except (json.JSONDecodeError, FileNotFoundError):
                logger.warning("Failed to parse package.json for JS tool configuration")
                # Fallback to basic commands
                js_config['lint'] = 'npx eslint . || echo "ESLint not configured, skipping"'
                js_config['test'] = 'npm test || echo "Tests not configured, skipping"'
        
        if js_config:
            config['javascript'] = js_config
            
        # Go validation tools
        go_config = {}
        if shutil.which('go'):
            if os.path.exists(os.path.join(self.workspace_root, 'go.mod')):
                go_config['lint'] = 'go vet ./...'
                go_config['test'] = 'go test ./...'
                go_config['security'] = 'go list -json -m all'  # Basic check for dependency info
        
        if go_config:
            config['go'] = go_config
            
        # C# validation for .NET projects
        csharp_config = {}
        if shutil.which('dotnet'):
            if any(f.endswith('.csproj') for f in os.listdir(self.workspace_root) if os.path.isfile(os.path.join(self.workspace_root, f))):
                csharp_config['build'] = 'dotnet build'
                csharp_config['test'] = 'dotnet test'
                
        if csharp_config:
            config['csharp'] = csharp_config
            
        return config

    def _run_command(self, command: str, tool_name: str) -> ValidationResult:
        """Runs a shell command safely in the workspace root."""
        result = ValidationResult(
            tool_name=tool_name,
            command=command,
            success=False,
            exit_code=-1,
            stdout="",
            stderr="",
            error_message=None,
            details=None  # Will be populated for some tools
        )
        try:
            logger.info(f"Running validation command ({tool_name}): '{command}' in {self.workspace_root}")
            # Use shlex.split for safer command parsing, especially with spaces/quotes
            # Run command using subprocess
            process = subprocess.run(
                shlex.split(command),
                cwd=self.workspace_root,
                capture_output=True,
                text=True,
                check=False, # Don't raise exception on non-zero exit code
                timeout=300 # Add a timeout (e.g., 5 minutes)
            )
            result['exit_code'] = process.returncode
            result['stdout'] = process.stdout.strip()
            result['stderr'] = process.stderr.strip()
            # Define success based on exit code (0 is typically success)
            result['success'] = process.returncode == 0
            
            # Parse tool-specific outputs
            result['details'] = self._parse_tool_output(tool_name, result['stdout'], result['stderr'], result['exit_code'])
            
            if result['success']:
                 logger.info(f"Command '{tool_name}' completed successfully (Exit Code: 0).")
                 # Log stdout/stderr only if verbose or if they contain output
                 if result['stdout']: logger.debug(f"{tool_name} STDOUT:\n{result['stdout']}")
                 if result['stderr']: logger.debug(f"{tool_name} STDERR:\n{result['stderr']}") # stderr might contain warnings even on success
            else:
                 logger.warning(f"Command '{tool_name}' failed (Exit Code: {process.returncode}).")
                 logger.warning(f"{tool_name} STDERR:\n{result['stderr']}")
                 if result['stdout']: logger.warning(f"{tool_name} STDOUT:\n{result['stdout']}")

        except FileNotFoundError:
            err_msg = f"Command not found for tool '{tool_name}': {shlex.split(command)[0]}. Is it installed and in PATH?"
            logger.error(err_msg)
            result['error_message'] = err_msg
            result['success'] = False
        except subprocess.TimeoutExpired:
             err_msg = f"Command '{tool_name}' timed out after 300 seconds."
             logger.error(err_msg)
             result['error_message'] = err_msg
             result['success'] = False
        except Exception as e:
            err_msg = f"Error executing command for tool '{tool_name}': {e}"
            logger.error(err_msg, exc_info=True)
            result['error_message'] = err_msg
            result['success'] = False
            
        return result
        
    def _parse_tool_output(self, tool_name: str, stdout: str, stderr: str, exit_code: int) -> Dict[str, Any]:
        """
        Parse tool-specific output into a structured format.
        This helps in providing more meaningful feedback about validation results.
        """
        details = {}
        tool_type = tool_name.split('_')[-1] if '_' in tool_name else ''
        
        # Parse based on tool name or type
        if 'pytest' in tool_name or (tool_type == 'test' and 'python' in tool_name.lower()):
            # Parse pytest output
            failed_match = re.search(r'(\d+) failed', stdout + stderr)
            passed_match = re.search(r'(\d+) passed', stdout + stderr)
            skipped_match = re.search(r'(\d+) skipped', stdout + stderr)
            
            details['tests_failed'] = int(failed_match.group(1)) if failed_match else 0
            details['tests_passed'] = int(passed_match.group(1)) if passed_match else 0
            details['tests_skipped'] = int(skipped_match.group(1)) if skipped_match else 0
            details['total_tests'] = details.get('tests_failed', 0) + details.get('tests_passed', 0) + details.get('tests_skipped', 0)
            
        elif 'flake8' in tool_name or 'pylint' in tool_name:
            # Parse linter output - count errors/warnings
            error_count = len(re.findall(r'^.+:\d+:\d+: [EF]\d+ ', stdout + stderr, re.MULTILINE))
            warning_count = len(re.findall(r'^.+:\d+:\d+: [CW]\d+ ', stdout + stderr, re.MULTILINE))
            
            details['error_count'] = error_count
            details['warning_count'] = warning_count
            details['total_issues'] = error_count + warning_count
            
        elif 'npm test' in tool_name or 'jest' in tool_name or 'mocha' in tool_name:
            # Parse JavaScript test output (npm test, Jest, Mocha)
            # Different test runners have different output formats
            if 'jest' in tool_name.lower() or 'jest' in stdout.lower():
                # Jest specific parsing
                failed_match = re.search(r'Tests:\s+(\d+) failed', stdout + stderr)
                passed_match = re.search(r'Tests:\s+(?:\d+ failed, )?(\d+) passed', stdout + stderr)
                
                details['tests_failed'] = int(failed_match.group(1)) if failed_match else 0
                details['tests_passed'] = int(passed_match.group(1)) if passed_match else 0
                details['total_tests'] = details.get('tests_failed', 0) + details.get('tests_passed', 0)
            
            elif 'mocha' in tool_name.lower():
                # Mocha specific parsing
                passed_match = re.search(r'(\d+) passing', stdout + stderr)
                failed_match = re.search(r'(\d+) failing', stdout + stderr)
                
                details['tests_passed'] = int(passed_match.group(1)) if passed_match else 0
                details['tests_failed'] = int(failed_match.group(1)) if failed_match else 0
                details['total_tests'] = details.get('tests_failed', 0) + details.get('tests_passed', 0)
            
            else:
                # Generic npm test parsing - look for typical test output patterns
                # This is a simplified approach; real parsing might need to be more sophisticated
                details['success'] = exit_code == 0
                details['output_summary'] = stdout[-200:] if stdout else "No output available"
        
        elif 'eslint' in tool_name:
            # Parse ESLint output
            error_match = re.search(r'(\d+) error', stdout + stderr)
            warning_match = re.search(r'(\d+) warning', stdout + stderr)
            
            details['error_count'] = int(error_match.group(1)) if error_match else 0
            details['warning_count'] = int(warning_match.group(1)) if warning_match else 0
            details['total_issues'] = details.get('error_count', 0) + details.get('warning_count', 0)
            
        elif 'go test' in tool_name:
            # Parse Go test output
            if exit_code == 0:
                details['tests_passed'] = True  # Go tests typically only report failures
                if 'PASS' in stdout:
                    pass_match = re.search(r'ok\s+\S+\s+([\d\.]+)s', stdout)
                    if pass_match:
                        details['duration_seconds'] = float(pass_match.group(1))
            else:
                details['tests_passed'] = False
                fail_match = re.search(r'FAIL\s+\S+\s+([\d\.]+)s', stdout + stderr)
                if fail_match:
                    details['duration_seconds'] = float(fail_match.group(1))
        
        elif 'dotnet test' in tool_name:
            # Parse .NET test output
            passed_match = re.search(r'Passed:\s+(\d+)', stdout + stderr)
            failed_match = re.search(r'Failed:\s+(\d+)', stdout + stderr)
            skipped_match = re.search(r'Skipped:\s+(\d+)', stdout + stderr)
            
            details['tests_passed'] = int(passed_match.group(1)) if passed_match else 0
            details['tests_failed'] = int(failed_match.group(1)) if failed_match else 0
            details['tests_skipped'] = int(skipped_match.group(1)) if skipped_match else 0
            details['total_tests'] = details.get('tests_passed', 0) + details.get('tests_failed', 0) + details.get('tests_skipped', 0)
            
        elif 'npm audit' in tool_name:
            # Parse npm audit output for security vulnerabilities
            high_match = re.search(r'(\d+) high', stdout + stderr)
            critical_match = re.search(r'(\d+) critical', stdout + stderr)
            moderate_match = re.search(r'(\d+) moderate', stdout + stderr)
            low_match = re.search(r'(\d+) low', stdout + stderr)
            
            details['high_severity'] = int(high_match.group(1)) if high_match else 0
            details['critical_severity'] = int(critical_match.group(1)) if critical_match else 0 
            details['moderate_severity'] = int(moderate_match.group(1)) if moderate_match else 0
            details['low_severity'] = int(low_match.group(1)) if low_match else 0
            details['total_vulnerabilities'] = (
                details.get('high_severity', 0) + 
                details.get('critical_severity', 0) + 
                details.get('moderate_severity', 0) + 
                details.get('low_severity', 0)
            )
        
        # Return parsed details or basic info
        if not details:
            details = {
                'success': exit_code == 0,
                'exit_code': exit_code
            }
            
        return details

    def run_configured_validations(self, language: str, validation_type: str) -> List[ValidationResult]:
        """
        Runs validations configured for a specific language and type (e.g., 'lint', 'test').

        Args:
            language: The programming language (e.g., 'python', 'javascript').
            validation_type: The type of validation ('lint', 'test', 'scan', etc.).

        Returns:
            A list of ValidationResult dictionaries, one for each command run.
        """
        results = []
        lang_config = self.config.get(language.lower(), {})
        command_to_run = lang_config.get(validation_type)
        tool_name = f"{language}_{validation_type}"
        
        if command_to_run:
            # Could potentially have multiple commands configured per type
            if isinstance(command_to_run, str):
                 results.append(self._run_command(command_to_run, tool_name))
            elif isinstance(command_to_run, list):
                 for i, cmd in enumerate(command_to_run):
                      results.append(self._run_command(cmd, f"{tool_name}_{i}"))
            else:
                 logger.warning(f"Invalid command configuration for {language}/{validation_type}: {command_to_run}")
        else:
            logger.info(f"No validation configured for language='{language}', type='{validation_type}'.")
            # Return a skipped result
            results.append(ValidationResult(
                 tool_name=tool_name, 
                 command=f"No command configured for {language}/{validation_type}", 
                 success=True, 
                 exit_code=0, 
                 stdout="Skipped", 
                 stderr="", 
                 error_message="Not configured",
                 details={"skipped": True, "reason": "No command configured"}
            ))

        return results
        
    def run_all_configured_validations(self) -> Dict[str, List[ValidationResult]]:
        """
        Runs all validations defined in the configuration for all languages.

        Returns:
            A dictionary where keys are languages and values are lists of 
            ValidationResult for that language.
        """
        all_results = {}
        logger.info("Running all configured validations...")
        for language, lang_config in self.config.items():
            logger.info(f"Running validations for language: {language}")
            lang_results = []
            for validation_type in lang_config.keys():
                lang_results.extend(self.run_configured_validations(language, validation_type))
            all_results[language] = lang_results
            
        logger.info("Finished running all configured validations.")
        return all_results
    
    def validate_changes(self, changed_files: List[str]) -> Dict[str, List[ValidationResult]]:
        """
        Run appropriate validations based on which files have changed.
        This is useful for CI/CD pipelines or pre-commit hooks.
        
        Args:
            changed_files: List of file paths that have changed
            
        Returns:
            Dictionary mapping languages to validation results
        """
        results = {}
        language_patterns = {
            'python': r'\.py$',
            'javascript': r'\.(js|jsx|ts|tsx)$',
            'go': r'\.go$',
            'csharp': r'\.(cs|csproj)$'
        }
        
        # Determine which languages were affected by changes
        affected_languages = set()
        for file_path in changed_files:
            for lang, pattern in language_patterns.items():
                if re.search(pattern, file_path):
                    affected_languages.add(lang)
                    break
        
        # Run validations only for affected languages
        for language in affected_languages:
            if language in self.config:
                lang_results = []
                for validation_type in self.config[language].keys():
                    lang_results.extend(self.run_configured_validations(language, validation_type))
                results[language] = lang_results
        
        return results
    
    def find_optimal_validations(self, 
                                 changed_files: List[str], 
                                 quick_mode: bool = False) -> Dict[str, List[Dict[str, Any]]]:
        """
        Smart function to determine the optimal validations to run based on:
        1. Which files changed
        2. How significant the changes are
        3. Available time/resources (quick_mode)
        
        Args:
            changed_files: List of file paths that have changed
            quick_mode: If True, only run essential validations (linting)
            
        Returns:
            Dictionary of validation commands that should be run
        """
        validations_to_run = {}
        
        # Group files by language
        language_files = {}
        for file_path in changed_files:
            file_ext = os.path.splitext(file_path)[1].lower()
            
            # Map extensions to languages
            if file_ext in ('.py'):
                language_files.setdefault('python', []).append(file_path)
            elif file_ext in ('.js', '.jsx', '.ts', '.tsx'):
                language_files.setdefault('javascript', []).append(file_path)
            elif file_ext in ('.go'):
                language_files.setdefault('go', []).append(file_path)
            elif file_ext in ('.cs'):
                language_files.setdefault('csharp', []).append(file_path)
        
        # For each affected language, determine validations to run
        for language, files in language_files.items():
            if language not in self.config:
                continue
                
            lang_validations = []
            
            # Always run linting in both quick and full mode
            if 'lint' in self.config[language]:
                lang_validations.append({
                    'type': 'lint',
                    'command': self.config[language]['lint'],
                    'priority': 'high'
                })
            
            # Run tests and other validations only in full mode
            if not quick_mode:
                if 'test' in self.config[language]:
                    lang_validations.append({
                        'type': 'test',
                        'command': self.config[language]['test'],
                        'priority': 'medium'
                    })
                
                # Only run security scans if files contain actual code changes
                # (not just comments or formatting)
                if 'security' in self.config[language]:
                    lang_validations.append({
                        'type': 'security',
                        'command': self.config[language]['security'],
                        'priority': 'low'
                    })
                    
            validations_to_run[language] = lang_validations
            
        return validations_to_run

# Example Usage
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    # Example config (replace with actual project commands)
    # The service now auto-detects tools, but we can still provide explicit config if needed
    sample_config = {
        'python': {
            'lint': 'flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics',
            'test': 'pytest' # Assumes pytest runs tests in the current dir or configured paths
        },
        'javascript': {
            'lint': 'npx eslint . || echo "ESLint not configured, skipping"',
            'test': 'npm test || echo "Tests not configured, skipping"'  
        },
        'go': {
             'lint': 'go vet ./...', # Basic go vet for linting
             'test': 'go test ./...' 
        }
    }
    
    # Run from the workspace root (or specify path)
    workspace = '.' 
    
    # Create service with auto-detected config
    validator = ValidationService(workspace_root=workspace)
    
    # Or create with explicit config:
    # validator = ValidationService(workspace_root=workspace, config=sample_config)

    print("\n--- Running Python Lint --- ")
    py_lint_results = validator.run_configured_validations('python', 'lint')
    for result in py_lint_results:
        print(f"Tool: {result['tool_name']}, Success: {result['success']}")
        if result['details']:
            print(f"Details: {json.dumps(result['details'], indent=2)}")

    print("\n--- Running Python Tests --- ")
    py_test_results = validator.run_configured_validations('python', 'test')
    for result in py_test_results:
        print(f"Tool: {result['tool_name']}, Success: {result['success']}")
        if result['details']:
            print(f"Details: {json.dumps(result['details'], indent=2)}")
    
    print("\n--- Running All Configured Validations --- ")
    all_res = validator.run_all_configured_validations()
    # Need custom JSON serialization if printing ValidationResult (TypedDict)
    # Simpliefied print:
    for lang, results in all_res.items():
         print(f"\n-- {lang.upper()} Results --")
         for res in results:
              status = "Success" if res['success'] else f"Failure (Code: {res['exit_code']})"
              print(f"  Tool: {res['tool_name']}, Status: {status}")
              if not res['success']:
                   print(f"    Stderr: {res['stderr'][:100]}...") # Show snippet of stderr on failure
              if res['error_message']:
                   print(f"    Execution Error: {res['error_message']}") 
              
    # Example of validating specific changes
    print("\n--- Validating Specific Changes --- ")
    # In a real scenario, these would come from git diff or similar
    sample_changed_files = ['src/main.py', 'src/app.js', 'docs/README.md']
    optimal_validations = validator.find_optimal_validations(sample_changed_files)
    print(f"Optimal validations to run: {json.dumps(optimal_validations, indent=2)}") 