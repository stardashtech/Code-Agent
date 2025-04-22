import logging
from typing import Optional, Dict, Any, List
import asyncio # Added for potential async operations if sandbox is used heavily

# Assuming SandboxRunner and SandboxExecutionResult are correctly defined and importable
# Adjust the import path if necessary based on your project structure
try:
    from app.services.sandbox_runner import SandboxRunner, SandboxExecutionResult
except ImportError:
    # Define dummy classes if SandboxRunner is not available, allowing ValidationService
    # to initialize but log warnings about limited functionality.
    logger.warning("Could not import SandboxRunner. Validation requiring code execution will be disabled.")
    SandboxRunner = None
    class SandboxExecutionResult:
         def __init__(self, success: bool, stdout: Optional[str] = None, stderr: Optional[str] = None, exit_code: Optional[int] = None, error: Optional[str] = None):
            self.success = success; self.stdout = stdout; self.stderr = stderr; self.exit_code = exit_code; self.error = error


logger = logging.getLogger(__name__)

class ValidationService:
    """
    Provides services for validating code, primarily security checks
    and potentially syntax/linting checks using a sandbox environment.
    """

    def __init__(self, sandbox_runner: Optional[SandboxRunner]):
        """
        Initializes the ValidationService.

        Args:
            sandbox_runner: An instance of SandboxRunner to execute validation scripts.
                            If None, validation capabilities will be limited.
        """
        self.sandbox_runner = sandbox_runner
        if not self.sandbox_runner:
            logger.warning("ValidationService initialized without a SandboxRunner. Code execution validation disabled.")

    async def validate_code_security(self, code: str, language: str) -> Dict[str, Any]:
        """
        Performs basic security checks on the code.
        Currently checks for potentially dangerous patterns like 'eval', 'exec', os.system calls in Python.
        Can be expanded to use linters or static analysis tools in the sandbox.

        Args:
            code: The code content to validate.
            language: The programming language (currently used for logging and language-specific checks).

        Returns:
            A dictionary containing:
            - 'passed': Boolean indicating if the code passed basic security checks.
            - 'issues': A list of strings describing found issues.
        """
        logger.info(f"Performing basic security validation for {language} code...")
        issues = []
        passed = True

        # Basic pattern matching for common insecure constructs
        # TODO: Make this more robust, potentially run tools like bandit in sandbox for Python
        lang_lower = language.lower() if language else 'unknown'

        if lang_lower == 'python':
            if "eval(" in code:
                issues.append("Found potentially unsafe 'eval()' call.")
                passed = False
            if "exec(" in code:
                issues.append("Found potentially unsafe 'exec()' call.")
                passed = False
            if "os.system(" in code:
                 issues.append("Found potentially unsafe 'os.system()' call.")
                 passed = False
            # Check for subprocess calls with shell=True more carefully
            if "subprocess" in code:
                 if "shell=True" in code:
                      # This is a basic check; a more sophisticated check would parse the AST
                      # or use a dedicated linter.
                      if "subprocess.call(" in code or \
                         "subprocess.run(" in code or \
                         "subprocess.check_call(" in code or \
                         "subprocess.check_output(" in code or \
                         "subprocess.Popen(" in code:
                           issues.append("Found potentially unsafe subprocess call with shell=True.")
                           passed = False
        elif lang_lower in ['javascript', 'typescript', 'js', 'ts']:
             # Example checks for JavaScript
             if "eval(" in code:
                  issues.append("Found potentially unsafe 'eval()' call.")
                  passed = False
             if "child_process" in code and ".exec(" in code: # Basic check for child_process.exec
                  issues.append("Found potentially unsafe 'child_process.exec()' call.")
                  passed = False
             if "dangerouslySetInnerHTML" in code: # React specific example
                  issues.append("Found potentially unsafe 'dangerouslySetInnerHTML'.")
                  passed = False
        # Add checks for other languages if needed (e.g., system() in C/C++, SQL injection patterns)

        if passed:
            logger.info("Basic security validation passed.")
        else:
            logger.warning(f"Basic security validation failed. Issues: {issues}")

        result = {
            "passed": passed,
            "issues": issues
        }
        return result

    async def validate_code_syntax(self, code: str, language: str) -> Dict[str, Any]:
        """
        Validates the syntax of the code using the sandbox runner if available.

        Args:
            code: The code content to validate.
            language: The programming language (e.g., 'python', 'javascript').

        Returns:
            A dictionary containing:
            - 'valid': Boolean indicating if the syntax is valid.
            - 'error': Error message if syntax is invalid, None otherwise.
        """
        if not self.sandbox_runner:
            logger.warning("Sandbox runner not available, skipping syntax validation.")
            # Cannot definitively say it's valid, maybe return unknown or False?
            # Returning True might be misleading. Let's return False with a reason.
            return {"valid": False, "error": "Syntax check skipped: Sandbox unavailable."}

        logger.info(f"Performing syntax validation for {language} code using sandbox...")
        lang_lower = language.lower() if language else 'unknown'

        # Determine the command and image based on language
        command: Optional[List[str]] = None
        temp_filename: str = "temp_script" # Base filename
        image_name: Optional[str] = None

        if lang_lower == 'python':
            command = ["python", "-m", "py_compile", "temp_script.py"]
            temp_filename = "temp_script.py"
            image_name = "python:3.11-alpine" # Specify a reasonable Python image
        elif lang_lower in ['javascript', 'js']:
            command = ["node", "--check", "temp_script.js"]
            temp_filename = "temp_script.js"
            image_name = "node:18-alpine" # Specify a reasonable Node image
        elif lang_lower == 'java':
             command = ["javac", "TempScript.java"]
             temp_filename = "TempScript.java" # Java requires class name to match filename
             image_name = "openjdk:17-jdk-alpine"
             # Need to wrap the code in a class for Java compilation
             # Use .format() to avoid f-string issues with braces
             java_template = """class TempScript {{
    public static void main(String[] args) {{}}
    // User code below:
{}
}}"""
             code = java_template.format(code)
        elif lang_lower == 'go':
             command = ["go", "build", "-o", "/dev/null", "temp_script.go"]
             temp_filename = "temp_script.go"
             image_name = "golang:1.20-alpine"
             # Go needs package declaration
             if not code.strip().startswith("package "):
                 code = f"""package main

{code}"""
        elif lang_lower in ['c', 'cpp', 'c++']:
             compiler = "g++" if lang_lower in ['cpp', 'c++'] else "gcc"
             command = [compiler, "-fsyntax-only", "temp_script.c"]
             temp_filename = "temp_script.c" # Use .c extension generally
             image_name = "gcc:latest" # Or specific version
        # Add commands for other languages (e.g., tsc for TypeScript, ruby -c for Ruby)
        else:
            logger.warning(f"Syntax validation not currently supported for language: {language}")
            return {"valid": False, "error": f"Syntax check skipped: Language '{language}' not supported."}

        # Prepare files for sandbox
        files = {temp_filename: code}

        try:
            # Execute the syntax check command in the sandbox
            logger.debug(f"Running syntax check in sandbox. Image: {image_name}, Command: {' '.join(command)}")
            sandbox_result: Optional[SandboxExecutionResult] = await self.sandbox_runner.run_code_in_sandbox(
                code="# Syntax check execution", # Dummy code, real work is in command
                language=language, # Pass language for context if runner uses it
                command=command,
                files=files,
                image_name=image_name,
                timeout=30 # More generous timeout for compilation/checking
            )

            # Add null check for sandbox_result
            if sandbox_result is None:
                 logger.error("Sandbox execution returned None during syntax check.")
                 return {"valid": False, "error": "Sandbox execution failed to return a result."}

            if sandbox_result.success and sandbox_result.exit_code == 0:
                logger.info(f"Syntax validation successful for {language}.")
                return {"valid": True, "error": None}
            else:
                # Combine stderr and stdout for better error reporting from compilers/interpreters
                # Ensure these are strings before concatenating
                stderr_str = sandbox_result.stderr or ""
                stdout_str = sandbox_result.stdout or ""
                error_message = stderr_str + "\n" + stdout_str
                if not error_message.strip() and sandbox_result.error:
                    error_message = sandbox_result.error # Use sandbox error if stdout/stderr empty
                elif not error_message.strip():
                     error_message = f"Unknown syntax error (Exit code: {sandbox_result.exit_code})"

                logger.warning(f"Syntax validation failed for {language}. Exit code: {sandbox_result.exit_code}. Output:\n{error_message}")
                # Limit error message length
                max_error_len = 500
                if len(error_message) > max_error_len:
                     error_message = error_message[:max_error_len] + "... (truncated)"
                return {"valid": False, "error": error_message.strip()}

        except Exception as e:
            logger.error(f"Error during sandbox execution for syntax validation: {e}", exc_info=True)
            return {"valid": False, "error": f"Exception during syntax validation: {e}"}

    # Potential future methods:
    # async def validate_code_linting(self, code: str, language: str) -> Dict[str, Any]:
    #     # Run linters like flake8, eslint, etc. in sandbox
    #     pass
    # async def validate_code_style(self, code: str, language: str) -> Dict[str, Any]:
    #     # Run formatters like black, prettier --check in sandbox
    #     pass


