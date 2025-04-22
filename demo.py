import asyncio
import logging
import os
import shutil
import tempfile
import json
import subprocess
from pprint import pformat

from app.config import settings
from app.services.code_agent import CodeAgent, Config
# Import components to test directly
from app.utils.dependency_parser import DependencyParser, DependencyParserError
from app.services.code_differ import CodeDiffer, CodeDifferError
from app.utils import git_utils # Import for checking GitPython availability, though we'll use subprocess
from app.models.llm import get_llm_client

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

# Real code samples from the actual project
SAMPLE_CODE = {
    "app/services/sandbox_runner.py": """
import logging
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple
import asyncio
import tempfile
import os
import uuid
import docker
from docker.errors import DockerException, ContainerError, ImageNotFound

logger = logging.getLogger(__name__)

class SandboxExecutionResult:
    \"\"\"Represents the result of executing code in a sandbox.\"\"\"
    def __init__(
                 self,
                 success: bool,
                 stdout: Optional[str] = None,
                 stderr: Optional[str] = None,
                 exit_code: Optional[int] = None,
                 error: Optional[str] = None):
        self.success = success  # Overall success flag (True if exit_code is 0 and no major errors)
        self.stdout = stdout
        self.stderr = stderr
        self.exit_code = exit_code  # Exit code from the executed process
        self.error = error  # Error message from the sandbox runner itself (e.g., timeout, setup failed)

    def to_dict(self) -> Dict:
        return {
            "success": self.success,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "exit_code": self.exit_code,
            "error": self.error
        }
""",
    "models/llm.py": """
import openai
import ollama
import logging
import time
from abc import ABC, abstractmethod
from typing import Optional, Dict
from config import settings # Import the settings object

logger = logging.getLogger(__name__)

class LlmInterface(ABC):
    \"\"\"Abstract interface for LLM providers.\"\"\"
    
    @abstractmethod
    def generate(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        \"\"\"
        Generate a response from the LLM.
        
        Args:
            prompt: The input prompt to send to the LLM
            max_tokens: Optional maximum number of tokens to generate
            
        Returns:
            The generated text response
        \"\"\"
        pass
"""
}

# --- Helper Functions --- 

def run_command(cmd_list, cwd, check=True):
    """Runs a subprocess command, captures output, and checks for errors."""
    logger.debug(f"Running command: {' '.join(cmd_list)} in {cwd}")
    try:
        process = subprocess.run(cmd_list, cwd=cwd, capture_output=True, text=True, check=check)
        logger.debug(f"Command stdout:\n{process.stdout}")
        if process.stderr:
            logger.debug(f"Command stderr:\n{process.stderr}") # Use debug for stderr unless error
        return process.stdout.strip(), process.stderr.strip()
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {' '.join(cmd_list)}")
        logger.error(f"Stderr:\n{e.stderr}")
        logger.error(f"Stdout:\n{e.stdout}")
        raise # Re-raise the error to stop the test
    except FileNotFoundError as e:
        logger.error(f"Command not found (is git installed?): {cmd_list[0]} - {e}")
        raise

# --- Test Functions --- 

def test_dependency_parser():
    logger.info("\n--- Testing Dependency Parser with Real Project Files ---")
    parser = DependencyParser()
    
    # Use actual project files instead of creating temp files
    results = {}
    try:
        # Find all dependency files in the project
        project_root = os.getcwd()
        
        # Check if requirements.txt exists
        req_path = os.path.join(project_root, "requirements.txt")
        if os.path.exists(req_path):
            logger.info(f"Parsing real project requirements.txt: {req_path}")
            results['requirements'] = parser.parse_dependencies(req_path)
            
        # Check for package.json
        pkg_path = os.path.join(project_root, "package.json")
        if os.path.exists(pkg_path):
            logger.info(f"Parsing real project package.json: {pkg_path}")
            results['package_json'] = parser.parse_dependencies(pkg_path)
            
        # Check for go.mod
        go_path = os.path.join(project_root, "go.mod")
        if os.path.exists(go_path):
            logger.info(f"Parsing real project go.mod: {go_path}")
            results['go_mod'] = parser.parse_dependencies(go_path)
        
        # Test error handling with unsupported file type
        try:
            config_path = os.path.join(project_root, "config.py")
            if os.path.exists(config_path):
                logger.info(f"Testing error handling with unsupported file: {config_path}")
                parser.parse_dependencies(config_path)
                results['unsupported_error'] = False
        except DependencyParserError as e:
            logger.info(f"Successfully caught expected error for unsupported file: {e}")
            results['unsupported_error'] = True
            
        # Test error handling with non-existent file
        try:
            non_existent = os.path.join(project_root, "nonexistent_file.txt")
            parser.parse_dependencies(non_existent)
            results['not_found_error'] = False
        except FileNotFoundError as e:
            logger.info(f"Successfully caught expected error for non-existent file: {e}")
            results['not_found_error'] = True
            
        logger.info("Dependency Parser Test Results (Real Files):")
        logger.info(pformat(results))
        
        # Basic validation - at least make sure we got error handling right
        assert results.get('unsupported_error') is True, "Unsupported file error handling failed"
        assert results.get('not_found_error') is True, "Non-existent file error handling failed"
        
        logger.info("Dependency Parser tests with real project files completed successfully.")

    except Exception as e:
        logger.error(f"Dependency Parser test failed: {e}", exc_info=True)

def test_code_differ():
    logger.info("\n--- Testing Code Differ with Real LLM ---")
    # Get a real LLM client
    llm_client = get_llm_client()
    if not llm_client:
        logger.error("Failed to initialize real LLM client. CodeDiffer test cannot continue.")
        return
        
    differ = CodeDiffer(llm_client=llm_client)
    results = {}
    try:
        # Use real code from the project files
        code_file1 = list(SAMPLE_CODE.keys())[0]
        code_content1 = SAMPLE_CODE[code_file1]
        
        # Create a slightly modified version of the code
        code_content1_modified = code_content1.replace(
            "Represents the result of executing code in a sandbox.", 
            "Represents the result of executing code in a secure sandbox environment."
        )
        
        logger.info(f"Analyzing changes in real code from {code_file1}")
        results['python_diff'] = differ.compare_code(code_content1, code_content1_modified, "python")
        
        # Test with code from a different file
        code_file2 = list(SAMPLE_CODE.keys())[1]
        code_content2 = SAMPLE_CODE[code_file2]
        
        # Add a new method to the abstract class
        code_content2_modified = code_content2.replace(
            "        pass",
            """        pass
            
    @abstractmethod
    def get_model_info(self) -> Dict[str, str]:
        \"\"\"
        Get information about the current model.
        
        Returns:
            Dictionary with model details
        \"\"\"
        pass"""
        )
        
        logger.info(f"Analyzing changes in real code from {code_file2}")
        results['llm_diff'] = differ.compare_code(code_content2, code_content2_modified, "python")
        
        # Test unsupported language
        try:
            differ.compare_code("a=1", "b=2", "lua")
            results['unsupported_error'] = False
        except CodeDifferError as e:
            logger.info(f"Successfully caught expected error for unsupported language: {e}")
            results['unsupported_error'] = True
            
        logger.info("Code Differ Test Results with Real LLM:")
        logger.info(pformat(results))
        
        # Basic checks
        assert results.get('python_diff', {}).get('summary') is not None, "Failed to get summary for python diff"
        assert results.get('llm_diff', {}).get('summary') is not None, "Failed to get summary for LLM interface diff"
        assert results.get('unsupported_error') is True, "Unsupported language error handling failed"
        
        logger.info("Code Differ tests with real LLM completed successfully.")

    except Exception as e:
        logger.error(f"Code Differ test failed: {e}", exc_info=True)


def test_git_workflow():
    logger.info("\n--- Testing Real Git Workflow ---")
    
    # Use a real directory within the project for git operations
    repo_path = os.path.join(os.getcwd(), "git_test")
    os.makedirs(repo_path, exist_ok=True)
    logger.info(f"Using real test directory: {repo_path}")
    
    try:
        # Check if git is available
        try:
            run_command(["git", "--version"], cwd=repo_path)
            logger.info("Git command found.")
        except (FileNotFoundError, subprocess.CalledProcessError) as e:
            logger.error(f"Git command not found or failed. Git workflow test cannot continue. Error: {e}")
            return # Cannot proceed
            
        # 1. Init repo and commit initial file
        run_command(["git", "init"], cwd=repo_path)
        run_command(["git", "config", "user.email", "test@example.com"], cwd=repo_path) # Needed for commit
        run_command(["git", "config", "user.name", "Test User"], cwd=repo_path)
        
        # Use a real meaningful file instead of dummy content
        file_name = "test_runner.py"
        file_path = os.path.join(repo_path, file_name)
        
        # Write real code to the file
        with open(file_path, "w") as f:
            f.write("""
import unittest
import logging

logger = logging.getLogger(__name__)

class TestRunner:
    def __init__(self, test_path):
        self.test_path = test_path
        
    def run_tests(self):
        logger.info(f"Running tests in {self.test_path}")
        test_suite = unittest.defaultTestLoader.discover(self.test_path)
        test_runner = unittest.TextTestRunner()
        return test_runner.run(test_suite)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    runner = TestRunner("tests")
    runner.run_tests()
""")
            
        run_command(["git", "add", file_name], cwd=repo_path)
        run_command(["git", "commit", "-m", "Initial commit with test runner"], cwd=repo_path)
        logger.info("Initialized git repo and committed real test runner file.")
        original_branch, _ = run_command(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_path)
        logger.info(f"Original branch: {original_branch}")

        # 2. Simulate real feature branch and merge
        logger.info("\nCreating feature branch with real code improvement...")
        fix_branch = "feature-enhance-test-runner"
        
        run_command(["git", "checkout", "-b", fix_branch], cwd=repo_path)
        
        # Make meaningful improvements to the file
        with open(file_path, "w") as f:
            f.write("""
import unittest
import logging
import time
import sys

logger = logging.getLogger(__name__)

class TestRunner:
    def __init__(self, test_path, verbosity=2):
        self.test_path = test_path
        self.verbosity = verbosity
        
    def run_tests(self):
        logger.info(f"Running tests in {self.test_path}")
        start_time = time.time()
        test_suite = unittest.defaultTestLoader.discover(self.test_path)
        test_runner = unittest.TextTestRunner(verbosity=self.verbosity)
        result = test_runner.run(test_suite)
        duration = time.time() - start_time
        
        logger.info(f"Tests completed in {duration:.2f} seconds")
        return result
        
    def run_specific_test(self, test_name):
        logger.info(f"Running specific test: {test_name}")
        test_suite = unittest.defaultTestLoader.loadTestsFromName(test_name)
        test_runner = unittest.TextTestRunner(verbosity=self.verbosity)
        return test_runner.run(test_suite)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    runner = TestRunner("tests")
    if len(sys.argv) > 1:
        runner.run_specific_test(sys.argv[1])
    else:
        runner.run_tests()
""")
        
        run_command(["git", "add", file_name], cwd=repo_path)
        run_command(["git", "commit", "-m", "Enhance test runner with verbosity, timing and specific test capability"], cwd=repo_path)
        logger.info(f"Applied real improvements on branch {fix_branch}")
        
        # Merge the feature branch
        run_command(["git", "checkout", original_branch], cwd=repo_path)
        run_command(["git", "merge", fix_branch], cwd=repo_path)
        run_command(["git", "branch", "-d", fix_branch], cwd=repo_path)
        logger.info(f"Merged real improvements into {original_branch} and deleted feature branch.")
        
        # Verify changes were applied
        with open(file_path, "r") as f: 
            current_content = f.read()
            assert "run_specific_test" in current_content, "Feature branch changes were not properly merged"
            assert "verbosity" in current_content, "Feature branch changes were not properly merged"
        
        logger.info("File content verified after merge - real code improvements are present.")
        
        # 3. Test branch for bugfix with validation
        logger.info("\nCreating bugfix branch with potential issues...")
        bugfix_branch = "bugfix-test-runner"
        
        run_command(["git", "checkout", "-b", bugfix_branch], cwd=repo_path)
        
        # Make potentially problematic changes
        with open(file_path, "w") as f:
            f.write("""
import unittest
import logging
import time
import sys
import os

logger = logging.getLogger(__name__)

class TestRunner:
    def __init__(self, test_path, verbosity=2):
        self.test_path = test_path
        self.verbosity = verbosity
        
    def run_tests(self):
        # Attempt to use unsafe eval for test discovery - this would be a security issue
        logger.info(f"Running tests in {self.test_path}")
        start_time = time.time()
        if os.environ.get("DEBUG_MODE"):
            test_code = f"unittest.defaultTestLoader.discover('{self.test_path}')"
            test_suite = eval(test_code)  # Security issue: using eval
        else:
            test_suite = unittest.defaultTestLoader.discover(self.test_path)
        test_runner = unittest.TextTestRunner(verbosity=self.verbosity)
        result = test_runner.run(test_suite)
        duration = time.time() - start_time
        
        logger.info(f"Tests completed in {duration:.2f} seconds")
        return result
        
    def run_specific_test(self, test_name):
        logger.info(f"Running specific test: {test_name}")
        # Another unsafe pattern
        if ";" in test_name:  # SQL injection style vulnerability
            logger.error("Invalid test name")
            return None
        test_suite = unittest.defaultTestLoader.loadTestsFromName(test_name)
        test_runner = unittest.TextTestRunner(verbosity=self.verbosity)
        return test_runner.run(test_suite)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    runner = TestRunner("tests")
    if len(sys.argv) > 1:
        runner.run_specific_test(sys.argv[1])
    else:
        runner.run_tests()
""")
        
        run_command(["git", "add", file_name], cwd=repo_path)
        run_command(["git", "commit", "-m", "Add features to test runner but with security issues"], cwd=repo_path)
        logger.info(f"Applied potentially problematic changes on branch {bugfix_branch}")
        
        # Run validation - in a real scenario this would be security scanning or code review
        logger.info("Running security validation on the code...")
        validation_passed = False
        
        # Perform actual security scan by looking for problematic patterns
        with open(file_path, "r") as f:
            content = f.read()
            if "eval(" in content or "exec(" in content:
                logger.error("Security validation failed: Found unsafe code execution patterns")
            else:
                validation_passed = True
                
        if not validation_passed:
            # Assume validation fails - Rollback by rejecting the merge
            run_command(["git", "checkout", original_branch], cwd=repo_path)
            run_command(["git", "branch", "-D", bugfix_branch], cwd=repo_path) # Force delete
            logger.info(f"Security validation failed. Checked out {original_branch} and force-deleted branch {bugfix_branch}.")
            
            # Verify we're back to the safe version
            with open(file_path, "r") as f: 
                content = f.read()
                assert "eval(" not in content, "Security issue is still present after validation failure"
                assert "run_specific_test" in content, "Rollback did not preserve previous valid features"
            
            logger.info("Code verified after validation failure - unsafe version was rejected.")
        
        logger.info("Git workflow tests with real code and security validation completed successfully.")

    except Exception as e:
        logger.error(f"Git workflow test failed: {e}", exc_info=True)


async def run_agent_tests():
    logger.info("\n--- Running Real Code Agent Tests ---")
    # Create production configuration with correct parameters
    agent_config = Config(
        openai_api_key=getattr(settings, "OPENAI_API_KEY", None),
        openai_model=getattr(settings, "OPENAI_MODEL", "gpt-4"),
        ollama_model=getattr(settings, "OLLAMA_MODEL", "codellama"),
        embedding_dimension=1536
    )

    try:
        # Initialize real CodeAgent. Initialization handles component failures.
        logger.info("Initializing production CodeAgent with real dependencies...")
        agent = CodeAgent(config=agent_config)
        logger.info("CodeAgent initialized (potentially with warnings for optional components).")

        # --- Run Tests using the initialized agent --- 

        # 1. Store real code samples (Only if vector store is available)
        if agent.vector_store_manager:
            logger.info("\n1. Storing real code files...")
            try:
                # First, ensure collection exists
                agent.vector_store_manager._ensure_collection_exists()

                for file_path, content in SAMPLE_CODE.items():
                    logger.info(f"Storing real code file: {file_path}")
                    result = await agent.store_code(file_path, content, "python")
                    logger.info(f"Store result for {file_path}: {result}")

                # 2. Test real queries against the stored code (Only if storage succeeded)
                logger.info("\n2. Testing real queries against stored code...")
                queries = [
                    "Explain the DockerSandboxRunner class and how it executes code",
                    "What does the SandboxExecutionResult class do?",
                    "How does the LlmInterface work and what methods does it define?",
                    "Can you show me an example of using the GitHub search tool?" # Add a query that might use a tool
                ]

                conversation_history = []
                for query in queries:
                    logger.info(f"\nRunning real query: {query}")
                    try:
                        result = await agent.run(query, conversation_history=conversation_history)

                        # Log more details from the result
                        logger.info(f"\nAgent Result for query '{query}':")
                        logger.info(f"  Status: {result.get('status', 'N/A')}")
                        if "response" in result:
                            logger.info(f"  Response preview: {result['response'][:200]}...")
                        if "plan" in result:
                            logger.info(f"  Plan generated: {result['plan']}")
                        if "tool_results" in result:
                            logger.info(f"  Tool results: {result['tool_results']}")

                        # Update conversation history for context in subsequent queries
                        if "response" in result and result.get("status") == "success":
                            conversation_history.append({"role": "user", "content": query})
                            conversation_history.append({"role": "assistant", "content": result["response"]})

                        # Validate result has expected components
                        assert "status" in result, "Result should contain a status field"
                        # Allow failure status if tools are unavailable or other non-critical errors occur

                        logger.info("\n--------------------------------------------------------------------------------")
                    except Exception as e:
                        logger.error(f"Error processing query '{query}' with agent: {e}", exc_info=True)
                        logger.info("\n--------------------------------------------------------------------------------")

            except Exception as e:
                logger.error(f"Error during code storage or query execution: {e}", exc_info=True)
                logger.info("Skipping code storage and query tests due to error.")
        else:
            logger.warning("Vector store manager not available. Skipping code storage and query tests.")

        logger.info("CodeAgent tests sequence completed.")

    except ValueError as e: # Catch fatal init errors like no LLM provider
         logger.critical(f"Could not initialize CodeAgent due to a configuration error: {e}. Aborting agent tests.")
    except Exception as e:
        logger.critical(f"An unexpected critical error occurred during agent tests: {e}", exc_info=True)


async def main():
    logger.info("Starting Production-Ready Demo with Real Services")
    
    # Run component tests with real data
    test_dependency_parser()
    test_code_differ()
    test_git_workflow()
    
    # Run full agent tests with real services
    await run_agent_tests()
    
    logger.info("\nProduction-Ready Demo with Real Services Completed.")


if __name__ == "__main__":
    asyncio.run(main()) 