import logging
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

class SandboxExecutionResult:
    """Represents the result of executing code in a sandbox."""
    def __init__(self,
                 success: bool,
                 stdout: Optional[str] = None,
                 stderr: Optional[str] = None,
                 exit_code: Optional[int] = None,
                 error: Optional[str] = None):
        self.success = success # Overall success flag (True if exit_code is 0 and no major errors)
        self.stdout = stdout
        self.stderr = stderr
        self.exit_code = exit_code # Exit code from the executed process
        self.error = error # Error message from the sandbox runner itself (e.g., timeout, setup failed)

    def to_dict(self) -> Dict:
        return {
            "success": self.success,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "exit_code": self.exit_code,
            "error": self.error
        }

class SandboxRunner(ABC):
    """Abstract base class for running code in a secure sandbox."""

    @abstractmethod
    async def run_code(
        self,
        code_to_run: str,
        timeout_seconds: int = 30 # Default timeout
    ) -> SandboxExecutionResult:
        """Runs the given code string within the sandbox environment.

        Args:
            code_to_run: The Python code string to execute.
            timeout_seconds: Maximum execution time in seconds.

        Returns:
            A SandboxExecutionResult object containing the execution details.
        """
        pass

    async def initialize(self):
        """Optional method for any setup needed before running code (e.g., pulling images)."""
        logger.info(f"Initializing sandbox runner: {self.__class__.__name__}")
        # Default implementation does nothing
        pass

    async def cleanup(self):
        """Optional method for cleanup after running code (e.g., stopping containers)."""
        logger.info(f"Cleaning up sandbox runner: {self.__class__.__name__}")
        # Default implementation does nothing
        pass

# Example of how a concrete implementation might look (DO NOT IMPLEMENT YET)
# class DockerSandboxRunner(SandboxRunner):
#     async def run_code(self, code_to_run: str, timeout_seconds: int = 30) -> SandboxExecutionResult:
#         # ... Docker-specific logic to:
#         # 1. Start a container from a pre-built Python image
#         # 2. Pass code_to_run to the container (e.g., via stdin or temp file)
#         # 3. Execute the Python code inside the container
#         # 4. Set resource limits and timeout
#         # 5. Capture stdout, stderr, exit code
#         # 6. Handle errors (timeout, container crash, etc.)
#         # 7. Stop/remove the container
#         # 8. Return SandboxExecutionResult
#         logger.warning("DockerSandboxRunner.run_code is not implemented yet.")
#         return SandboxExecutionResult(success=False, error="Not implemented") 