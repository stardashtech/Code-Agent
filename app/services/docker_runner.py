import asyncio
import logging
import os
import tempfile
from typing import Dict, Optional, Tuple

import docker
from docker.errors import APIError, ContainerError, DockerException, ImageNotFound

from .sandbox_runner import SandboxExecutionResult, SandboxRunner

logger = logging.getLogger(__name__)

DEFAULT_PYTHON_IMAGE = "python:3.10-slim"
CONTAINER_WORKDIR = "/app"
CONTAINER_SCRIPT_PATH = os.path.join(CONTAINER_WORKDIR, "script.py")

class DockerSandboxRunner(SandboxRunner):
    """Runs Python code securely within a Docker container."""

    def __init__(self,
                 python_image: str = DEFAULT_PYTHON_IMAGE,
                 network_disabled: bool = True,
                 mem_limit: str = "128m", # Example memory limit
                 cpus: float = 0.5): # Example CPU limit
        """Initializes the DockerSandboxRunner.

        Args:
            python_image: The Docker image to use for running Python code.
            network_disabled: Whether to disable networking within the container.
            mem_limit: Memory limit for the container (e.g., '128m', '1g').
            cpus: CPU limit for the container (e.g., 0.5 means half a CPU core).
        """
        try:
            self.client = docker.from_env()
            self.client.ping() # Test connection
            logger.info("Docker client initialized successfully.")
        except DockerException as e:
            logger.error(f"Failed to initialize Docker client: {e}. Is Docker running?", exc_info=True)
            # Depending on requirements, you might want to raise an error here
            # or allow the application to continue without sandbox functionality.
            self.client = None # Indicate that the runner is unusable
            raise RuntimeError(f"Docker client initialization failed: {e}") from e

        self.python_image = python_image
        self.network_disabled = network_disabled
        self.mem_limit = mem_limit
        self.cpus = cpus

        # Ensure the image exists locally or pull it
        self._ensure_image_exists()

    def _ensure_image_exists(self):
        """Checks if the Python image exists locally, pulls if not."""
        if not self.client:
            logger.warning("Cannot ensure image exists: Docker client not initialized.")
            return
        try:
            self.client.images.get(self.python_image)
            logger.info(f"Docker image '{self.python_image}' found locally.")
        except ImageNotFound:
            logger.warning(f"Docker image '{self.python_image}' not found locally. Pulling...")
            try:
                self.client.images.pull(self.python_image)
                logger.info(f"Successfully pulled image '{self.python_image}'.")
            except APIError as e:
                logger.error(f"Failed to pull image '{self.python_image}': {e}", exc_info=True)
                raise RuntimeError(f"Failed to pull required Docker image: {e}") from e
        except APIError as e:
            logger.error(f"Error checking for image '{self.python_image}': {e}", exc_info=True)
            # Handle other potential API errors during image check

    async def run_code(
        self,
        code_to_run: str,
        timeout_seconds: int = 30
    ) -> SandboxExecutionResult:
        """Runs the given code string within the Docker container.

        Uses asyncio.to_thread to run synchronous Docker operations.
        """
        if not self.client:
            return SandboxExecutionResult(success=False, error="Docker client not initialized.")

        # Use a temporary directory to securely handle the script file
        # This directory will be mounted as a volume in the container
        with tempfile.TemporaryDirectory() as temp_dir:
            host_script_path = os.path.join(temp_dir, "script.py")
            try:
                with open(host_script_path, "w", encoding="utf-8") as f:
                    f.write(code_to_run)
            except IOError as e:
                logger.error(f"Failed to write code to temporary file {host_script_path}: {e}", exc_info=True)
                return SandboxExecutionResult(success=False, error=f"Failed to create temporary script file: {e}")

            # Define volume mapping: mount the temp dir to the container's workdir
            volumes = {temp_dir: {'bind': CONTAINER_WORKDIR, 'mode': 'ro'}} # Read-only mount

            # Run the synchronous Docker execution in a separate thread
            try:
                result_dict = await asyncio.to_thread(
                    self._execute_in_container,
                    host_script_path,
                    volumes,
                    timeout_seconds
                )
                return SandboxExecutionResult(**result_dict)

            except asyncio.TimeoutError:
                # This might be redundant if _execute_in_container handles timeout internally,
                # but it's good practice for the async layer.
                logger.warning(f"Code execution timed out after {timeout_seconds} seconds (async layer).")
                return SandboxExecutionResult(success=False, exit_code=None, error="Execution timed out")
            except Exception as e:
                logger.error(f"Unexpected error during async execution of Docker runner: {e}", exc_info=True)
                return SandboxExecutionResult(success=False, error=f"Internal runner error: {e}")


    def _execute_in_container(self, script_path: str, volumes: Dict, timeout_seconds: int) -> Dict:
        """Synchronous helper method to run code in a Docker container."""
        container = None
        try:
            logger.info(f"Running code from {script_path} in Docker container ({self.python_image})...")
            container = self.client.containers.run(
                image=self.python_image,
                command=["python", CONTAINER_SCRIPT_PATH],
                volumes=volumes,
                working_dir=CONTAINER_WORKDIR,
                remove=False, # Keep container for log retrieval, remove in finally
                stderr=True,
                stdout=True,
                detach=True, # Run detached to manage timeout/waiting
                network_disabled=self.network_disabled,
                mem_limit=self.mem_limit,
                nano_cpus=int(self.cpus * 1e9) # Convert float cpus to nano_cpus
            )

            # Wait for container completion with timeout
            # container.wait() returns a dictionary like {'StatusCode': 0, 'Error': None}
            result = container.wait(timeout=timeout_seconds)
            exit_code = result.get("StatusCode")
            container_error = result.get("Error") # Potential error message from Docker daemon side

            stdout = container.logs(stdout=True, stderr=False).decode("utf-8", errors="replace")
            stderr = container.logs(stdout=False, stderr=True).decode("utf-8", errors="replace")

            logger.info(f"Container finished. Exit Code: {exit_code}")
            if stderr:
                logger.debug(f"Container stderr:\n{stderr}")
            if stdout:
                logger.debug(f"Container stdout:\n{stdout}")

            if exit_code == 0 and not container_error:
                return {
                    "success": True,
                    "stdout": stdout,
                    "stderr": stderr,
                    "exit_code": exit_code,
                    "error": None
                }
            else:
                # Handle non-zero exit code or Docker-level error
                error_message = f"Execution failed with exit code {exit_code}."
                if container_error:
                    error_message += f" Docker reported error: {container_error}"
                if stderr: # Include stderr in the error message if present
                     error_message += f"\nStderr: {stderr[:500]}{'...' if len(stderr) > 500 else ''}" # Truncate long stderr
                logger.warning(error_message)
                return {
                    "success": False,
                    "stdout": stdout,
                    "stderr": stderr,
                    "exit_code": exit_code,
                    "error": error_message # Combine exit code info and stderr
                }

        except ContainerError as e:
            # This catches errors where the command inside the container failed (non-zero exit)
            # The wait() result often provides better info, but this is a fallback.
            logger.error(f"ContainerError during code execution: {e}", exc_info=True)
            return {
                "success": False,
                "stdout": e.container.logs(stdout=True, stderr=False).decode("utf-8", errors="replace") if e.container else None,
                "stderr": e.stderr.decode("utf-8", errors="replace") if e.stderr else str(e),
                "exit_code": e.exit_status,
                "error": f"Container execution error: {e.stderr.decode('utf-8', errors='replace') if e.stderr else str(e)}"
            }
        except (APIError, DockerException) as e:
            # Catch other Docker-related errors (e.g., image issues, daemon connection)
            logger.error(f"Docker API error during code execution: {e}", exc_info=True)
            return {
                "success": False,
                "error": f"Docker API error: {e}"
            }
        except Exception as e:
             # Handle potential timeout from container.wait() or other unexpected errors
            if "Timeout" in str(e): # Check if the error is related to timeout
                 logger.warning(f"Code execution timed out after {timeout_seconds} seconds (sync layer).")
                 return {
                     "success": False,
                     "exit_code": None,
                     "error": "Execution timed out"
                 }
            else:
                 logger.error(f"Unexpected synchronous error during container execution: {e}", exc_info=True)
                 return {
                     "success": False,
                     "error": f"Unexpected container execution error: {e}"
                 }
        finally:
            if container:
                try:
                    container.remove(force=True) # Ensure container is removed
                    logger.debug(f"Removed container {container.id[:12]}.")
                except APIError as e:
                    logger.warning(f"Could not remove container {container.id[:12]}: {e}")


    async def initialize(self):
        """Ensures the Docker image is available before first use."""
        logger.info(f"Initializing sandbox runner: {self.__class__.__name__}")
        if not self.client:
            logger.error("Cannot initialize: Docker client not available.")
            return
        # Run the potentially blocking image check/pull in a thread
        await asyncio.to_thread(self._ensure_image_exists)

    async def cleanup(self):
        """Optional cleanup (e.g., prune unused containers/images)."""
        logger.info(f"Cleaning up sandbox runner: {self.__class__.__name__}")
        # Potentially add logic to prune containers/networks if needed,
        # but typically handled by `remove=True` in run.
        pass 