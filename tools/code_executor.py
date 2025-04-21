import docker
import tempfile
import os
import platform
from docker.errors import DockerException
from config import settings
from agent.logger import get_logger

logger = get_logger(__name__)

class CodeExecutor:
    """
    Executes code in a container using Docker.
    Improved error handling for Docker client initialization.
    """

    def __init__(self):
        self.client = None
        try:
            # docker.from_env() uses DOCKER_HOST env var if set,
            # otherwise defaults based on OS.
            self.client = docker.from_env()
            # Verify the Docker daemon is running and responsive.
            if not self.client.ping():
                raise DockerException("Docker daemon responded to connection but is not responsive (ping failed).")
            logger.info("Docker client initialized successfully.")
            
        except DockerException as e:
            err_msg = (
                f"Failed to initialize Docker client: {e}\n" \
                f"Please ensure:\n" \
                f"  1. The Docker daemon is running.\n" \
                f"  2. You have the necessary permissions to access the Docker socket.\n"
            )
            # Provide more specific path info based on OS
            docker_host = os.getenv("DOCKER_HOST")
            if docker_host:
                 err_msg += f"  3. The DOCKER_HOST environment variable ('{docker_host}') is set correctly.\n"
            else:
                socket_path = "/var/run/docker.sock" if platform.system() != "Windows" else "npipe:////./pipe/docker_engine"
                err_msg += f"  3. The Docker socket exists and is accessible at the default location ('{socket_path}') or DOCKER_HOST is set correctly.\n"
                
            logger.error(err_msg)
            # Re-raise a more informative error or handle it appropriately
            # For now, we keep self.client as None, execute() should check this.
            # raise ConnectionError(err_msg) from e # Option: Raise a specific error
        except Exception as e:
            # Catch other unexpected errors during initialization
            logger.error(f"Unexpected error initializing Docker client: {e}")
            # raise RuntimeError(f"Unexpected Docker init error: {e}") from e

    def execute(self, code: str) -> dict:
        result = {"output": None, "error": None}
        
        if not self.client:
            result["error"] = "Code execution failed: Docker client is not initialized. Check logs for details."
            return result
            
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp:
            tmp.write(code)
            tmp_filename = tmp.name

        try:
            # We run with a mount instead of copying the file into the container
            container = self.client.containers.run(
                settings.DOCKER_IMAGE,
                command=f"python /tmp/{os.path.basename(tmp_filename)}",
                volumes={os.path.dirname(tmp_filename): {'bind': '/tmp', 'mode': 'ro'}},
                detach=True,
                stdout=True,
                stderr=True,
                tty=False
            )
            container.wait(timeout=settings.CODE_EXECUTION_TIMEOUT)
            logs = container.logs().decode("utf-8")
            result["output"] = logs
            container.remove()
        except Exception as e:
            result["error"] = f"Code execution error: {e}"
        finally:
            try:
                os.remove(tmp_filename)
            except Exception as e:
                logger.error("Error deleting temporary file: %s", e)
        return result 