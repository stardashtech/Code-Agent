import docker
import tempfile
import os
from config import CODE_EXECUTION_TIMEOUT, DOCKER_IMAGE
from agent.logger import get_logger

logger = get_logger(__name__)

class CodeExecutor:
    """
    Executes code in a container using Docker.
    This example creates and runs a simple Docker Python container.
    """

    def __init__(self):
        try:
            self.client = docker.from_env()
        except Exception as e:
            logger.error("Failed to initialize Docker client: %s", e)
            raise

    def execute(self, code: str) -> dict:
        result = {"output": None, "error": None}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp:
            tmp.write(code)
            tmp_filename = tmp.name

        try:
            # We run with a mount instead of copying the file into the container
            container = self.client.containers.run(
                DOCKER_IMAGE,
                command=f"python /tmp/{os.path.basename(tmp_filename)}",
                volumes={os.path.dirname(tmp_filename): {'bind': '/tmp', 'mode': 'ro'}},
                detach=True,
                stdout=True,
                stderr=True,
                tty=False
            )
            container.wait(timeout=CODE_EXECUTION_TIMEOUT)
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