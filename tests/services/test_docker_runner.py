import asyncio
import pytest
from unittest.mock import MagicMock, patch, AsyncMock

import docker
from docker.errors import APIError, ContainerError, DockerException, ImageNotFound

from app.services.docker_runner import (DEFAULT_PYTHON_IMAGE,
                                        DockerSandboxRunner,
                                        SandboxExecutionResult)

# Use pytest-asyncio for async tests
pytestmark = pytest.mark.asyncio

@pytest.fixture
def mock_docker_client():
    """Fixture to provide a mocked Docker client."""
    with patch('docker.from_env') as mock_from_env:
        mock_client = MagicMock(spec=docker.DockerClient)
        mock_client.ping.return_value = True
        mock_client.images = MagicMock()
        mock_client.containers = MagicMock()
        mock_from_env.return_value = mock_client
        yield mock_client

@pytest.fixture
def runner(mock_docker_client): # Depends on the mocked client
    """Fixture to create a DockerSandboxRunner instance with a mocked client."""
    # Mock image check during init
    mock_docker_client.images.get.return_value = MagicMock() # Assume image exists

    runner_instance = DockerSandboxRunner()
    # Ensure the runner uses the mocked client we created
    runner_instance.client = mock_docker_client
    return runner_instance

# --- Test Class --- #

class TestDockerSandboxRunner:

    async def test_init_success(self, mock_docker_client):
        """Test successful initialization when Docker is available."""
        mock_docker_client.images.get.return_value = MagicMock() # Image found
        try:
            runner_instance = DockerSandboxRunner()
            assert runner_instance.client == mock_docker_client
            mock_docker_client.ping.assert_called_once()
            mock_docker_client.images.get.assert_called_once_with(DEFAULT_PYTHON_IMAGE)
            mock_docker_client.images.pull.assert_not_called()
        except RuntimeError:
            pytest.fail("Initialization failed unexpectedly.")

    async def test_init_pull_image(self, mock_docker_client):
        """Test initialization pulls image if not found locally."""
        mock_docker_client.images.get.side_effect = ImageNotFound("Not found")
        mock_docker_client.images.pull.return_value = MagicMock()

        try:
            runner_instance = DockerSandboxRunner()
            assert runner_instance.client == mock_docker_client
            mock_docker_client.ping.assert_called_once()
            mock_docker_client.images.get.assert_called_once_with(DEFAULT_PYTHON_IMAGE)
            mock_docker_client.images.pull.assert_called_once_with(DEFAULT_PYTHON_IMAGE)
        except RuntimeError:
            pytest.fail("Initialization failed unexpectedly during image pull.")

    async def test_init_docker_unavailable(self):
        """Test initialization raises RuntimeError if Docker is unavailable."""
        with patch('docker.from_env') as mock_from_env:
            mock_from_env.side_effect = DockerException("Docker daemon not running")
            with pytest.raises(RuntimeError, match="Docker client initialization failed"):
                DockerSandboxRunner()

    # --- run_code Tests --- #

    async def test_run_code_success(self, runner: DockerSandboxRunner, mock_docker_client):
        """Test successful code execution."""
        mock_container = MagicMock()
        mock_container.wait.return_value = {'StatusCode': 0, 'Error': None}
        mock_container.logs.side_effect = [
            b'print("Hello") output', # stdout
            b''                     # stderr
        ]
        mock_container.remove.return_value = None
        mock_docker_client.containers.run.return_value = mock_container

        code = "print('Hello')"
        result = await runner.run_code(code)

        assert result.success is True
        assert result.exit_code == 0
        assert result.stdout == 'print("Hello") output'
        assert result.stderr == ''
        assert result.error is None
        mock_docker_client.containers.run.assert_called_once()
        mock_container.wait.assert_called_once_with(timeout=30) # Default timeout
        mock_container.remove.assert_called_once()

    async def test_run_code_with_error(self, runner: DockerSandboxRunner, mock_docker_client):
        """Test execution with non-zero exit code and stderr."""
        mock_container = MagicMock()
        mock_container.wait.return_value = {'StatusCode': 1, 'Error': None}
        mock_container.logs.side_effect = [
            b'',                             # stdout
            b'Traceback...\nValueError: bad' # stderr
        ]
        mock_container.remove.return_value = None
        mock_docker_client.containers.run.return_value = mock_container

        code = "raise ValueError('bad')"
        result = await runner.run_code(code)

        assert result.success is False
        assert result.exit_code == 1
        assert result.stdout == ''
        assert result.stderr == 'Traceback...\nValueError: bad'
        assert 'Execution failed with exit code 1' in result.error
        assert 'ValueError: bad' in result.error # Check stderr is included
        mock_docker_client.containers.run.assert_called_once()
        mock_container.wait.assert_called_once_with(timeout=30)
        mock_container.remove.assert_called_once()

    async def test_run_code_timeout(self, runner: DockerSandboxRunner, mock_docker_client):
        """Test code execution timeout."""
        mock_container = MagicMock()
        # Simulate timeout exception from container.wait()
        mock_container.wait.side_effect = docker.errors.APIError("Timeout waiting for container")
        mock_container.remove.return_value = None
        mock_docker_client.containers.run.return_value = mock_container

        code = "import time; time.sleep(5)"
        result = await runner.run_code(code, timeout_seconds=2)

        assert result.success is False
        assert result.exit_code is None
        assert result.stdout is None # Or potentially empty depending on exact mock
        assert result.stderr is None # Or potentially empty
        assert 'Timeout' in result.error # Check for timeout specific error
        mock_docker_client.containers.run.assert_called_once()
        mock_container.wait.assert_called_once_with(timeout=2) # Check custom timeout
        mock_container.remove.assert_called_once()

    # Placeholder for a test simulating ContainerError if needed
    # async def test_run_code_container_error(self, runner, mock_docker_client):
    #     pass

    async def test_run_code_docker_api_error(self, runner: DockerSandboxRunner, mock_docker_client):
        """Test handling of Docker API errors during container run."""
        mock_docker_client.containers.run.side_effect = APIError("Docker daemon error")

        code = "print('test')"
        result = await runner.run_code(code)

        assert result.success is False
        assert 'Docker API error' in result.error
        assert 'Docker daemon error' in result.error

    async def test_run_code_client_not_initialized(self):
        """Test run_code returns error if Docker client failed initialization."""
        # Simulate failed init by patching from_env
        with patch('docker.from_env') as mock_from_env:
            mock_from_env.side_effect = DockerException("No Docker")
            with pytest.raises(RuntimeError): # Init raises error
                runner_instance = DockerSandboxRunner()

            # Create an instance manually *after* ensuring client is None
            runner_instance = object.__new__(DockerSandboxRunner) # Bypass __init__ check
            runner_instance.client = None

            result = await runner_instance.run_code("print(1)")
            assert result.success is False
            assert result.error == "Docker client not initialized."

    # Placeholder - Add tests for file writing errors, volume mounting etc. if desired. 