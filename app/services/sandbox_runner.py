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

class DockerSandboxRunner(SandboxRunner):
    """Runs code safely within Docker containers."""
    
    def __init__(self, 
                 image: str = "python:3.9-slim",
                 cpu_limit: float = 1.0,
                 memory_limit: str = "256m",
                 network_disabled: bool = True,
                 working_dir: str = "/code"):
        """Initialize Docker sandbox configuration.
        
        Args:
            image: Docker image to use for execution
            cpu_limit: CPU limit as float (1.0 = 1 core)
            memory_limit: Memory limit (e.g., '256m')
            network_disabled: Whether to disable network access
            working_dir: Working directory inside container
        """
        self.image = image
        self.cpu_limit = cpu_limit
        self.memory_limit = memory_limit
        self.network_disabled = network_disabled
        self.working_dir = working_dir
        self.client = None
        self.container_prefix = "sandbox_"
        
    async def initialize(self):
        """Initialize Docker client and pull the required image."""
        await super().initialize()
        
        try:
            # Connect to Docker daemon
            self.client = docker.from_env()
            
            # Pull the required image
            logger.info(f"Pulling Docker image: {self.image}")
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, 
                                       lambda: self.client.images.pull(self.image))
            logger.info(f"Successfully pulled image: {self.image}")
            
        except DockerException as e:
            logger.error(f"Failed to initialize Docker sandbox: {str(e)}")
            raise RuntimeError(f"Docker initialization failed: {str(e)}")
    
    async def run_code(self, 
                       code_to_run: str, 
                       timeout_seconds: int = 30,
                       language: str = "python") -> SandboxExecutionResult:
        """Run code within a Docker container with appropriate safeguards.
        
        Args:
            code_to_run: The code string to execute
            timeout_seconds: Maximum execution time in seconds
            language: Programming language of the code (defaults to 'python')
            
        Returns:
            SandboxExecutionResult with execution details
        """
        if not self.client:
            await self.initialize()
            
        container = None
        temp_dir = None
        
        try:
            # Create unique ID for this execution
            execution_id = str(uuid.uuid4())[:8]
            container_name = f"{self.container_prefix}{execution_id}"
            
            # Create temporary directory to share with container
            temp_dir = tempfile.mkdtemp(prefix="docker_sandbox_")
            
            # Save code to a file
            file_extension = self._get_file_extension(language)
            code_filename = f"code{file_extension}"
            code_path = os.path.join(temp_dir, code_filename)
            
            with open(code_path, 'w') as f:
                f.write(code_to_run)
            
            # Prepare container configuration
            volumes = {
                temp_dir: {'bind': self.working_dir, 'mode': 'ro'}
            }
            
            # Prepare run command based on language
            command = self._get_run_command(language, code_filename)
            
            # Start container with appropriate limits
            logger.info(f"Starting container {container_name} with {language} code")
            loop = asyncio.get_event_loop()
            container = await loop.run_in_executor(
                None,
                lambda: self.client.containers.run(
                    self.image,
                    command,
                    name=container_name,
                    volumes=volumes,
                    working_dir=self.working_dir,
                    cpu_quota=int(self.cpu_limit * 100000),
                    mem_limit=self.memory_limit,
                    network_disabled=self.network_disabled,
                    cap_drop=['ALL'],
                    security_opt=['no-new-privileges'],
                    detach=True,
                    remove=False,
                    read_only=True
                )
            )
            
            # Wait for execution with timeout
            try:
                exit_code = await asyncio.wait_for(
                    loop.run_in_executor(None, container.wait),
                    timeout=timeout_seconds
                )
                
                # Get logs
                stdout = container.logs(stdout=True, stderr=False).decode('utf-8', errors='replace')
                stderr = container.logs(stdout=False, stderr=True).decode('utf-8', errors='replace')
                
                success = exit_code['StatusCode'] == 0
                return SandboxExecutionResult(
                    success=success,
                    stdout=stdout,
                    stderr=stderr,
                    exit_code=exit_code['StatusCode']
                )
                
            except asyncio.TimeoutError:
                logger.warning(f"Container {container_name} timed out after {timeout_seconds}s")
                return SandboxExecutionResult(
                    success=False,
                    error=f"Execution timed out after {timeout_seconds} seconds"
                )
                
        except (DockerException, ContainerError, ImageNotFound) as e:
            logger.error(f"Docker execution error: {str(e)}")
            return SandboxExecutionResult(
                success=False,
                error=f"Docker execution failed: {str(e)}"
            )
        except Exception as e:
            logger.exception(f"Unexpected error in Docker sandbox: {str(e)}")
            return SandboxExecutionResult(
                success=False,
                error=f"Sandbox error: {str(e)}"
            )
        finally:
            # Clean up container
            if container:
                try:
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(None, container.stop)
                    await loop.run_in_executor(None, container.remove)
                except Exception as e:
                    logger.error(f"Error cleaning up container: {str(e)}")
            
            # Clean up temporary directory
            if temp_dir and os.path.exists(temp_dir):
                try:
                    import shutil
                    shutil.rmtree(temp_dir)
                except Exception as e:
                    logger.error(f"Error removing temporary directory: {str(e)}")
    
    def _get_file_extension(self, language: str) -> str:
        """Get the appropriate file extension for a given language."""
        extensions = {
            "python": ".py",
            "javascript": ".js",
            "nodejs": ".js",
            "ruby": ".rb",
            "go": ".go",
            "java": ".java",
            "csharp": ".cs",
            "c": ".c",
            "cpp": ".cpp",
            "php": ".php"
        }
        return extensions.get(language.lower(), ".txt")
    
    def _get_run_command(self, language: str, filename: str) -> str:
        """Get the appropriate command to execute code in the given language."""
        commands = {
            "python": ["python", filename],
            "javascript": ["node", filename],
            "nodejs": ["node", filename],
            "ruby": ["ruby", filename],
            "go": ["go", "run", filename],
            "java": ["java", filename],  # Assumes compiled separately
            "csharp": ["dotnet", "run", filename],  # Assumes .NET Core
            "c": ["gcc", "-o", "program", filename, "&&", "./program"],
            "cpp": ["g++", "-o", "program", filename, "&&", "./program"],
            "php": ["php", filename]
        }
        
        cmd = commands.get(language.lower())
        if not cmd:
            raise ValueError(f"Unsupported language: {language}")
            
        # If multiple args, join them for shell execution
        if isinstance(cmd, list):
            return cmd
        return cmd
    
    async def cleanup(self):
        """Clean up any resources created by the Docker sandbox."""
        await super().cleanup()
        
        if self.client:
            try:
                # Find and remove all containers created by this runner
                containers = self.client.containers.list(
                    all=True, 
                    filters={"name": self.container_prefix}
                )
                
                loop = asyncio.get_event_loop()
                for container in containers:
                    try:
                        logger.info(f"Cleaning up container: {container.name}")
                        if container.status == 'running':
                            await loop.run_in_executor(None, container.stop)
                        await loop.run_in_executor(None, container.remove)
                    except Exception as e:
                        logger.error(f"Error removing container {container.name}: {str(e)}")
                        
            except Exception as e:
                logger.error(f"Error during Docker cleanup: {str(e)}")
            
            # Close the Docker client
            try:
                self.client.close()
                self.client = None
            except:
                pass 