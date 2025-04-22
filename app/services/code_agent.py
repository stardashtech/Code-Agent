import asyncio
import json
import logging
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple
import aiohttp
import redis
import requests
from openai import OpenAI
from qdrant_client import QdrantClient
import hashlib
from datetime import datetime, timedelta
import re
import ast # Import ast module

from app.config import settings
from app.agents.reflector import Reflector
from app.agents.planner import Planner
from app.services.vector_store_manager import VectorStoreManager
from app.services.sandbox_runner import SandboxRunner, SandboxExecutionResult
from app.services.docker_runner import DockerSandboxRunner
from app.services.plan_executor import PlanExecutor
from app.tools.web_search import TavilySearchProvider # <-- Added import

# Import providers from their new locations
from app.tools.github_search import GitHubSearchProvider # <-- Removed GitHubApiClient import
from app.tools.stackoverflow_search import StackOverflowSearchProvider
# Import other potential clients (even if not fully implemented yet)
from app.tools.pypi_client import PyPiClient
# Import placeholder or future clients (adjust paths if needed)
from app.tools.npm_client import NpmClient
from app.tools.go_proxy_client import GoProxyClient
from app.tools.nuget_client import NuGetClient
from app.services.doc_scraper import DocumentationScraper
from app.services.llm_info_extractor import LlmInfoExtractor

# ---> ADD IMPORT FOR ValidationService <--- 
from app.services.validation_service import ValidationService

logger = logging.getLogger(__name__)

class Config:
    """Configuration class for CodeAgent"""
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        openai_model: str = "gpt-4",
        openrouter_api_key: Optional[str] = None,
        openrouter_model: str = "openai/gpt-4",
        ollama_model: str = "codellama",
        embedding_dimension: int = 1536,
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        cache_ttl: int = 3600
    ):
        self.openai_api_key = openai_api_key
        self.openai_model = openai_model
        self.openrouter_api_key = openrouter_api_key
        self.openrouter_model = openrouter_model
        self.ollama_model = ollama_model
        self.embedding_dimension = embedding_dimension
        self.qdrant_host = qdrant_host
        self.qdrant_port = qdrant_port
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.cache_ttl = cache_ttl

class LLMProvider:
    """Base class for LLM providers"""
    async def generate_completion(self, messages: List[Dict[str, str]], temperature: float = 0.7) -> str:
        raise NotImplementedError
        
    async def generate_embedding(self, text: str) -> List[float]:
        raise NotImplementedError

class OpenAIProvider(LLMProvider):
    """OpenAI API provider"""
    def __init__(self, api_key: str, model: str):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        
    async def generate_completion(self, messages: List[Dict[str, str]], temperature: float = 0.7) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature
        )
        return response.choices[0].message.content
        
    async def generate_embedding(self, text: str) -> List[float]:
        try:
            response = self.client.embeddings.create(
                model="text-embedding-3-small",
                input=text,
                dimensions=settings.embedding_dimension
            )
            embedding = response.data[0].embedding
            
            # Verify embedding dimension
            if len(embedding) != settings.embedding_dimension:
                logger.error(f"OpenAI returned wrong embedding dimension: got {len(embedding)}, expected {settings.embedding_dimension}")
                # Truncate or pad to match required dimension
                if len(embedding) > settings.embedding_dimension:
                    embedding = embedding[:settings.embedding_dimension]
                else:
                    embedding.extend([0.0] * (settings.embedding_dimension - len(embedding)))
                    
            return embedding
        except Exception as e:
            logger.error(f"Error generating OpenAI embedding: {str(e)}")
            raise

class OpenRouterProvider(LLMProvider):
    """OpenRouter API provider for accessing multiple LLM models through a unified API"""
    def __init__(self, api_key: str, model: str):
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1"
        )
        self.model = model
        logger.info(f"Initialized OpenRouterProvider with model {model}")
        
    async def generate_completion(self, messages: List[Dict[str, str]], temperature: float = 0.7) -> str:
        """Generate completion using OpenRouter API."""
        try:
            # Log request details
            logger.debug(f"Sending request to OpenRouter - Model: {self.model}, Temperature: {temperature}")
            logger.debug(f"Messages: {messages}")
            
            # Make API request with proper parameters
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=2000,
                timeout=30
            )
            
            # Log raw response for debugging
            logger.debug(f"OpenRouter raw response: {response}")
            
            # Validate response structure
            if not response or not hasattr(response, 'choices') or not response.choices:
                raise ValueError("Invalid or empty response from OpenRouter")
                
            choice = response.choices[0]
            if not choice or not hasattr(choice, 'message') or not choice.message:
                raise ValueError("No valid message in OpenRouter response")
                
            content = choice.message.content
            if not content or not isinstance(content, str):
                raise ValueError("Invalid content in OpenRouter response")
                
            # Log successful completion
            logger.info("Successfully generated completion via OpenRouter")
            return content.strip()
            
        except Exception as e:
            logger.error(f"OpenRouter completion failed: {str(e)}")
            raise  # Re-raise the exception to be handled by caller
        
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embeddings using OpenRouter API."""
        try:
            # OpenRouter doesn't support embeddings directly
            # You might want to implement a different embedding provider here
            raise NotImplementedError("OpenRouter does not support embeddings")
        except Exception as e:
            logger.error(f"Error generating embedding via OpenRouter: {str(e)}")
            raise

class OllamaProvider(LLMProvider):
    """Ollama API provider"""
    def __init__(self, model: str = settings.ollama_model):
        self.model = model
        self.base_url = settings.ollama_base_url
        # Use a specific model for embeddings that supports the feature
        self.embedding_model = "nomic-embed-text"
        
    async def generate_completion(self, messages: List[Dict[str, str]], temperature: float = 0.7) -> str:
        async with aiohttp.ClientSession() as session:
            try:
                prompt = self._convert_messages_to_prompt(messages)
                async with session.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "temperature": temperature,
                        "stream": False
                    }
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"Ollama API error: {error_text}")
                        
                    result = await response.json()
                    return result["response"]
                    
            except Exception as e:
                logger.error(f"Ollama completion error: {str(e)}")
                raise
                
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embeddings using Ollama API with a specific embedding model"""
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    f"{self.base_url}/api/embeddings",
                    json={
                        "model": self.embedding_model,
                        "prompt": text
                    }
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"Ollama embeddings API error: {error_text}")
                        
                    result = await response.json()
                    embedding = result["embedding"]
                    
                    # Verify embedding dimension
                    if len(embedding) != settings.embedding_dimension:
                        logger.error(f"Ollama returned wrong embedding dimension: got {len(embedding)}, expected {settings.embedding_dimension}")
                        # Truncate or pad to match required dimension
                        if len(embedding) > settings.embedding_dimension:
                            embedding = embedding[:settings.embedding_dimension]
                        else:
                            embedding.extend([0.0] * (settings.embedding_dimension - len(embedding)))
                            
                    return embedding
                    
            except Exception as e:
                logger.error(f"Error generating Ollama embedding: {str(e)}")
                raise
                
    def _convert_messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert chat messages to a single prompt string"""
        prompt = ""
        for message in messages:
            role = message["role"]
            content = message["content"]
            
            if role == "system":
                prompt += f"System: {content}\n"
            elif role == "user":
                prompt += f"Human: {content}\n"
            elif role == "assistant":
                prompt += f"Assistant: {content}\n"
                
        return prompt.strip()

class VLLMProvider(LLMProvider):
    """Provider for high-performance inference using vLLM API."""
    
    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        embedding_model: Optional[str] = None,
    ):
        """Initialize the vLLM provider.
        
        Args:
            base_url: Base URL for the vLLM API. Defaults to settings.vllm_url.
            model: Model to use for completions. Defaults to settings.vllm_model.
            embedding_model: Model to use for embeddings. Defaults to settings.vllm_embedding_model.
        """
        settings = settings
        self.base_url = base_url or settings.vllm_url
        self.model = model or settings.vllm_model
        self.embedding_model = embedding_model or settings.vllm_embedding_model
        self.client = aiohttp.AsyncClient(timeout=60.0)
        logger.info(f"Initialized vLLM provider with URL: {self.base_url}, model: {self.model}")
    
    async def generate_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1000,
        top_p: float = 0.95,
    ) -> str:
        """Generate a completion using the vLLM API.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'.
            temperature: Sampling temperature.
            max_tokens: Maximum number of tokens to generate.
            top_p: Top-p sampling parameter.
            
        Returns:
            Generated text response.
            
        Raises:
            Exception: If the API request fails.
        """
        try:
            # Format messages into prompt
            prompt = ""
            for msg in messages:
                role = msg["role"]
                content = msg["content"]
                if role == "system":
                    prompt += f"System: {content}\n"
                elif role == "user":
                    prompt += f"User: {content}\n"
                elif role == "assistant":
                    prompt += f"Assistant: {content}\n"
            prompt += "Assistant: "

            # Prepare request payload
            payload = {
                "prompt": prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": top_p,
                "stream": False
            }

            # Make request to vLLM API
            response = await self.client.post(
                f"{self.base_url}/v1/completions",
                json=payload
            )
            
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"vLLM API request failed with status {response.status}: {error_text}")
                
            result = await response.json()
            return result["choices"][0]["text"]
            
        except Exception as e:
            logger.error(f"Error generating completion with vLLM: {str(e)}")
            raise

    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embeddings using the vLLM embedding model.
        
        Args:
            text: Text to generate embeddings for.
            
        Returns:
            List of embedding values.
            
        Raises:
            Exception: If the embedding request fails.
        """
        try:
            response = await self.client.post(
                f"{self.base_url}/v1/embeddings",
                json={
                    "input": text,
                    "model": self.embedding_model,
                    "dimensions": settings.embedding_dimension
                }
            )
            
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"Embedding request failed with status {response.status}: {error_text}")
                
            result = await response.json()
            embedding = result["data"][0]["embedding"]
            
            # Check embedding dimension
            if len(embedding) != settings.embedding_dimension:
                logger.error(f"Embedding dimension mismatch: got {len(embedding)}, expected {settings.embedding_dimension}")
                # Truncate or pad the embedding to match the expected dimension
                if len(embedding) > settings.embedding_dimension:
                    embedding = embedding[:settings.embedding_dimension]
                else:
                    embedding.extend([0.0] * (settings.embedding_dimension - len(embedding)))
                    
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding with vLLM: {str(e)}")
            raise

class CodeAgent:
    """Main agent class that orchestrates LLM interactions and tool usage."""
    def __init__(self, config: Config):
        """Initialize the code agent with providers and managers."""
        self.config = config
        # Separate providers for completion and embedding
        self.completion_provider: Optional[LLMProvider] = None
        self.embedding_provider: Optional[LLMProvider] = None
        # self.provider is now an alias for completion_provider for backward compatibility
        # with components like Reflector and Planner that expect a single provider.
        self.provider: Optional[LLMProvider] = None 
        self.executor = ThreadPoolExecutor()
        self.sandbox_runner: Optional[SandboxRunner] = None
        self.plan_executor: Optional[PlanExecutor] = None
        self.web_search_provider: Optional[TavilySearchProvider] = None
        self.github_search_provider: Optional[GitHubSearchProvider] = None
        self.stackoverflow_search_provider: Optional[StackOverflowSearchProvider] = None
        # Explicitly define client attributes expected by PlanExecutor (mostly as placeholders)
        # self.github_client: Optional[GitHubApiClient] = None <-- Removed
        self.pypi_client: Optional[PyPiClient] = None
        # ---> Initialize NpmClient attribute <--- 
        self.npm_client: Optional[NpmClient] = None
        self.go_proxy_client: Optional[GoProxyClient] = None
        self.nuget_client: Optional[NuGetClient] = None
        self.doc_scraper: Optional[DocumentationScraper] = None
        self.llm_info_extractor: Optional[LlmInfoExtractor] = None
        self.validation_service: Optional[ValidationService] = None
        self.workspace_root = os.getcwd() # Define workspace root

        # --- Initialize LLM Providers ---
        
        # 1. Initialize Completion Provider
        completion_provider_initialized = False
        if settings.llm_provider == "openai" and settings.openai_api_key:
            logger.info(f"Initializing OpenAI as completion provider (Model: {settings.openai_model})")
            try:
                self.completion_provider = OpenAIProvider(settings.openai_api_key, settings.openai_model)
                completion_provider_initialized = True
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI completion provider: {e}")
        elif settings.llm_provider == "openrouter" and settings.openrouter_api_key:
            logger.info(f"Initializing OpenRouter as completion provider (Model: {settings.openrouter_model})")
            try:
                self.completion_provider = OpenRouterProvider(settings.openrouter_api_key, settings.openrouter_model)
                completion_provider_initialized = True
            except Exception as e:
                 logger.warning(f"Failed to initialize OpenRouter completion provider: {e}")
        elif settings.llm_provider == "ollama":
            logger.info(f"Initializing Ollama as completion provider (Model: {settings.ollama_model})")
            try:
                 self.completion_provider = OllamaProvider(model=settings.ollama_model)
                 completion_provider_initialized = True
            except Exception as e:
                 logger.warning(f"Failed to initialize Ollama completion provider: {e}")
        elif settings.llm_provider == "vllm":
            logger.info("Initializing vLLM as completion provider")
            try:
                self.completion_provider = VLLMProvider()
                completion_provider_initialized = True
            except Exception as e:
                 logger.warning(f"Could not initialize vLLM completion provider: {e}. Check configuration.")
        else:
            logger.warning(f"Completion LLM Provider '{settings.llm_provider}' not recognized or required API key missing.")

        # Fallback for Completion Provider (try Ollama if primary failed and wasn't Ollama)
        if not completion_provider_initialized and settings.llm_provider != "ollama" and settings.ollama_base_url:
             logger.info("Attempting Fallback: Initializing Ollama as completion provider")
             try:
                 self.completion_provider = OllamaProvider(model=settings.ollama_model)
                 completion_provider_initialized = True
             except Exception as e:
                 logger.warning(f"Failed to initialize Ollama fallback completion provider: {e}")

        if not self.completion_provider:
            # If still no completion provider, this is a critical failure
            raise ValueError("FATAL: No valid LLM provider could be initialized for completions.")
            
        # Alias self.provider to self.completion_provider for compatibility
        self.provider = self.completion_provider
        logger.info(f"Completion provider set to: {self.completion_provider.__class__.__name__}")

        # 2. Initialize Embedding Provider (Prioritize Ollama, fallback to OpenAI)
        embedding_provider_initialized = False
        # Try Ollama first for embeddings if configured
        if settings.ollama_base_url:
            logger.info(f"Attempting to initialize Ollama as embedding provider (Model: {OllamaProvider().embedding_model})")
            try:
                self.embedding_provider = OllamaProvider() # Uses default embedding model internally
                # Test connection immediately
                # await self.embedding_provider.generate_embedding("test") # Cannot await in __init__
                logger.info(f"Successfully initialized Ollama as embedding provider.")
                embedding_provider_initialized = True
            except Exception as e:
                 logger.warning(f"Could not initialize Ollama as embedding provider: {e}. Will try fallback.")
                 
        # Fallback to OpenAI if Ollama failed or wasn't configured, and OpenAI is configured
        if not embedding_provider_initialized and settings.openai_api_key:
            logger.info("Attempting Fallback: Initializing OpenAI as embedding provider")
            try:
                 # Use the same OpenAI client instance if completion provider is also OpenAI
                 if isinstance(self.completion_provider, OpenAIProvider):
                     self.embedding_provider = self.completion_provider
                     logger.info("Reusing OpenAI completion provider instance for embeddings.")
                 else:
                     self.embedding_provider = OpenAIProvider(settings.openai_api_key, settings.openai_model) # Model choice might not matter much here
                     logger.info("Initialized separate OpenAI provider instance for embeddings.")
                 embedding_provider_initialized = True
            except Exception as e:
                 logger.warning(f"Could not initialize OpenAI as embedding provider: {e}")

        # Fallback to vLLM if others failed and vLLM is configured
        if not embedding_provider_initialized and settings.vllm_url and settings.vllm_embedding_model:
             logger.info("Attempting Fallback: Initializing vLLM as embedding provider")
             try:
                  if isinstance(self.completion_provider, VLLMProvider):
                       self.embedding_provider = self.completion_provider # Reuse if possible
                       logger.info("Reusing vLLM completion provider instance for embeddings.")
                  else:
                       self.embedding_provider = VLLMProvider() # Creates new instance
                       logger.info("Initialized separate vLLM provider instance for embeddings.")
                  embedding_provider_initialized = True
             except Exception as e:
                  logger.warning(f"Could not initialize vLLM as embedding provider: {e}")

        if not self.embedding_provider:
             # This is serious, as vector store relies on it. Log critical error.
             logger.critical("FATAL: No embedding provider (Ollama, OpenAI, or vLLM) could be initialized. Vector search will fail.")
             # We might allow the agent to continue without embeddings, but log appropriately.
             # For now, let initialization continue but VectorStoreManager might fail later.
        else:
             logger.info(f"Embedding provider set to: {self.embedding_provider.__class__.__name__}")

        # --- Initialize Agents (Pass the *completion* provider) ---
        if not self.completion_provider: # Redundant check, but safe
             raise ValueError("FATAL: Completion provider not set before initializing dependent agents.")
        self.reflector = Reflector(self.completion_provider)
        self.planner = Planner(self.completion_provider)

        # Initialize Search Providers and Clients
        try:
            self.github_search_provider = GitHubSearchProvider(access_token=settings.github_token)
            logger.info(f"Initialized GitHubSearchProvider (Token Present: {bool(settings.github_token)}).")
        except Exception as e:
            logger.warning(f"Could not initialize GitHubSearchProvider: {e}. GitHub search disabled.")
            self.github_search_provider = None

        try:
             self.stackoverflow_search_provider = StackOverflowSearchProvider(api_key=settings.stackoverflow_key)
             logger.info(f"Initialized StackOverflowSearchProvider (Key Present: {bool(settings.stackoverflow_key)}).")
        except Exception as e:
             logger.warning(f"Could not initialize StackOverflowSearchProvider: {e}. StackOverflow search disabled.")
             self.stackoverflow_search_provider = None

        try:
            self.web_search_provider = TavilySearchProvider()
            logger.info("Initialized TavilySearchProvider.")
        except Exception as e:
            logger.error(f"Failed to initialize TavilySearchProvider: {e}. Web search disabled.", exc_info=True)
            self.web_search_provider = None

        # Initialize other clients (as None for now, implement later if needed)
        # Example for PyPi:
        # try:
        #     self.pypi_client = PyPiClient()
        #     logger.info("Initialized PyPiClient.")
        # except Exception as e:
        #     logger.warning(f"Could not initialize PyPiClient: {e}")
        #     self.pypi_client = None
        self.pypi_client = None # Placeholder
        self.go_proxy_client = None # Placeholder
        self.nuget_client = None # Placeholder
        self.doc_scraper = None # Placeholder - Needs implementation
        self.llm_info_extractor = None # Placeholder - Needs implementation


        # Initialize Redis Client (for embedding cache)
        try:
            self.redis_client = redis.Redis.from_url(
                settings.redis_url,
                password=settings.redis_password,
                db=settings.redis_db,
                decode_responses=True
            )
            self.redis_client.ping()
            logger.info("Redis client initialized successfully.")
        except redis.exceptions.ConnectionError as e:
             logger.warning(f"Could not connect to Redis: {e}. Caching disabled.")
             self.redis_client = None
        except Exception as e:
            logger.error(f"An unexpected error occurred during Redis initialization: {e}", exc_info=True)
            self.redis_client = None

        # Initialize Vector Store Manager
        try:
            # Check if embedding provider is available before initializing VectorStoreManager
            if not self.embedding_provider:
                 raise ValueError("Embedding provider is not available, cannot initialize VectorStoreManager.")
                 
            self.vector_store_manager = VectorStoreManager(
                qdrant_host=f"http://{config.qdrant_host}:{config.qdrant_port}",
                embedding_dimension=config.embedding_dimension,
                embedding_func=self.generate_embedding # This method now uses self.embedding_provider
            )
            logger.info("VectorStoreManager initialized successfully.")
        except Exception as e:
             logger.critical(f"FATAL: Failed to initialize VectorStoreManager: {e}", exc_info=True)
             # Decide if agent should raise or continue with degraded functionality
             # raise # Vector store might be critical
             self.vector_store_manager = None # Allow continuation but log critical failure
             logger.critical("Continuing without VectorStoreManager functionality.")


        # Initialize Sandbox Runner
        try:
            self.sandbox_runner = DockerSandboxRunner()
            logger.info("DockerSandboxRunner initialized successfully.")
        except RuntimeError as e:
             # This specifically catches the Docker client init failure
             logger.error(f"Failed to initialize DockerSandboxRunner: {e}. Code execution/validation disabled.", exc_info=False) # Don't need full TB for this
             self.sandbox_runner = None
        except Exception as e:
             logger.error(f"An unexpected error occurred during DockerSandboxRunner initialization: {e}. Code execution disabled.", exc_info=True)
             self.sandbox_runner = None

        # ---> MODIFIED Validation Service Initialization <--- 
        try:
            # Initialize ValidationService using the created class
            if self.sandbox_runner: # Only initialize if runner exists
                self.validation_service = ValidationService(self.sandbox_runner)
                logger.info("ValidationService initialized.")
            else:
                 logger.warning("Sandbox runner not available, skipping ValidationService initialization.")
                 self.validation_service = None
        # Keep existing NameError catch just in case import fails
        except NameError:
             logger.error("'ValidationService' class not found. Ensure app/services/validation_service.py exists and is importable. Validation disabled.")
             self.validation_service = None
        except Exception as e:
             # Log specific error during ValidationService init
             logger.error(f"Failed to initialize ValidationService: {e}", exc_info=True)
             self.validation_service = None
        # ---> END MODIFIED Block <--- 

        # Initialize Plan Executor - MUST be last as it depends on agent attributes
        try:
            self.plan_executor = PlanExecutor(self)
            logger.info("PlanExecutor initialized successfully.")
        except Exception as e:
            logger.critical(f"FATAL: Failed to initialize PlanExecutor: {e}", exc_info=True)
            raise # Plan executor is critical

        # ---> Initialize NpmClient <--- 
        try:
            # Initialize NpmClient (consider session management)
            self.npm_client = NpmClient()
            logger.info("Initialized NpmClient.")
        except Exception as e:
             logger.warning(f"Could not initialize NpmClient: {e}", exc_info=True)
             self.npm_client = None
        # ---> END NpmClient Initialization <--- 

        logger.info("CodeAgent initialized successfully (Potential warnings above for optional components).")
        
    async def generate_completion(self, messages: List[Dict[str, str]], temperature: float = 0.7) -> str:
        # Use the dedicated completion provider
        if not self.completion_provider:
            raise RuntimeError("Completion LLM Provider not initialized.")
        logger.debug(f"Using completion provider: {self.completion_provider.__class__.__name__}")
        try:
            return await self.completion_provider.generate_completion(messages, temperature)
        except Exception as e:
            logger.error(f"Error generating completion: {str(e)}")
            raise
            
    async def generate_embedding(self, text: str) -> List[float]:
        # Use the dedicated embedding provider
        if not self.embedding_provider:
             logger.error("Embedding provider not initialized. Cannot generate embedding.")
             # Return a zero vector or raise error? Raising seems safer.
             raise RuntimeError("Embedding LLM Provider not initialized.")
             
        logger.debug(f"Using embedding provider: {self.embedding_provider.__class__.__name__}")
             
        cache_key = f"embedding:{hashlib.sha256(text.encode()).hexdigest()}"
        
        # --- Cache Check ---
        if self.redis_client:
            try:
                cached_embedding_json = self.redis_client.get(cache_key)
                if cached_embedding_json:
                    cached_embedding = json.loads(cached_embedding_json)
                    # Validate cache entry before returning
                    if isinstance(cached_embedding, list) and len(cached_embedding) == self.config.embedding_dimension:
                        logger.debug(f"Cache hit for embedding: {cache_key[:15]}...")
                        return cached_embedding
                    else:
                        logger.warning(f"Invalid cache format for {cache_key} (dim: {len(cached_embedding)} vs {self.config.embedding_dimension}). Fetching fresh.")
                else:
                     logger.debug(f"Cache miss for embedding: {cache_key[:15]}...")
            except redis.exceptions.RedisError as e:
                logger.warning(f"Redis GET error for {cache_key}: {e}. Fetching fresh.")
            except json.JSONDecodeError as e:
                 logger.warning(f"Cache JSON decode error for {cache_key}: {e}. Fetching fresh.")
            except Exception as e: # Catch unexpected errors during cache check
                 logger.warning(f"Unexpected error during cache check for {cache_key}: {e}. Fetching fresh.")

        # --- Generate Embedding using dedicated provider ---
        try:
            start_time = time.time()
            embedding = await self.embedding_provider.generate_embedding(text)
            duration = time.time() - start_time
            logger.debug(f"Generated embedding via {self.embedding_provider.__class__.__name__} in {duration:.2f}s")

            # --- Dimension Validation (already done in providers, but double-check maybe?) ---
            if len(embedding) != self.config.embedding_dimension:
                logger.error(f"{self.embedding_provider.__class__.__name__} returned wrong dimension: {len(embedding)} != {self.config.embedding_dimension}. Attempting fix.")
                # Apply padding/truncation (redundant if providers do it, but safe)
                if len(embedding) > self.config.embedding_dimension:
                    embedding = embedding[:self.config.embedding_dimension]
                else:
                    embedding.extend([0.0] * (self.config.embedding_dimension - len(embedding)))

            # --- Store in Cache ---
            if self.redis_client:
                try:
                    embedding_json = json.dumps(embedding)
                    self.redis_client.setex(cache_key, self.config.cache_ttl, embedding_json)
                    logger.debug(f"Stored embedding in cache: {cache_key[:15]}...")
                except redis.exceptions.RedisError as e:
                    logger.warning(f"Redis SETEX error for {cache_key}: {e}")
                except Exception as e: # Catch unexpected errors during cache storage
                     logger.warning(f"Unexpected error storing embedding in cache for {cache_key}: {e}")

            return embedding
        except NotImplementedError:
             logger.critical(f"FATAL: Configured embedding provider {self.embedding_provider.__class__.__name__} does not support embeddings!")
             raise RuntimeError(f"Embedding provider {self.embedding_provider.__class__.__name__} does not support embeddings.")
        except Exception as e:
            logger.error(f"Error generating embedding via provider {self.embedding_provider.__class__.__name__}: {str(e)}", exc_info=True)
            raise # Re-raise the original error

    async def _generate_code_fix(self, query: str, code_snippets: List[Dict[str, Any]], 
                          analysis: str) -> Dict[str, Any]:
        # Use self.generate_completion which now points to the correct provider
        if not self.completion_provider: # Check completion provider specifically
             logger.error("Completion LLM Provider not available for code fix generation.")
             return self._generate_error_response("Completion LLM Provider not configured.")

        if not code_snippets:
            logger.warning("No code snippets provided for fix generation")
            return self._generate_error_response("No code snippets available for analysis")

        relevant_snippets = sorted(code_snippets, key=lambda x: x.get('similarity', 0), reverse=True)[:2]
        code_context = "\n\n".join([
            f"File: {snippet.get('file_path', 'unknown')}\n```\n{snippet.get('code', '')}\n```"
            for snippet in relevant_snippets
        ])

        system_prompt = """You are an expert code reviewer and fixer. Analyze the user query, the provided code context, and the initial analysis. Provide specific code fixes or improvements in a structured JSON format.

VERY IMPORTANT RESPONSE RULES:
1. Respond ONLY with a valid JSON object.
2. DO NOT use markdown formatting (```json ... ```) around the JSON object itself.
3. DO NOT include any explanatory text outside the JSON object.
4. Ensure all strings within the JSON are properly escaped (e.g., use \\n for newlines in code).
5. Use double quotes for all JSON keys and string values.
6. If no fix is needed or possible, provide an explanation within the JSON structure.
7. The value for the 'fixed_code' key MUST be a single JSON string containing the complete fixed code snippet (with newlines escaped as \\n), OR null if no single code block fix is applicable.
8. The value for the 'file_path' key MUST be a JSON string representing the primary file path affected, OR null if not applicable or affects multiple files.
9. REPEAT: 'fixed_code' and 'file_path' MUST be JSON strings or null. DO NOT use lists or other types for these fields.

Example response format:
{
    "explanation": "Brief explanation of the fix or why no fix is needed.",
    "fixed_code": "def example():\\n    return True", // MUST be a single JSON string or null
    "file_path": "path/to/relevant/file.py", // MUST be a JSON string or null
    "changes": [ // Optional: specific line-by-line changes
        {
            "line_number": 1,
            "original": "old code line",
            "replacement": "new code line",
            "explanation": "Reason for this specific change."
        }
    ]
}"""

        user_prompt = f"""User Query: {query}\n\nCode Context:\n{code_context}\n\nPrevious Analysis:\n{analysis}\n\nPlease provide the fix in the specified JSON format."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        max_retries = 2
        attempts = 0
        while attempts <= max_retries:
            attempts += 1
            try:
                logger.info(f"Attempt {attempts}: Generating code fix...")
                response_text = await self.generate_completion(messages)
                logger.debug(f"Raw LLM fix response: {response_text}")
                
                # Clean the response
                cleaned_response_text = self._clean_json_response(response_text)
                logger.debug(f"Cleaned LLM fix response: {cleaned_response_text}")

                # Attempt to parse as JSON
                try:
                    fix_details = json.loads(cleaned_response_text)
                    if not isinstance(fix_details, dict):
                         raise json.JSONDecodeError("Response is not a JSON object.", cleaned_response_text, 0)
                         
                    # Basic validation of structure (can be expanded)
                    # Enhanced validation (from ISSUE-001)
                    if 'fixed_code' in fix_details and not isinstance(fix_details['fixed_code'], (str, type(None))):
                        logger.warning(f"LLM returned non-string/null type for fixed_code: {type(fix_details['fixed_code'])}. Setting to null.")
                        fix_details['fixed_code'] = None
                    if 'file_path' in fix_details and not isinstance(fix_details['file_path'], (str, type(None))):
                        logger.warning(f"LLM returned non-string/null type for file_path: {type(fix_details['file_path'])}. Setting to null.")
                        fix_details['file_path'] = None
                        
                    # Syntax check (from ISSUE-002)
                    fixed_code_to_check = fix_details.get('fixed_code')
                    syntax_check_result = {"status": "not_attempted", "error": None}
                    if isinstance(fixed_code_to_check, str) and fixed_code_to_check.strip():
                        try:
                            # Clean up potential LLM artifacts before parsing
                            cleaned_code = fixed_code_to_check.strip().lstrip("`").rstrip("`").strip()
                            # Remove potential language identifier like ```python
                            if cleaned_code.startswith("python\n"):
                                cleaned_code = cleaned_code[len("python\n"):]
                            elif cleaned_code.startswith("python "):
                                 cleaned_code = cleaned_code[len("python "):]
                                 
                            ast.parse(cleaned_code) 
                            syntax_check_result = {"status": "passed", "error": None}
                            logger.info("Generated fixed_code passed syntax check.")
                        except SyntaxError as syn_err:
                            logger.warning(f"Generated fixed_code failed syntax check: {syn_err}")
                            logger.debug(f"Problematic Code:\n---\n{fixed_code_to_check}\n---")
                            syntax_check_result = {"status": "failed", "error": str(syn_err)}
                        except Exception as parse_e: # Catch other potential errors during parse
                             logger.error(f"Unexpected error during fixed_code syntax check: {parse_e}")
                             syntax_check_result = {"status": "error", "error": f"Internal error during syntax check: {parse_e}"}
                    elif fixed_code_to_check is None:
                        syntax_check_result = {"status": "skipped", "error": "No fixed code provided."}
                    else:
                        syntax_check_result = {"status": "skipped", "error": "Fixed code was empty or invalid type."}
                        
                    fix_details['syntax_check'] = syntax_check_result
                    fix_details["status"] = "success" # Mark success if JSON parsed
                    return fix_details

                except json.JSONDecodeError as e:
                    logger.warning(f"Attempt {attempts}: Failed to parse LLM response as JSON: {e}")
                    if attempts > max_retries:
                        logger.error("Max retries reached. Failed to get valid JSON for code fix.")
                        return self._generate_error_response(f"LLM failed to produce valid JSON: {cleaned_response_text}")
                    await asyncio.sleep(1) # Wait before retry

            except Exception as e:
                logger.error(f"Attempt {attempts}: Error generating code fix: {e}", exc_info=True)
                if attempts > max_retries:
                    return self._generate_error_response(f"Error generating code fix: {e}")
                await asyncio.sleep(1)
                
        # Should not be reached if logic is correct, but as a safeguard
        return self._generate_error_response("Failed to generate code fix after retries.")

    def _clean_json_response(self, response: str) -> str:
        response = response.strip()
        if response.startswith("```json"):
            response = response[len("```json"):].strip()
        elif response.startswith("```"):
             response = response[len("```"):].strip()
             
        if response.endswith("```"):
            response = response[:-len("```")]
            
        # Sometimes models might still include explanations after the JSON block
        # Find the first { and the last } to extract the core JSON part
        first_brace = response.find('{')
        last_brace = response.rfind('}')

        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            potential_json = response[first_brace:last_brace+1]
            # Quick check if it looks like valid JSON before returning
            # This isn't foolproof but catches simple explanation text appended after
            try:
                json.loads(potential_json) # Test if it parses
                return potential_json
            except json.JSONDecodeError:
                logger.warning("Could not isolate JSON block reliably, returning original cleaned string.")
                return response # Fallback to original cleaned string if extraction fails
        else:
            logger.warning("Could not find valid JSON start/end braces, returning original cleaned string.")
            return response # Fallback if braces aren't found

    def _validate_and_fix_response(self, response: Dict[str, Any], provider: LLMProvider) -> Dict[str, Any]:
        # This method seems deprecated or unused based on current code flow
        # Keeping it for reference but potentially remove later
        logger.warning("_validate_and_fix_response called - may be deprecated")
        # Simple validation example
        if "fixed_code" not in response or "file_path" not in response:
            logger.warning(f"Invalid response format from {provider.__class__.__name__}. Attempting to fix.")
            # Attempt to fix or return error structure
            return self._generate_error_response("Invalid response format received")
        return response
        
    def _generate_error_response(self, error_message: str) -> Dict[str, Any]:
        return {
            "status": "error",
            "explanation": error_message,
            "fixed_code": None,
            "file_path": None,
            "changes": [],
            "syntax_check": {"status": "n/a", "error": "Error response generated."}
        }
        
    # --- Data Storage Methods ---
    async def store_code(self, file_path: str, content: str, language: str, clear_existing: bool = False) -> Dict[str, Any]:
        if clear_existing:
            await self.vector_store_manager.delete_points_by_path(file_path)
            logger.info(f"Cleared existing vector points for file: {file_path}")
            
        if not content or not content.strip():
             logger.warning(f"Attempted to store empty content for {file_path}. Skipping.")
             return {"status": "skipped", "reason": "Empty content"}

        try:
            point_id = await self.vector_store_manager.store_code(file_path, content, language)
            return {"status": "success", "point_id": point_id, "file": file_path}
        except Exception as e:
            logger.error(f"Failed to store code for {file_path}: {e}", exc_info=True)
            return {"status": "error", "file": file_path, "error": str(e)}

    async def _save_interaction(self, query: str, response_data: Dict) -> None:
        # Simplified interaction saving
        interaction_data = {
            "query": query,
            "response_status": response_data.get("status"),
            "response_message": response_data.get("message")
            # Add other relevant fields from response if needed
        }
        try:
            await self.vector_store_manager.save_interaction(interaction_data)
        except Exception as e:
            logger.error(f"Failed to save interaction: {e}", exc_info=True)

    # --- Core Agent Logic Methods ---
    async def process_mcp_request(self, message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        # Placeholder for MCP interaction logic
        logger.info(f"Received MCP request: {message}")
        # Example: Parse message, call other methods, return MCP response
        if message.startswith("fix:"):
            query = message[len("fix:"):].strip()
            return await self.run(query)
        else:
            return {"status": "error", "message": "Unknown MCP command"}
            
    async def process_query(self, query: str) -> Dict:
         # Deprecated? Replaced by run method?
         logger.warning("process_query called - likely deprecated, use run() instead.")
         # Simplified example: just run the main loop
         return await self.run(query)

    async def generate_code_fix(self, query: str, code_snippets: List[Dict[str, Any]], error_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        # This seems like a duplicate or older version of _generate_code_fix
        logger.warning("generate_code_fix called - likely deprecated, use _generate_code_fix internally.")
        # Combine context
        analysis_context = "Error Info:\n" + json.dumps(error_info, indent=2) if error_info else "No specific error info provided."
        return await self._generate_code_fix(query, code_snippets, analysis_context)

    async def analyze_code(self, code: str, original_query: Optional[str] = None) -> str:
        # This method uses self.generate_completion, which uses self.completion_provider
        if not self.completion_provider:
            logger.error("Completion LLM Provider not available for code analysis.")
            return "Error: Completion LLM Provider not configured."
            
        system_prompt = "You are an expert code reviewer. Analyze the following code snippet(s) in the context of the user query, identify potential issues, suggest improvements, and explain the overall functionality."
        user_prompt = f"User Query (optional context): {original_query}\n\nCode to Analyze:\n```\n{code}\n```\n\nPlease provide a concise analysis:"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        try:
            analysis = await self.generate_completion(messages)
            return analysis
        except Exception as e:
            logger.error(f"Error during code analysis: {e}", exc_info=True)
            return f"Error during analysis: {e}"
            
    async def generate_test(self, code: str, test_type: str = "unit") -> Dict[str, Any]:
        # This method uses self.generate_completion, which uses self.completion_provider
        if not self.completion_provider:
            return self._generate_error_response("Completion LLM Provider not configured for test generation.")

        system_prompt = f"You are a test generation expert. Generate a {test_type} test case for the given code. Respond ONLY in valid JSON format like {{\"test_code\": \"...\", \"explanation\": \"...\"}}. Ensure code is properly escaped."
        user_prompt = f"Generate a {test_type} test for this code:\n```\n{code}\n```"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        try:
            response_text = await self.generate_completion(messages)
            cleaned_response_text = self._clean_json_response(response_text)
            test_details = json.loads(cleaned_response_text)
            test_details["status"] = "success"
            return test_details
        except Exception as e:
            logger.error(f"Error generating test: {e}", exc_info=True)
            return self._generate_error_response(f"Failed to generate test: {e}")

    async def explain_code(self, code: str, language: str) -> str:
        # This method uses self.generate_completion, which uses self.completion_provider
        if not self.completion_provider:
             return "Error: Completion LLM Provider not configured."

        system_prompt = "You are an expert code explainer. Provide a clear and concise explanation of the given code snippet."
        user_prompt = f"Explain this {language} code:\n```\n{code}\n```"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        try:
            explanation = await self.generate_completion(messages)
            return explanation
        except Exception as e:
            logger.error(f"Error explaining code: {e}", exc_info=True)
            return f"Error during explanation: {e}"

    async def run(self, query: str, conversation_history: Optional[list[Dict[str, str]]] = None) -> Dict[str, Any]:
        logger.info(f"Starting agent run for query: '{query[:50]}...'")
        response_data = {"results": {}, "status": "processing", "message": "Starting analysis..."}
        execution_context = {"query": query, "conversation_history": conversation_history or []}

        try:
            # 1. Reflection Phase (Clarity, Keywords, Decomposition, Plan)
            logger.info("Phase 1: Reflection")
            
            # --- ENHANCE-007: Provide code context to Reflector --- 
            # Perform a quick initial search to get potential context
            initial_code_context = None
            # Ensure embedding provider is available before attempting search
            if self.embedding_provider and self.vector_store_manager:
                 try:
                     # Simple search using the raw query
                     initial_snippets = await self.vector_store_manager.search_code(query, filter_dict={"type": "code_master"})
                     if initial_snippets:
                         # Provide file paths as context
                         file_paths = list(set([snippet.get('file_path', 'unknown') for snippet in initial_snippets]))
                         initial_code_context = f"Potentially relevant files found: {', '.join(file_paths)}"
                         logger.info(f"Providing initial code context to Reflector: {initial_code_context}")
                 except RuntimeError as search_err: # Catch specific error if embedding provider failed
                      logger.warning(f"Initial context search skipped: Embedding provider error: {search_err}")
                 except Exception as search_err:
                      logger.warning(f"Initial context search failed: {search_err}")
            else:
                 logger.warning("Skipping initial context search: Embedding provider or Vector Store Manager not available.")
            # --- End ENHANCE-007 --- 

            # --- Call individual reflection and planning steps (using completion provider via self.reflector/self.planner) ---
            code_context_for_reflection = initial_code_context # Use the context found

            clarity_result = await self.reflector.assess_query_clarity(
                query=query,
                conversation_history=execution_context["conversation_history"],
                code_context=code_context_for_reflection
            )
            
            is_clear = clarity_result == 'CLEAR'
            clarification_question = None if is_clear else clarity_result
            clarity_reason = None if is_clear else "Query assessed as ambiguous by Reflector." # Simplified reason

            keywords = await self.reflector.extract_keywords(
                query=query, 
                conversation_history=execution_context["conversation_history"]
            )
            
            decomposed_queries = await self.reflector.decompose_query(
                query=query, 
                conversation_history=execution_context["conversation_history"]
            )

            # Use the planner to create the plan
            plan = await self.planner.create_plan(
                query=query, 
                extracted_keywords=keywords, 
                decomposed_queries=decomposed_queries
            )
            
            # --- Consolidate results ---
            reflection_results = {
                "is_clear": is_clear,
                "clarity_reason": clarity_reason,
                "clarification_question": clarification_question,
                "keywords": keywords,
                "decomposition": decomposed_queries,
                "plan": plan
            }
            # --- End Reflector/Planner calls ---
            
            execution_context.update(reflection_results)
            response_data["results"]["reflection"] = reflection_results # Include reflection details

            if not is_clear: # Use the determined clarity flag
                logger.warning(f"Query assessed as unclear: {clarity_reason}")
                response_data["status"] = "clarification_needed"
                response_data["message"] = clarification_question or "The query is unclear. Please provide more details."
                return response_data # Return early for clarification
                
            if not plan: # Check if plan generation failed
                 logger.error("Planner failed to generate a plan.")
                 return self._generate_error_response("Failed to generate execution plan.")
                 
            # plan = reflection_results["plan"] # Plan is already assigned
            logger.info(f"Generated Plan: {json.dumps(plan, indent=2)}")

            # 2. Execution Phase (using PlanExecutor)
            logger.info("Phase 2: Execution")
            if not self.plan_executor:
                 logger.error("PlanExecutor not initialized.")
                 return self._generate_error_response("Internal Error: PlanExecutor not available.")
                 
            await self.plan_executor.execute_plan(plan, execution_context, response_data)
            # execute_plan updates response_data directly
            
            # 3. Finalization/Saving (Optional)
            logger.info("Phase 3: Finalization")
            # Final status/message should be set by execute_plan
            if response_data.get("status") == "processing": # Ensure a final status is set
                 response_data["status"] = "completed"
                 response_data["message"] = response_data.get("message", "Processing completed, but final status unclear.")
                 
            # await self._save_interaction(query, response_data) # Optionally save

        except Exception as e:
            logger.critical(f"Critical error during agent run for query '{query}': {e}", exc_info=True)
            # Return a standard error structure
            response_data = self._generate_error_response(f"Agent run failed: {e}")
            response_data["results"] = execution_context # Include context for debugging

        logger.info(f"Agent run finished for query '{query[:50]}...' Status: {response_data.get('status')}")
        return response_data 

    # Add a cleanup method if clients create their own sessions
    async def close_resources(self):
         """Closes any resources managed by the agent, like network sessions."""
         logger.info("Closing agent resources...")
         closed_successfully = True
         if isinstance(self.pypi_client, PyPiClient):
             try:
                 await self.pypi_client.close_session()
                 logger.info("Closed PyPiClient session.")
             except Exception as e:
                 logger.error(f"Error closing PyPiClient session: {e}", exc_info=True)
                 closed_successfully = False
                 
         # ---> Add NpmClient closing logic <--- 
         if isinstance(self.npm_client, NpmClient):
             try:
                 await self.npm_client.close_session()
                 logger.info("Closed NpmClient session.")
             except Exception as e:
                 logger.error(f"Error closing NpmClient session: {e}", exc_info=True)
                 closed_successfully = False
         # ---> END NpmClient closing logic <--- 

         # Add closing logic for GoProxyClient, NuGetClient, DocScraper, LlmInfoExtractor if they manage resources
         # ... rest of close_resources ...

         if closed_successfully:
              logger.info("Agent resources closed successfully.")
         else:
              logger.warning("Some agent resources failed to close properly.") 