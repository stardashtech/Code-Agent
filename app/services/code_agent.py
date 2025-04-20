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
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, PointStruct, VectorParams
from qdrant_client.http.models import Filter as QdrantFilter
from qdrant_client.http.models import FieldCondition, MatchValue
import hashlib
from datetime import datetime, timedelta
import re

from app.config import settings
from app.agents.reflector import Reflector
from app.agents.planner import Planner

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

class CodeSearchProvider:
    """Base class for code search providers"""
    async def search(self, query: str, language: Optional[str] = None) -> List[Dict[str, Any]]:
        raise NotImplementedError

class GitHubSearchProvider(CodeSearchProvider):
    """GitHub code search provider"""
    def __init__(self, access_token: Optional[str] = None):
        self.access_token = access_token
        self.headers = {"Authorization": f"token {access_token}"} if access_token else {}
        
    async def search(self, query: str, language: Optional[str] = None) -> List[Dict[str, Any]]:
        search_query = f"{query} language:{language}" if language else query
        async with aiohttp.ClientSession() as session:
            async with session.get(
                "https://api.github.com/search/code",
                params={"q": search_query},
                headers=self.headers
            ) as response:
                result = await response.json()
                return result.get("items", [])

class StackOverflowSearchProvider(CodeSearchProvider):
    """Stack Overflow search provider"""
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        
    async def search(self, query: str, language: Optional[str] = None) -> List[Dict[str, Any]]:
        params = {
            "site": "stackoverflow",
            "key": self.api_key,
            "tagged": language if language else None,
            "q": query
        }
        async with aiohttp.ClientSession() as session:
            async with session.get(
                "https://api.stackexchange.com/2.3/search/advanced",
                params={k: v for k, v in params.items() if v is not None}
            ) as response:
                result = await response.json()
                return result.get("items", [])

class CodeAgent:
    """Main agent class that handles LLM interactions"""
    def __init__(self, config: Config):
        """Initialize the code agent with the appropriate provider based on config"""
        self.config = config
        self.provider: Optional[LLMProvider] = None
        self.executor = ThreadPoolExecutor()
        
        # Initialize the appropriate provider based on configuration
        if config.openai_api_key:
            logger.info("Initializing OpenAI provider")
            self.provider = OpenAIProvider(config.openai_api_key, config.openai_model)
            
        elif config.openrouter_api_key:
            logger.info("Initializing OpenRouter provider")
            self.provider = OpenRouterProvider(config.openrouter_api_key, config.openrouter_model)
            
        else:
            logger.info("Initializing Ollama provider")
            self.provider = OllamaProvider(model=config.ollama_model)
            
        if not self.provider:
            raise ValueError("No valid LLM provider could be initialized")
            
        # Initialize the Reflector agent
        self.reflector = Reflector(self.provider)
        # Initialize the Planner agent (passing the provider for future use)
        self.planner = Planner(self.provider)
        
        # Initialize Qdrant client
        self.qdrant_client = QdrantClient(
            url=config.qdrant_host # qdrant_host from Config actually holds the full URL
        )
        
        # Initialize Redis client
        try:
            self.redis_client = redis.Redis(
                host=config.redis_host,
                port=config.redis_port,
                decode_responses=True
            )
            self.redis_client.ping() # Test connection
            logger.info("Redis client initialized successfully.")
        except redis.exceptions.ConnectionError as e:
             logger.warning(f"Could not connect to Redis at {config.redis_host}:{config.redis_port}. Caching will be disabled. Error: {e}")
             self.redis_client = None
        
        # Set collection name
        self.collection_name = self._get_collection_name()
        
        # Ensure collection exists
        self._ensure_collection_exists()
        
        logger.info("CodeAgent initialized successfully")
        
    async def generate_completion(self, messages: List[Dict[str, str]], temperature: float = 0.7) -> str:
        """Generate completion using the configured provider"""
        if not self.provider:
            raise RuntimeError("LLM Provider not initialized.")
        try:
            return await self.provider.generate_completion(messages, temperature)
        except Exception as e:
            logger.error(f"Error generating completion: {str(e)}")
            raise
            
    async def generate_embedding(self, text: str) -> List[float]:
        """Get embedding for text using the configured provider, with caching."""
        # Generate cache key
        cache_key = f"embedding:{hashlib.sha256(text.encode()).hexdigest()}"

        # Try to get from cache
        if self.redis_client:
            try:
                cached_embedding_json = self.redis_client.get(cache_key)
                if cached_embedding_json:
                    cached_embedding = json.loads(cached_embedding_json)
                    # Basic validation
                    if isinstance(cached_embedding, list) and len(cached_embedding) == self.config.embedding_dimension:
                        logger.info(f"Cache hit for embedding: {cache_key}")
                        return cached_embedding
                    else:
                        logger.warning(f"Invalid embedding format found in cache for key {cache_key}. Fetching fresh.")
                else:
                     logger.info(f"Cache miss for embedding: {cache_key}")
            except redis.exceptions.RedisError as e:
                logger.warning(f"Redis error while getting cache key {cache_key}: {e}. Fetching fresh.")
            except json.JSONDecodeError as e:
                 logger.warning(f"Error decoding cached embedding for key {cache_key}: {e}. Fetching fresh.")

        # If not in cache or cache failed, generate embedding
        # Ensure provider exists before calling generate_embedding
        if not self.provider:
             raise RuntimeError("LLM Provider not initialized when trying to generate embedding.")

        try:
            # Call the provider's method, not the agent's recursive call
            embedding = await self.provider.generate_embedding(text)

            # Store in cache if Redis is available
            if self.redis_client:
                try:
                    embedding_json = json.dumps(embedding)
                    self.redis_client.setex(cache_key, self.config.cache_ttl, embedding_json)
                    logger.info(f"Stored embedding in cache: {cache_key}")
                except redis.exceptions.RedisError as e:
                    logger.warning(f"Redis error while setting cache key {cache_key}: {e}")
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding via provider: {str(e)}", exc_info=True)
            raise # Re-raise the exception

    def _get_collection_name(self) -> str:
        """Get the collection name based on the current configuration"""
        return f"code_embeddings_{self.config.embedding_dimension}"
        
    def _ensure_collection_exists(self):
        """Ensure the vector store collection exists with correct configuration"""
        try:
            collections = self.qdrant_client.get_collections().collections
            exists = any(c.name == self.collection_name for c in collections)
            
            if not exists:
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.config.embedding_dimension,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Created new collection: {self.collection_name}")
            else:
                logger.info(f"Using existing collection: {self.collection_name}")
                
        except Exception as e:
            logger.error(f"Error ensuring collection exists: {str(e)}")
            raise
            
    async def search_code(self, query: str) -> List[Dict]:
        """Search for relevant code snippets based on the query"""
        try:
            # Get query embedding
            query_embedding = await self.generate_embedding(query)
            
            # Search for similar vectors
            results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=5
            )
            
            if not results:  # Check if results list is empty
                return []
                
            # Process and return results
            processed_results = []
            for match in results:  # Results is already a list of ScoredPoint objects
                payload = match.payload
                # Filter out results where 'code' is missing or None
                if payload and payload.get('code') is not None:
                    processed_results.append({
                        'code': payload.get('code', ''), # Ensure 'code' key is used
                        'file_path': payload.get('file_path', ''),
                        'language': payload.get('language', 'unknown'),
                        'similarity': match.score
                    })
                else:
                     logger.warning(f"Skipping Qdrant result with missing or empty code payload: ID {match.id}")

            return processed_results
            
        except Exception as e:
            logger.error(f"Error searching code: {str(e)}", exc_info=True)
            return []
            
    async def index_code(self, code: str, metadata: Dict) -> None:
        """Index a code snippet with its embedding"""
        try:
            # Generate embedding for code
            code_embedding = await self.generate_embedding(code)
            
            # Add document to vector store
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=[
                    models.PointStruct(
                        id=str(uuid.uuid4()),
                        vector=code_embedding,
                        payload={
                            "code": code, # Ensure 'code' key is used for storing content
                            "file_path": metadata.get('file_path', ''),
                            "language": metadata.get('language', 'unknown'),
                            **metadata
                        }
                    )
                ]
            )
            
        except Exception as e:
            logger.error(f"Error indexing code: {str(e)}")
            raise

    async def _generate_code_fix(self, query: str, code_snippets: List[Dict[str, Any]], 
                          analysis: str) -> Dict[str, Any]: # Analysis is now string
        """Generate code fix using LLM based on the query and found code snippets."""
        if not self.provider:
             logger.error("LLM Provider not available for code fix generation.")
             return self._generate_error_response("LLM Provider not configured.")

        if not code_snippets:
            logger.warning("No code snippets provided for fix generation")
            return self._generate_error_response("No code snippets available for analysis")

        # Prepare context from code snippets (limit to most relevant)
        # Using similarity score which is available from search_code results
        relevant_snippets = sorted(code_snippets, key=lambda x: x.get('similarity', 0), reverse=True)[:2]
        code_context = "\n\n".join([
            f"File: {snippet.get('file_path', 'unknown')}\n```\n{snippet.get('code', '')}\n```" # Use 'code' key
            for snippet in relevant_snippets
        ])

        system_prompt = """You are an expert code reviewer and fixer. Analyze the user query, the provided code context, and the initial analysis. Provide specific code fixes or improvements in a structured JSON format.

IMPORTANT RESPONSE RULES:
1. Respond ONLY with a valid JSON object.
2. DO NOT use markdown formatting (```json ... ```) around the JSON object itself.
3. DO NOT include any explanatory text outside the JSON object.
4. Ensure all strings within the JSON are properly escaped (e.g., use \\n for newlines in code).
5. Use double quotes for all JSON keys and string values.
6. If no fix is needed or possible, provide an explanation within the JSON structure.
7. The value for the 'fixed_code' key MUST be a single JSON string containing the complete fixed code snippet, OR null if not applicable.
8. The value for the 'file_path' key MUST be a single JSON string representing the primary file path affected, OR null if not applicable.

Example response format:
{
    "explanation": "Brief explanation of the fix or why no fix is needed.",
    "fixed_code": "def example():\\n    return True", // MUST be a single string or null
    "file_path": "path/to/relevant/file", // MUST be a single string or null
    "changes": [ // Optional: specific line-by-line changes
        {
            "line_number": 1,
            "original": "old code line",
            "replacement": "new code line",
            "explanation": "Reason for this specific change."
        }
    ]
}"""

        fix_prompt = f"""Review and suggest improvements for the code based on the query and analysis:

User Query: {query}

Initial Analysis:
{analysis}

Relevant Code Context:
{code_context}

STRICT RESPONSE RULES (REPEAT):
1. Return ONLY a valid JSON object adhering to the structure specified in the system prompt.
2. NO markdown formatting around the JSON.
3. NO explanatory text outside the JSON object.
4. Ensure all JSON strings are properly escaped (use \\n for newlines).
5. Use double quotes for all keys and string values.
6. The value for 'fixed_code' MUST be a single JSON string or null.
7. The value for 'file_path' MUST be a single JSON string or null.
8. Provide specific changes in the 'changes' array if applicable."""

        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": fix_prompt}
            ]

            logger.debug(f"Attempting code fix generation with provider: {self.provider.__class__.__name__}")
            # Use the instance provider and await the coroutine
            raw_content = await self.provider.generate_completion(messages, temperature=0.2)

            # Clean and validate the response
            cleaned_content = self._clean_json_response(raw_content)
            if not cleaned_content or cleaned_content == "{}":
                logger.warning(f"Provider {self.provider.__class__.__name__} returned empty or invalid JSON structure for code fix.")
                # Fallback: return the raw analysis if fix generation failed badly
                return {"status": "warning", "explanation": "Failed to generate a structured code fix. Analysis provided.", "analysis": analysis}

            try:
                code_fix = json.loads(cleaned_content)
            except json.JSONDecodeError as je:
                logger.error(f"JSON parsing error for code fix from {self.provider.__class__.__name__}: {str(je)}")
                logger.debug(f"Problematic JSON content for code fix: {cleaned_content[:500]}...")
                # Fallback: return the raw analysis
                return {"status": "error", "explanation": f"Failed to parse code fix JSON: {je}", "analysis": analysis}

            # Validate and fix the response structure
            # Pass the provider for metadata logging
            code_fix = self._validate_and_fix_response(code_fix, self.provider)

            # Optional: Syntax check (ensure fixed_code is present and valid)
            if code_fix.get('fixed_code') and isinstance(code_fix['fixed_code'], str):
                try:
                    compile(code_fix['fixed_code'].replace('\\n', '\n'), '<string>', 'exec')
                    code_fix['syntax_check'] = {'status': 'passed'}
                except SyntaxError as se:
                    logger.warning(f"Syntax error in generated fixed_code: {str(se)}")
                    code_fix['syntax_check'] = {'status': 'failed', 'error': str(se)}
            else:
                 # If fixed_code isn't provided or is not a string, indicate syntax check wasn't applicable
                 if 'syntax_check' not in code_fix:
                    code_fix['syntax_check'] = {'status': 'not_applicable'}

            logger.info(f"Successfully generated code fix with {self.provider.__class__.__name__}")
            code_fix['status'] = 'success' # Add status if not already present from validation
            return code_fix

        except Exception as e:
            error_msg = f"Unexpected error generating code fix: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return self._generate_error_response(error_msg)

    def _clean_json_response(self, response: str) -> str:
        """Clean and prepare the response for JSON parsing."""
        if not response:
            logger.error("Empty response received for JSON cleaning.")
            return "{}"

        try:
            # Attempt 1: Direct parsing (ideal case)
            try:
                json.loads(response.strip())
                return response.strip()
            except json.JSONDecodeError:
                logger.debug("Direct JSON parsing failed, attempting cleaning...")

            # Attempt 2: Find JSON within potential markdown code blocks
            code_block_pattern = r"```(?:json)?\s*({.*?})\s*```" # Capture content within {} inside block
            match = re.search(code_block_pattern, response, re.DOTALL)
            if match:
                potential_json = match.group(1).strip()
                try:
                    json.loads(potential_json)
                    logger.info("Extracted valid JSON from markdown code block.")
                    return potential_json
                except json.JSONDecodeError as e:
                    logger.debug(f"Invalid JSON found within code block: {e}")

            # Attempt 3: Find the outermost JSON object aggressively
            json_outer_pattern = r"^.*?(\{.*\}).*?$" # Match from start to first { and last } to end
            match = re.search(json_outer_pattern, response, re.DOTALL)
            if match:
                potential_json = match.group(1).strip()
                try:
                    json.loads(potential_json)
                    logger.info("Extracted potentially valid JSON using outer braces match.")
                    return potential_json
                except json.JSONDecodeError as e:
                     logger.debug(f"Failed to parse JSON extracted via outer braces: {e}")
                     # Attempt cleanup on this extracted part
                     # Simple fix: remove trailing commas before closing brace/bracket
                     cleaned_json = re.sub(r',(\s*[}\]])', r'\1', potential_json)
                     try:
                         json.loads(cleaned_json)
                         logger.info("Successfully parsed JSON after cleaning trailing commas.")
                         return cleaned_json
                     except json.JSONDecodeError as final_e:
                          logger.error(f"Could not parse JSON even after aggressive extraction and cleaning: {final_e}")


            logger.error("Could not extract valid JSON from response after multiple attempts.")
            logger.debug(f"Original response causing JSON cleaning failure: {response[:500]}...")
            return "{}"

        except Exception as e:
            logger.error(f"Unexpected error during JSON cleaning: {str(e)}")
            return "{}"

    def _validate_and_fix_response(self, response: Dict[str, Any], provider: LLMProvider) -> Dict[str, Any]:
        """Validate and fix the response structure, ensuring expected fields are present."""
        if not isinstance(response, dict):
            logger.error("Response to validate is not a dictionary")
            return self._generate_error_response("Invalid response format (not a dictionary).")

        # Expected fields based on the _generate_code_fix prompt structure
        expected_fields = {
            'explanation': (str, ""),
            'fixed_code': (str, None), # Optional, default None
            'file_path': (str, None), # Optional, default None
            'changes': (list, []) # Optional list, default empty
        }

        validated_response = {}
        validation_issues = False

        for field, (expected_type, default_value) in expected_fields.items():
            value = response.get(field)
            if value is None:
                if default_value is not None:
                     validated_response[field] = default_value
                     if field not in ['fixed_code', 'file_path', 'changes']: # Missing mandatory fields is an issue
                         logger.warning(f"Missing field '{field}' in code fix response. Using default: '{default_value}'")
                         validation_issues = True
                else:
                     validated_response[field] = None # Keep optional fields as None if missing

            elif not isinstance(value, expected_type):
                logger.warning(f"Incorrect type for field '{field}' in code fix response. Expected {expected_type}, got {type(value)}. Attempting to use default.")
                validated_response[field] = default_value
                validation_issues = True
            else:
                validated_response[field] = value

        # Specific validation for 'changes' list structure if present
        if validated_response['changes'] is not None:
            valid_changes = []
            change_item_keys = {'line_number', 'original', 'replacement', 'explanation'}
            for item in validated_response['changes']:
                if isinstance(item, dict) and change_item_keys.issubset(item.keys()):
                    valid_changes.append(item)
                else:
                    logger.warning(f"Invalid item found in 'changes' list: {item}. Skipping.")
                    validation_issues = True
            validated_response['changes'] = valid_changes

        # Add metadata
        validated_response['metadata'] = {
            'timestamp': datetime.now().isoformat(),
            'provider': provider.__class__.__name__ if provider else 'unknown'
        }

        # Set status based on validation
        validated_response['status'] = 'warning' if validation_issues else response.get('status', 'success')

        # Calculate quality score (simple version: 1.0 if no issues, 0.5 if warnings, 0.0 if error)
        if validated_response['status'] == 'success':
             validated_response['quality_score'] = 1.0
        elif validated_response['status'] == 'warning':
             validated_response['quality_score'] = 0.5
        else: # error or missing status assumed error
             validated_response['quality_score'] = 0.0

        # Preserve original error info if passed through
        if 'error' in response and isinstance(response['error'], dict):
             validated_response['error'] = response['error']
             validated_response['status'] = 'error'
             validated_response['quality_score'] = 0.0


        return validated_response

    def _generate_error_response(self, error_message: str) -> Dict[str, Any]:
        """Generate a standardized error response."""
        provider_name = self.provider.__class__.__name__ if self.provider else 'unknown'

        return {
            'explanation': f"Error: {error_message}",
            'fixed_code': None,
            'file_path': None,
            'changes': [],
            'error': {
                'message': error_message,
                'timestamp': datetime.now().isoformat()
            },
            'status': 'error',
            'quality_score': 0.0,
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'provider': provider_name
            }
        }

    async def _save_interaction(self, query: str, code_snippets: List[Dict[str, Any]],
                         code_fix: Dict[str, Any]) -> None:
        """Save the interaction details to the vector store for future reference."""
        interaction_text = f"""
        Query: {query}
        Code Snippets Found: {len(code_snippets)}
        Files: {', '.join(s.get('file_path', 'unknown') for s in code_snippets)}
        Result Explanation:
        {code_fix.get('explanation', 'No explanation provided')}
        Fix Status: {code_fix.get('status', 'unknown')}
        """
        try:
            # Await the async embedding method
            embedding = await self.generate_embedding(interaction_text)
        except Exception as e:
            logger.error(f"Failed to generate embedding for saving interaction: {str(e)}", exc_info=True)
            logger.warning("Skipping saving interaction due to embedding error.")
            return

        try:
            point_id = str(uuid.uuid4())
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=[
                    PointStruct(
                        id=point_id,
                        vector=embedding,
                        payload={
                            "query": query,
                            "timestamp": datetime.now().isoformat(),
                            "type": "interaction",
                            "num_snippets": len(code_snippets),
                            "has_fix": bool(code_fix.get('fixed_code') or code_fix.get('changes')),
                            "fix_status": code_fix.get('status', 'unknown')
                        }
                    )
                ]
            )
            logger.info(f"Saved interaction to vector store with ID: {point_id}")
        except Exception as e:
            logger.error(f"Failed to save interaction to vector store: {str(e)}", exc_info=True)

    async def store_code(self, file_path: str, content: str, language: str, clear_existing: bool = False) -> Dict[str, Any]: # Changed default clear_existing to False
        """Store code in the vector database for future searches."""
        try:
            # Clear existing collection ONLY if explicitly requested
            if clear_existing:
                logger.warning(f"Clearing existing collection: {self.collection_name}")
                self._delete_collection()
                self._ensure_collection_exists()

            # Get embedding for the code content
            embedding = await self.generate_embedding(content)

            # Generate a UUID for the point based on file path and content hash for potential deduplication
            hash_object = hashlib.sha256(f"{file_path}::{content}".encode())
            point_id = str(uuid.UUID(hash_object.hexdigest()[:32]))

            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=[
                    PointStruct(
                        id=point_id,
                        vector=embedding,
                        payload={
                            "file_path": file_path,
                            "code": content, # Changed 'content' to 'code'
                            "language": language,
                            "type": "code_snippet",
                            "timestamp": datetime.now().isoformat()
                        }
                    )
                ]
            )

            logger.info(f"Stored/Updated code from {file_path} in Qdrant with ID: {point_id}")
            return {"status": "success", "message": f"Code from {file_path} stored successfully", "id": point_id}

        except Exception as e:
            logger.error(f"Error storing code for {file_path}: {str(e)}", exc_info=True)
            # Raise the error to be handled by the caller
            raise

    async def process_mcp_request(self, message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process a Model Context Protocol (MCP) request."""
        try:
            # Run the agent and wrap the response in MCP format
            result = await self.run(message) # Assumes run() is the main entry point

            # Add MCP-specific context handling
            mcp_context = context.copy() if context else {}
            mcp_context["last_query"] = message
            mcp_context["last_result_status"] = result.get("status")

            return {
                "result": result,
                "mcp_context": mcp_context
            }
        except Exception as e:
            logger.error(f"Error processing MCP request: {str(e)}", exc_info=True)
            return {
                "result": {
                    "error": f"Error processing request: {str(e)}",
                    "status": "error"
                },
                "mcp_context": context
            }

    async def process_query(self, query: str) -> Dict:
        """Process a user query to return relevant code snippets with explanations"""
        # This function might be less relevant if `run` is the main entry point
        # Or it could be refactored to be a part of the `run` workflow
        try:
            # 1. Assess Clarity (using the new reflector)
            clarity_assessment = await self.reflector.assess_query_clarity(query)
            if clarity_assessment != "CLEAR":
                 logger.info(f"Query assessed as potentially ambiguous: {clarity_assessment}")
                 # For this specific method, return the assessment directly
                 return {
                     "status": "clarification_needed",
                     "message": "Query may be ambiguous.",
                     "assessment": clarity_assessment,
                     "results": []
                 }

            # 2. Search for relevant code snippets
            search_results = await self.search_code(query)

            if not search_results:
                return {
                    "status": "success",
                    "message": "No relevant code snippets found",
                    "results": []
                }

            # 3. Generate explanations for each result
            processed_results = []
            for result in search_results:
                if result.get('code') and result.get('language'):
                    explanation = await self.explain_code(result['code'], result['language'])
                    result['explanation'] = explanation # Add explanation to the result dict
                    processed_results.append(result)
                else:
                     logger.warning(f"Skipping explanation for result with missing code/language: {result.get('file_path')}")

            return {
                "status": "success",
                "message": f"Found {len(processed_results)} relevant code snippets with explanations",
                "results": processed_results
            }

        except Exception as e:
            logger.error(f"Error processing query: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "message": f"Error processing query: {str(e)}",
                "results": []
            }

    async def generate_code_fix(self, query: str, code_snippets: List[Dict[str, Any]], error_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate a code fix based on the query, relevant code snippets, and optional error info"""
        # This method might be refactored or called by the main `run` method.
        # For simplicity, keeping it separate for now, but it duplicates some logic from _generate_code_fix
        # Recommendation: Consolidate fix generation logic.
        logger.warning("generate_code_fix method called - consider consolidating with _generate_code_fix or calling from run()")

        if not self.provider:
            return {"status": "error", "message": "LLM Provider not initialized."} 

        try:
            snippets_with_details = []
            for snippet in code_snippets:
                if snippet.get('code'): # Only process snippets with code
                    code_lines = snippet['code'].split('\n')
                    snippets_with_details.append({
                        'code': snippet['code'],
                        'file_path': snippet.get('file_path', 'unknown'),
                        'language': snippet.get('language', 'unknown'),
                        'similarity': snippet.get('similarity', 0),
                        'start_line': 1,
                        'end_line': len(code_lines)
                    })

            if not snippets_with_details:
                 return {
                     "status": "error",
                     "message": "No valid code snippets provided to generate a fix."
                 }

            # Prepare context (simplified for this example)
            context_limit = 2
            relevant_snippets = sorted(snippets_with_details, key=lambda x: x['similarity'], reverse=True)[:context_limit]
            context = "\n\n".join([
                f"--- Code Snippet from {snippet['file_path']} ---\n{snippet['code']}"
                for snippet in relevant_snippets
            ])

            error_context = ""
            if error_info:
                error_context = f"\n--- Error Info ---\nMessage: {error_info.get('message', 'N/A')}"

            system_prompt = "You are an expert code assistant. Analyze the query, code, and error. Suggest a fix."
            user_prompt = f"Query: {query}\n\nCode Context:\n{context}{error_context}\n\nSuggest a fix:"

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]

            fix_suggestion_text = await self.generate_completion(messages)

            return {
                "status": "success",
                "explanation": fix_suggestion_text,
                "code_snippets_used": snippets_with_details
            }

        except Exception as e:
            logger.error(f"Error in external generate_code_fix: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "message": f"Error generating code fix: {str(e)}"
            }

    async def analyze_code(self, code: str) -> str:
        """Analyze code snippets and provide insights"""
        if not self.provider:
            return "Error: LLM Provider not initialized."
        try:
            if not code or code.isspace():
                logger.warning("analyze_code received empty or whitespace-only code.")
                return "No code provided for analysis."

            messages = [
                {
                    "role": "system",
                    "content": "You are an expert code analyst. Analyze the provided code snippet(s) and provide insights about their functionality, structure, potential issues, and possible improvements. Be concise and clear."
                },
                {
                    "role": "user",
                    "content": f"Please analyze the following code and provide insights:\n\n```\n{code}\n```"
                }
            ]

            response = await self.provider.generate_completion(messages)
            return response.strip()

        except Exception as e:
            logger.error(f"Error analyzing code: {str(e)}", exc_info=True)
            return f"Error occurred while analyzing the code: {str(e)}"

    async def generate_test(self, code: str, test_type: str = "unit") -> Dict[str, Any]:
        """Generate test cases for the given code"""
        if not self.provider:
            return {"status": "error", "message": "LLM Provider not initialized."}
        try:
            if not code or code.isspace():
                 return {"status": "error", "message": "Cannot generate tests for empty code."}

            messages = [
                {
                    "role": "system",
                    "content": f"You are an expert in software testing, specializing in {test_type} tests. Generate comprehensive test cases for the provided code. Use standard testing libraries and practices for the likely language of the code."
                },
                {
                    "role": "user",
                    "content": f"Please generate {test_type} tests for this code:\n\n```\n{code}\n```\n\nInclude:\n1. Test cases covering main functionality\n2. Edge cases and error conditions (e.g., invalid inputs, boundary values)\n3. Clear test descriptions or function names\n4. Appropriate assertions to verify correct behavior. Only output the test code."
                }
            ]

            test_code = await self.generate_completion(messages)

            # Basic cleanup: remove potential markdown fences if the LLM included them
            test_code = re.sub(r"^```(?:python|py)?\s*", "", test_code, flags=re.MULTILINE)
            test_code = re.sub(r"\s*```$", "", test_code, flags=re.MULTILINE)

            return {
                "status": "success",
                "test_code": test_code.strip()
            }

        except Exception as e:
            logger.error(f"Error generating test: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "message": f"Error generating test: {str(e)}"
            }

    async def explain_code(self, code: str, language: str) -> str:
        """Generate an explanation for a code snippet"""
        if not self.provider:
             return "Error: LLM Provider not initialized."
        try:
            if not code or code.isspace():
                 return "No code provided for explanation."

            messages = [
                {
                    "role": "system",
                    "content": f"You are an expert code assistant. Analyze the {language} code snippet and provide a clear, concise explanation of what it does, suitable for someone learning the language or encountering the code for the first time."
                },
                {
                    "role": "user",
                    "content": f"Please explain this {language} code snippet:\n\n```\n{code}\n```"
                }
            ]

            explanation = await self.generate_completion(messages)
            return explanation.strip()

        except Exception as e:
            logger.error(f"Error generating explanation: {str(e)}", exc_info=True)
            return f"Error generating explanation: {str(e)}"

    def _delete_collection(self):
        """Delete the existing collection if it exists"""
        try:
            collections = self.qdrant_client.get_collections().collections
            if any(c.name == self.collection_name for c in collections):
                self.qdrant_client.delete_collection(collection_name=self.collection_name)
                logger.info(f"Deleted collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error deleting collection: {str(e)}")
            # Don't raise here, allow ensure_collection_exists to handle creation

    async def run(self, query: str, conversation_history: Optional[list[Dict[str, str]]] = None) -> Dict[str, Any]:
        """Run the agent's main workflow with a query and return results."""
        start_time = time.time()
        logger.info(f"Running agent workflow for query: '{query}'")
        if conversation_history:
            logger.info(f"Using conversation history with {len(conversation_history)} messages.")

        response_data = {
            "status": "processing",
            "query": query,
            "clarification_needed": None,
            "extracted_keywords": [], # Added field for keywords
            "decomposed_queries": [], # Added field for sub-queries
            "initial_plan": [],       # Type hint implicitly updated by assignment
            "code_snippets": [],
            "result_summary": {},
            "metadata": {}
        }

        try:
            # --- 1. Reflection Phase (Query Clarification) --- #
            clarity_assessment = await self.reflector.assess_query_clarity(query, conversation_history)
            if clarity_assessment != "CLEAR":
                logger.info(f"Query assessed as potentially ambiguous. Assessment: {clarity_assessment}")
                response_data["clarification_needed"] = clarity_assessment
                # For MVP, we log ambiguity but proceed anyway.
                # In a full implementation, might return here or ask user.
            else:
                 logger.info("Query assessed as clear.")
                 response_data["clarification_needed"] = "Query assessed as clear."

            # Extract Keywords
            extracted_keywords = await self.reflector.extract_keywords(query, conversation_history)
            response_data["extracted_keywords"] = extracted_keywords
            # MVP: We extract keywords but don't use them to modify the search yet.

            # Decompose Query
            decomposed_queries = await self.reflector.decompose_query(query, conversation_history)
            response_data["decomposed_queries"] = decomposed_queries
            # MVP: We decompose the query but run the search based on the original query for now.
            search_query = query # Use original query for MVP search

            # --- Create Initial Plan --- #
            initial_plan = await self.planner.create_initial_plan(query, extracted_keywords, decomposed_queries)
            response_data["initial_plan"] = initial_plan
            # MVP: We generate the plan but don't strictly follow it yet.

            # --- 2. Action Phase (Code Search) --- #
            # Use search_query variable (which is original query in MVP)
            code_snippets = await self.search_code(search_query)
            response_data["code_snippets"] = code_snippets

            if not code_snippets:
                logger.info("No relevant code snippets found.")
                response_data["status"] = "success"
                response_data["message"] = "No relevant code snippets found for the query."
                response_data["result_summary"] = {
                    "analysis_summary": "No code snippets to analyze.",
                    "fix_details": self._generate_error_response("No code snippets found to generate a fix.")
                }
            else:
                logger.info(f"Found {len(code_snippets)} relevant code snippets.")

                # --- 3. Action Phase (Code Analysis) --- #
                combined_code_for_analysis = "\n\n---\n\n".join([
                    f"File: {snippet['file_path']}\n```\n{snippet['code']}\n```"
                    for snippet in code_snippets if snippet.get('code')
                ])

                if not combined_code_for_analysis.strip():
                    analysis = "Could not perform analysis: Found snippets lack code content."
                    logger.warning(analysis)
                else:
                    analysis = await self.analyze_code(combined_code_for_analysis)
                    logger.info("Code analysis completed.")
                response_data["result_summary"]["analysis_summary"] = analysis

                # --- 4. Action Phase (Code Fix Generation) --- #
                # Use the internal method for structured JSON output
                code_fix = await self._generate_code_fix(query, code_snippets, analysis)
                logger.info("Code fix generation process completed.")
                response_data["result_summary"]["fix_details"] = code_fix
                response_data["status"] = code_fix.get("status", "success") # Reflect status from fix generation
                response_data["message"] = code_fix.get("explanation", "Analysis and fix generation completed.")

            # --- 5. Post-Processing (Save Interaction) --- #
            # Use the structured fix data for saving
            # Ensure fix_details exists and is a dict before trying to use it
            final_fix_data = response_data.get("result_summary", {}).get("fix_details", {})
            if not isinstance(final_fix_data, dict):
                final_fix_data = {} # Default to empty dict if invalid type

            try:
                await self._save_interaction(
                     query,
                     response_data['code_snippets'],
                     final_fix_data
                 )
            except Exception as save_e:
                 logger.error(f"Failed to save interaction: {save_e}", exc_info=True)


        except Exception as e:
            logger.error(f"Error during agent run workflow: {str(e)}", exc_info=True)
            response_data["status"] = "error"
            response_data["message"] = f"An error occurred while processing the query: {str(e)}"
            # Ensure result_summary reflects the error state
            if not response_data.get("result_summary") or response_data["result_summary"].get("fix_details", {}).get("status") != 'error':
                 response_data["result_summary"] = {
                      "analysis_summary": response_data.pop("analysis", f"Error during analysis: {str(e)}"),
                      "fix_details": self._generate_error_response(f"Workflow error occurred: {str(e)}")
                 }

        finally:
            # Add final metadata
            end_time = time.time()
            duration = end_time - start_time

            # Basic workflow success check for meta-reflection MVP
            steps_ok = [
                response_data.get("clarification_needed") is not None, # Check if assessment ran
                response_data.get("extracted_keywords") is not None, # Check if extraction ran
                response_data.get("decomposed_queries") is not None, # Check if decomposition ran
                response_data.get("code_snippets") is not None, # Check if search ran (might be empty list)
                response_data.get("result_summary") is not None and response_data["result_summary"].get("fix_details", {}).get("status") != "error" # Check fix status
            ]
            workflow_success = all(steps_ok)

            response_data["metadata"] = {
                "query": query,
                "duration_seconds": round(duration, 2),
                "timestamp": datetime.now().isoformat(),
                "provider_used": self.provider.__class__.__name__ if self.provider else 'unknown',
                "workflow_completed_successfully": workflow_success # Meta-reflection MVP
            }
            logger.info(f"Agent run finished. Status: {response_data['status']}, Duration: {duration:.2f}s, Success: {workflow_success}")

        return response_data 