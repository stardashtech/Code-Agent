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
from qdrant_client.http.models import Distance, PointStruct, VectorParams
from qdrant_client.http.models import Filter as QdrantFilter
from qdrant_client.http.models import FieldCondition, MatchValue
import hashlib
from datetime import datetime, timedelta

from app.config import settings

logger = logging.getLogger(__name__)

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
        # Initialize Ollama for embeddings
        self.ollama_provider = OllamaProvider()
        
    async def generate_completion(self, messages: List[Dict[str, str]], temperature: float = 0.7) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature
        )
        return response.choices[0].message.content
        
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embeddings using Ollama since OpenRouter doesn't support embeddings"""
        try:
            return await self.ollama_provider.generate_embedding(text)
        except Exception as e:
            logger.error(f"Error generating embedding via Ollama: {str(e)}")
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
    """Code Agent service that handles code search, analysis, and correction using LLMs."""
    
    def __init__(self):
        """Initialize the CodeAgent service with necessary clients and configuration."""
        self.executor = ThreadPoolExecutor(max_workers=settings.worker_count)
        
        # Set embedding dimension based on model
        if settings.openrouter_embedding_provider == "openai":
            settings.embedding_dimension = 1024  # OpenAI text-embedding-3-small dimension
        else:
            settings.embedding_dimension = 768   # nomic-embed-text dimension
            
        logger.info(f"Using embedding dimension: {settings.embedding_dimension}")
        
        # Initialize Redis client
        try:
            self.redis_client = redis.from_url(
                settings.redis_url,
                password=settings.redis_password,
                decode_responses=True,
                socket_timeout=5.0,
                socket_connect_timeout=5.0,
                retry_on_timeout=True,
                max_connections=10,
                health_check_interval=30,
            )
            # Test connection
            self.redis_client.ping()
            logger.info("Successfully initialized Redis client")
        except redis.ConnectionError as e:
            logger.info("Redis server is not available, continuing without caching")
            self.redis_client = None
        except Exception as e:
            logger.warning(f"Unexpected error initializing Redis client: {str(e)}")
            self.redis_client = None
        
        # Initialize LLM providers
        self.llm_providers: List[LLMProvider] = []
        
        if settings.llm_provider == "openai":
            # Initialize OpenAI provider if API key is available
            if settings.openai_api_key:
                try:
                    self.llm_providers.append(OpenAIProvider(settings.openai_api_key, settings.openai_model))
                    logger.info("Successfully initialized OpenAI provider with model: %s", settings.openai_model)
                except Exception as e:
                    logger.error(f"Failed to initialize OpenAI provider: {str(e)}")
            else:
                logger.error("OpenAI selected as provider but API key is missing")
                
        elif settings.llm_provider == "openrouter":
            # Initialize OpenRouter provider if API key is available
            if settings.openrouter_api_key:
                try:
                    self.llm_providers.append(OpenRouterProvider(
                        settings.openrouter_api_key,
                        settings.openrouter_model
                    ))
                    logger.info("Successfully initialized OpenRouter provider with model: %s", settings.openrouter_model)
                except Exception as e:
                    logger.error(f"Failed to initialize OpenRouter provider: {str(e)}")
            else:
                logger.error("OpenRouter selected as provider but API key is missing")
                
        elif settings.llm_provider == "ollama":
            # Initialize Ollama provider
            try:
                self.llm_providers.append(OllamaProvider())
                logger.info("Successfully initialized Ollama provider with model: %s and embedding model: %s",
                           settings.ollama_model, settings.ollama_embedding_model)
            except Exception as e:
                logger.error(f"Failed to initialize Ollama provider: {str(e)}")
                
        elif settings.llm_provider == "vllm":
            # Initialize vLLM provider
            try:
                self.llm_providers.append(VLLMProvider())
                logger.info("Successfully initialized vLLM provider with model: %s and embedding model: %s",
                           settings.vllm_model, settings.vllm_embedding_model)
            except Exception as e:
                logger.error(f"Failed to initialize vLLM provider: {str(e)}")
        else:
            logger.error(f"Unknown LLM provider: {settings.llm_provider}")
            
        # Ensure at least one provider is available
        if not self.llm_providers:
            raise ValueError(f"No LLM providers available for selected provider: {settings.llm_provider}")
            
        # Initialize code search providers
        self.search_providers: List[CodeSearchProvider] = []
        
        # Initialize GitHub search
        if settings.github_token:
            self.search_providers.append(GitHubSearchProvider(settings.github_token))
            
        # Initialize Stack Overflow search
        if settings.stackoverflow_key:
            self.search_providers.append(StackOverflowSearchProvider(settings.stackoverflow_key))
            
        # Initialize vector store
        try:
            if not settings.qdrant_url:
                raise ValueError("Qdrant URL is required")
                
            logger.info(f"Connecting to Qdrant at: {settings.qdrant_url}")
            self.qdrant_client = QdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key or None,
                timeout=10
            )
            self.collection_name = self._get_collection_name()
            self._ensure_collection_exists()
            logger.info("Successfully initialized Qdrant client")
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant client: {str(e)}", exc_info=True)
            raise
        
        # Log successful initialization
        provider_info = f"OpenAI ({settings.openai_model})" if settings.llm_provider == "openai" else f"Ollama ({settings.ollama_model})"
        logger.info("CodeAgent initialized with provider: %s and collection: %s", 
                   provider_info, self.collection_name)

    def _get_collection_name(self) -> str:
        """Get the collection name based on the current provider and embedding dimension."""
        base_name = settings.qdrant_collection
        provider = settings.llm_provider
        dim = settings.embedding_dimension
        return f"{base_name}_{provider}_{dim}"

    def get_collection_name(self) -> str:
        """Public method to get the current collection name."""
        return self._get_collection_name()

    def _delete_collection(self) -> None:
        """Delete the current collection if it exists."""
        try:
            collections = self.qdrant_client.get_collections()
            if self.collection_name in [c.name for c in collections.collections]:
                logger.info(f"Deleting collection: {self.collection_name}")
                self.qdrant_client.delete_collection(self.collection_name)
                logger.info(f"Successfully deleted collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error deleting collection: {str(e)}")
            raise

    def _ensure_collection_exists(self) -> None:
        """Ensure the vector collection exists with correct parameters"""
        try:
            # List all collections
            collections = self.qdrant_client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            # Delete all code_snippets collections
            for name in collection_names:
                if name.startswith("code_snippets"):
                    logger.info(f"Deleting collection: {name}")
                    self.qdrant_client.delete_collection(name)
            
            # Create new collection
            logger.info(f"Creating new Qdrant collection: {self.collection_name}")
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=settings.embedding_dimension,
                    distance=Distance.COSINE
                )
            )
            logger.info("Successfully created Qdrant collection")
            
        except Exception as e:
            logger.error(f"Error ensuring collection exists: {str(e)}")
            raise

    async def run(self, query: str) -> Dict[str, Any]:
        """Run the agent on the given query asynchronously."""
        logger.info("Processing query: %s", query)
        
        # Delegate CPU-bound operations to the thread pool
        return await asyncio.get_event_loop().run_in_executor(
            self.executor, self._run_sync, query
        )

    def _run_sync(self, query: str) -> Dict[str, Any]:
        """Synchronous implementation of the agent run logic."""
        try:
            # 1. Analyze the query
            analysis = self._analyze_query(query)
            
            # Check if analysis has an error
            if "error_type" in analysis and analysis["error_type"] in ["initialization_error", "api_error"]:
                return {
                    "analysis": analysis,
                    "code_snippets": [],
                    "code_fix": {
                        "explanation": f"Cannot generate code fix: {analysis.get('description', 'Unknown error')}",
                        "fixed_code": "",
                        "file_path": "",
                        "changes": []
                    }
                }
            
            # 2. Search for relevant code
            code_snippets = self._search_code(query, analysis.get("search_terms", []))
            
            # 3. Generate fix with LLM
            if not code_snippets:
                logger.warning("No code snippets found for query: %s", query)
                return {
                    "analysis": analysis,
                    "code_snippets": [],
                    "code_fix": {
                        "explanation": "No relevant code found for the query",
                        "fixed_code": "",
                        "file_path": "",
                        "changes": []
                    }
                }
            
            code_fix = self._generate_code_fix(query, code_snippets, analysis)
            
            # 4. Save the interaction
            try:
                self._save_interaction(query, code_snippets, code_fix)
            except Exception as e:
                logger.error(f"Error saving interaction: {str(e)}")
                # Continue even if saving fails
            
            return {
                "analysis": analysis,
                "code_snippets": code_snippets,
                "code_fix": code_fix
            }
        except Exception as e:
            logger.error("Error processing query: %s", str(e), exc_info=True)
            # Return a structured response even in case of error
            return {
                "analysis": {
                    "error_type": "processing_error",
                    "description": f"Error processing query: {str(e)}"
                },
                "code_snippets": [],
                "code_fix": {
                    "explanation": f"Error during processing: {str(e)}",
                    "fixed_code": "",
                    "file_path": "",
                    "changes": []
                }
            }

    def _analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze the query to understand the problem and extract search terms."""
        system_prompt = """You are an elite senior software architect and security expert with extensive experience in multiple programming languages and frameworks. Your expertise includes:

1. Security vulnerability detection and remediation
2. Code quality assessment and improvement
3. Performance optimization
4. Design patterns and best practices
5. Language-specific idioms and conventions

For each query, provide a comprehensive analysis including:
1. Problem categorization and severity assessment
2. Potential security implications
3. Performance considerations
4. Best practices violations
5. Language-specific concerns
6. Architectural impact

Always respond in valid JSON format with no markdown or code blocks."""

        analysis_prompt = f"""Analyze the following code-related query with a security-first mindset:

Query: {query}

Provide a detailed analysis in JSON format with the following structure:

{{
    "error_type": "Specific error category or problem type",
    "severity": "high|medium|low",
    "security_impact": {{
        "level": "critical|high|medium|low|none",
        "vulnerabilities": ["List of potential security issues"],
        "mitigations": ["Recommended security measures"]
    }},
    "search_terms": ["Key terms for finding relevant code"],
    "file_types": ["Relevant file extensions"],
    "language": "Primary programming language",
    "frameworks": ["Relevant frameworks"],
    "performance_impact": {{
        "level": "high|medium|low|none",
        "concerns": ["List of performance considerations"]
    }},
    "best_practices": {{
        "violations": ["Potential violations of best practices"],
        "recommendations": ["Recommended improvements"]
    }},
    "description": "Detailed problem description",
    "metadata": {{
        "complexity": "high|medium|low",
        "scope": "security|performance|functionality|maintenance",
        "priority": "high|medium|low"
    }}
}}

Focus on security implications and best practices while maintaining high code quality standards."""
        
        try:
            # Try each provider in sequence until one succeeds
            for provider in self.llm_providers:
                try:
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": analysis_prompt}
                    ]
                    content = asyncio.run(provider.generate_completion(messages, temperature=0.2))
                    
                    logger.debug("Raw analysis response: %s", content)
                    
                    try:
                        analysis = json.loads(content)
                        
                        # Validate and ensure required fields
                        required_fields = {
                            'error_type': str,
                            'severity': str,
                            'security_impact': dict,
                            'search_terms': list,
                            'file_types': list,
                            'language': str,
                            'performance_impact': dict,
                            'best_practices': dict,
                            'description': str,
                            'metadata': dict
                        }
                        
                        # Initialize missing fields with default values
                        for field, field_type in required_fields.items():
                            if field not in analysis:
                                if field_type == str:
                                    analysis[field] = ""
                                elif field_type == list:
                                    analysis[field] = []
                                elif field_type == dict:
                                    analysis[field] = {}
                        
                        # Add quality score based on analysis completeness
                        total_fields = len(required_fields)
                        filled_fields = sum(1 for f in required_fields if analysis.get(f))
                        analysis['quality_score'] = filled_fields / total_fields
                        
                        logger.info("Query analysis completed with quality score: %.2f", analysis['quality_score'])
                        return analysis
                        
                    except json.JSONDecodeError as je:
                        logger.error("JSON parsing error at position %d: %s", je.pos, je.msg)
                        logger.error("Problematic content: %s", content[max(0, je.pos-50):min(len(content), je.pos+50)])
                        continue
                        
                except Exception as e:
                    logger.error(f"Error with provider {provider.__class__.__name__}: {str(e)}")
                    continue
            
            # All providers failed
            return {
                "error_type": "provider_error",
                "severity": "low",
                "security_impact": {"level": "none", "vulnerabilities": [], "mitigations": []},
                "search_terms": [query],
                "file_types": [],
                "language": "unknown",
                "performance_impact": {"level": "none", "concerns": []},
                "best_practices": {"violations": [], "recommendations": []},
                "description": "All providers failed to analyze query",
                "metadata": {"complexity": "low", "scope": "functionality", "priority": "low"},
                "quality_score": 0.0
            }
            
        except Exception as e:
            logger.error("Error analyzing query: %s", str(e), exc_info=True)
            return {
                "error_type": "analysis_error",
                "severity": "low",
                "security_impact": {"level": "none", "vulnerabilities": [], "mitigations": []},
                "search_terms": [query],
                "file_types": [],
                "language": "unknown",
                "performance_impact": {"level": "none", "concerns": []},
                "best_practices": {"violations": [], "recommendations": []},
                "description": f"Error during analysis: {str(e)}",
                "metadata": {"complexity": "low", "scope": "functionality", "priority": "low"},
                "quality_score": 0.0
            }

    def _search_code(self, query: str, search_terms: List[str]) -> List[Dict[str, Any]]:
        """Search for relevant code snippets in the vector database."""
        try:
            # 1. Get vector embedding for the query
            embedding = self._get_embedding(query)
            
            # Log search parameters at INFO level
            logger.info("Searching for code snippets - Query: '%s', Terms: %s", 
                       query, ', '.join(search_terms))
            
            # 2. Perform the search using query_vector with lower threshold
            search_results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=embedding,
                limit=10,  # Increased limit to get more candidates
                with_payload=True,
                with_vectors=False,
                score_threshold=0.3  # Lowered threshold for more results
            )
            
            # Log initial results count
            logger.info("Found %d initial results", len(search_results))
            
            # 3. Format and filter results with quality checks
            code_snippets = []
            for result in search_results:
                # Skip invalid results
                if not result.payload or 'content' not in result.payload:
                    logger.debug("Skipping result with missing payload or content")
                    continue
                
                content = result.payload.get('content', '').strip()
                
                # Skip empty or very short content
                if not content or len(content.split()) < 3:
                    logger.debug("Skipping result with insufficient content: %s", 
                               result.payload.get('file_path'))
                    continue
                
                # Basic content quality check
                if content.count('\n') < 2 and len(content) < 50:
                    logger.debug("Skipping low quality content: %s", 
                               result.payload.get('file_path'))
                    continue
                
                snippet = {
                    'file_path': result.payload.get('file_path', 'unknown'),
                    'content': content,
                    'language': result.payload.get('language', ''),
                    'score': result.score,
                    'type': result.payload.get('type', 'code_snippet'),
                    'lines': content.count('\n') + 1,
                    'chars': len(content)
                }
                code_snippets.append(snippet)
            
            # Sort by score but boost longer, more complete snippets
            def snippet_score(s):
                # Combine relevance score with content quality metrics
                base_score = s['score']
                length_boost = min(s['lines'] / 10, 1.0)  # Boost for longer snippets
                return base_score * (1 + length_boost * 0.2)  # 20% max boost for length
            
            sorted_snippets = sorted(code_snippets, key=snippet_score, reverse=True)
            
            # Take top 5 after quality sorting
            final_snippets = sorted_snippets[:5]
            
            logger.info("Returning %d quality code snippets (from %d candidates)", 
                       len(final_snippets), len(code_snippets))
            
            # Log detailed snippet info at debug level
            for idx, snippet in enumerate(final_snippets, 1):
                logger.debug("Snippet %d: %s (score: %.3f, lines: %d)", 
                           idx, snippet['file_path'], snippet['score'], snippet['lines'])
            
            return final_snippets
            
        except Exception as e:
            logger.error("Error searching code: %s", str(e), exc_info=True)
            return []  # Return empty list on error

    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for text, using Redis cache if available."""
        try:
            # Create cache key using SHA-256
            cache_key = f"embedding:{hashlib.sha256(text.encode()).hexdigest()}"
            
            # Try to get from cache if Redis is available
            if self.redis_client is not None:
                try:
                    cached = self.redis_client.get(cache_key)
                    if cached:
                        logger.debug("Cache hit for embedding")
                        embedding = json.loads(cached)
                        # Ensure cached embedding has correct dimension
                        if len(embedding) != settings.embedding_dimension:
                            logger.warning(f"Cached embedding has wrong dimension {len(embedding)}, clearing cache")
                            self.redis_client.delete(cache_key)
                        else:
                            return embedding
                    logger.debug("Cache miss for embedding")
                except redis.RedisError as e:
                    logger.warning(f"Redis error while getting embedding: {str(e)}")
                    # Continue with provider call on Redis error
            
            # Get embedding from provider
            for provider in self.llm_providers:
                try:
                    embedding = asyncio.run(provider.generate_embedding(text))
                    
                    # Verify embedding dimension
                    if len(embedding) != settings.embedding_dimension:
                        logger.error(f"Provider {provider.__class__.__name__} returned wrong embedding dimension: got {len(embedding)}, expected {settings.embedding_dimension}")
                        continue
                    
                    # Cache the result if Redis is available
                    if self.redis_client is not None:
                        try:
                            self.redis_client.setex(
                                cache_key,
                                timedelta(hours=24),
                                json.dumps(embedding)
                            )
                        except redis.RedisError as e:
                            logger.warning(f"Redis error while caching embedding: {str(e)}")
                            # Continue even if caching fails
                            
                    return embedding
                except Exception as e:
                    logger.error(f"Error getting embedding from provider {provider.__class__.__name__}: {str(e)}")
                    continue
                    
            # All providers failed, return zero vector of correct dimension
            logger.error("All providers failed to generate embedding")
            return [0.0] * settings.embedding_dimension
            
        except Exception as e:
            logger.error(f"Unexpected error in _get_embedding: {str(e)}")
            return [0.0] * settings.embedding_dimension

    def _generate_code_fix(self, query: str, code_snippets: List[Dict[str, Any]], 
                          analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate code fix using LLM based on the query and found code snippets."""
        if not code_snippets:
            logger.warning("No code snippets provided for fix generation")
            return self._generate_error_response("No code snippets available for analysis")
            
        # Prepare context from code snippets (limit to most relevant)
        relevant_snippets = sorted(code_snippets, key=lambda x: x.get('score', 0), reverse=True)[:2]
        code_context = "\n\n".join([
            f"File: {snippet.get('file_path', 'unknown')}\n```{snippet.get('language', '')}\n{snippet.get('content', '')}\n```"
            for snippet in relevant_snippets
        ])
        
        system_prompt = """You are an elite senior software architect and security expert. Your task is to analyze code and provide fixes in a specific JSON format.

IMPORTANT: 
1. Your response must be ONLY valid JSON, with no additional text or markdown
2. All code in the response must be properly escaped for JSON
3. All strings must use double quotes, not single quotes
4. Ensure all code is syntactically correct before including it

Example response format:
{
    "explanation": "Brief explanation of changes",
    "fixed_code": "def example():\\n    return True",
    "file_path": "path/to/file",
    "changes": [
        {
            "line_number": 1,
            "original": "old code",
            "replacement": "new code",
            "explanation": "why changed"
        }
    ],
    "security_review": {
        "vulnerabilities_fixed": ["list", "of", "fixes"],
        "security_improvements": ["list", "of", "improvements"],
        "risk_level": "high|medium|low"
    }
}"""

        fix_prompt = f"""Analyze and fix this code:

Query: {query}
Analysis: {json.dumps(analysis, indent=2)}
Context: {code_context}

Requirements:
1. Response must be ONLY a JSON object
2. All code must be properly escaped for JSON
3. Use double quotes for strings
4. Ensure all code is syntactically valid
5. Include complete implementation

Fields required in response:
1. explanation: Brief explanation of changes
2. fixed_code: Complete fixed code (properly escaped)
3. file_path: Path to file being modified
4. changes: Array of specific changes
5. security_review: Security analysis object

DO NOT include any text outside the JSON structure."""
        
        try:
            # Try each provider in sequence until one succeeds
            for provider in self.llm_providers:
                try:
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": fix_prompt}
                    ]
                    content = asyncio.run(provider.generate_completion(messages, temperature=0.2))
                    
                    logger.debug("Raw code fix response: %s", content)
                    
                    try:
                        # Clean and validate the response
                        content = self._clean_json_response(content)
                        
                        # Parse the cleaned JSON
                        code_fix = json.loads(content)
                        
                        # Validate and fix the code fix
                        code_fix = self._validate_and_fix_response(code_fix)
                        
                        logger.info("Generated code fix with quality score: %.2f", code_fix.get('quality_score', 0))
                        return code_fix
                        
                    except json.JSONDecodeError as je:
                        logger.error("JSON parsing error in code fix at position %d: %s", je.pos, je.msg)
                        logger.debug("Problematic content: %s", content[max(0, je.pos-50):min(len(content), je.pos+50)])
                        continue
                except Exception as e:
                    logger.error(f"Error with provider {provider.__class__.__name__}: {str(e)}")
                    continue
            
            # All providers failed
            return self._generate_error_response("All providers failed to generate code fix")
            
        except Exception as e:
            logger.error("Error generating code fix: %s", str(e), exc_info=True)
            return self._generate_error_response(f"Error generating code fix: {str(e)}")

    def _clean_json_response(self, content: str) -> str:
        """Clean and prepare the response for JSON parsing."""
        # Remove any markdown code blocks
        content = content.replace("```json", "").replace("```", "")
        
        # Remove any leading/trailing whitespace
        content = content.strip()
        
        # Remove any non-JSON content before the first { or after the last }
        start = content.find("{")
        end = content.rfind("}") + 1
        if start >= 0 and end > start:
            content = content[start:end]
            
        # Replace any single quotes with double quotes (only outside of code blocks)
        in_code = False
        cleaned = []
        for char in content:
            if char == '`':
                in_code = not in_code
            elif char == "'" and not in_code:
                char = '"'
            cleaned.append(char)
        content = ''.join(cleaned)
        
        return content

    def _validate_and_fix_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and fix the response structure."""
        # Required fields with their default values
        required_fields = {
            'explanation': '',
            'fixed_code': '',
            'file_path': '',
            'changes': [],
            'security_review': {
                'vulnerabilities_fixed': [],
                'security_improvements': [],
                'risk_level': 'unknown'
            },
            'quality_metrics': {
                'maintainability': 'unknown',
                'complexity': 'unknown',
                'test_coverage_impact': 'unknown'
            }
        }
        
        # Ensure all required fields exist
        for field, default in required_fields.items():
            if field not in response:
                response[field] = default
            elif isinstance(default, dict) and isinstance(response[field], dict):
                # Ensure nested fields exist
                for nested_field, nested_default in default.items():
                    if nested_field not in response[field]:
                        response[field][nested_field] = nested_default
        
        # Validate code syntax if possible
        if response['fixed_code']:
            try:
                compile(response['fixed_code'], '<string>', 'exec')
            except SyntaxError as e:
                logger.warning(f"Syntax error in generated code: {str(e)}")
                response['syntax_check'] = {
                    'status': 'failed',
                    'error': str(e)
                }
            else:
                response['syntax_check'] = {
                    'status': 'passed'
                }
        
        # Calculate quality score
        total_fields = len(required_fields)
        filled_fields = sum(1 for f in required_fields if response.get(f))
        response['quality_score'] = filled_fields / total_fields
        
        return response

    def _generate_error_response(self, error_message: str) -> Dict[str, Any]:
        """Generate a standardized error response for code fix failures."""
        return {
            "explanation": error_message,
            "fixed_code": "",
            "file_path": "",
            "changes": [],
            "security_review": {
                "vulnerabilities_fixed": [],
                "security_improvements": [],
                "risk_level": "unknown"
            },
            "quality_metrics": {
                "maintainability": "unknown",
                "complexity": "unknown",
                "test_coverage_impact": "unknown"
            },
            "performance_impact": {
                "description": "Unable to analyze performance impact",
                "recommendations": []
            },
            "testing_recommendations": [],
            "documentation_updates": [],
            "quality_score": 0.0
        }

    def _save_interaction(self, query: str, code_snippets: List[Dict[str, Any]], 
                         code_fix: Dict[str, Any]) -> None:
        """Save the interaction details to the vector store for future reference."""
        # Create a more concise text of the interaction for embedding
        interaction_text = f"""
        Query: {query}
        
        Code Snippets: {len(code_snippets)} found
        {', '.join(s.get('file_path', 'unknown') for s in code_snippets)}
        
        Solution:
        {code_fix.get('explanation', 'No explanation provided')}
        """
        
        # Get embedding for the interaction
        try:
            embedding = self._get_embedding(interaction_text)
            
            # Generate a UUID for the point
            point_id = str(uuid.uuid4())
            
            # Save minimal data to Qdrant
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=[
                    PointStruct(
                        id=point_id,
                        vector=embedding,
                        payload={
                            "query": query,
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                            "type": "interaction",
                            "num_snippets": len(code_snippets),
                            "has_fix": bool(code_fix.get('fixed_code'))
                        }
                    )
                ]
            )
            logger.info(f"Saved interaction to vector store with ID: {point_id}")
        except Exception as e:
            logger.error(f"Failed to save interaction to vector store: {str(e)}")
            raise

    async def store_code(self, file_path: str, content: str, language: str, clear_existing: bool = True) -> Dict[str, Any]:
        """Store code in the vector database for future searches."""
        return await asyncio.get_event_loop().run_in_executor(
            self.executor, self._store_code_sync, file_path, content, language, clear_existing
        )

    def _store_code_sync(self, file_path: str, content: str, language: str, clear_existing: bool = True) -> Dict[str, Any]:
        """Synchronous implementation of code storage."""
        try:
            # Clear existing collection if requested
            if clear_existing:
                self._delete_collection()
                self._ensure_collection_exists()
            
            # Get embedding for the code content
            embedding = self._get_embedding(content)
            
            # Generate a UUID for the point
            point_id = str(uuid.uuid4())
            
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=[
                    PointStruct(
                        id=point_id,
                        vector=embedding,
                        payload={
                            "file_path": file_path,
                            "content": content,
                            "language": language,
                            "type": "code_snippet"
                        }
                    )
                ]
            )
            
            logger.info(f"Stored code from {file_path} to Qdrant with ID: {point_id}")
            return {"status": "success", "message": f"Code from {file_path} stored successfully"}
            
        except Exception as e:
            logger.error("Error storing code: %s", str(e), exc_info=True)
            raise

    async def process_mcp_request(self, message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process a Model Context Protocol (MCP) request."""
        try:
            # Run the agent and wrap the response in MCP format
            result = await self.run(message)
            
            # Add MCP-specific context handling
            mcp_context = context.copy() if context else {}
            mcp_context["last_query"] = message
            
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