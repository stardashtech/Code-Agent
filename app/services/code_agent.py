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
        self.provider = None
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
            self.provider = OllamaProvider()
            
        if not self.provider:
            raise ValueError("No valid LLM provider could be initialized")
            
        # Initialize Qdrant client
        self.qdrant_client = QdrantClient(
            host=config.qdrant_host,
            port=config.qdrant_port
        )
        
        # Initialize Redis client
        self.redis_client = redis.Redis(
            host=config.redis_host,
            port=config.redis_port,
            decode_responses=True
        )
        
        # Set collection name
        self.collection_name = self._get_collection_name()
        
        # Ensure collection exists
        self._ensure_collection_exists()
        
        logger.info("CodeAgent initialized successfully")
        
    async def generate_completion(self, messages: List[Dict[str, str]], temperature: float = 0.7) -> str:
        """Generate completion using the configured provider"""
        try:
            return await self.provider.generate_completion(messages, temperature)
        except Exception as e:
            logger.error(f"Error generating completion: {str(e)}")
            raise
            
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embeddings using the configured provider"""
        try:
            return await self.provider.generate_embedding(text)
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise

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
            query_embedding = await self._get_embedding(query)
            
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
                processed_results.append({
                    'code': payload.get('code', ''),
                    'file_path': payload.get('file_path', ''),
                    'language': payload.get('language', 'unknown'),
                    'similarity': match.score
                })
                
            return processed_results
            
        except Exception as e:
            logger.error(f"Error searching code: {str(e)}")
            return []
            
    async def index_code(self, code: str, metadata: Dict) -> None:
        """Index a code snippet with its embedding"""
        try:
            # Generate embedding for code
            code_embedding = await self._get_embedding(code)
            
            # Add document to vector store
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=[
                    models.PointStruct(
                        id=str(uuid.uuid4()),
                        vector=code_embedding,
                        payload={
                            "code": code,
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

    async def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for text using the configured provider"""
        try:
            return await self.generate_embedding(text)
        except Exception as e:
            logger.error(f"Error getting embedding: {str(e)}")
            raise

    def _generate_code_fix(self, query: str, code_snippets: List[Dict[str, Any]], 
                          analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate code fix using LLM based on the query and found code snippets."""
        if not code_snippets:
            logger.warning("No code snippets provided for fix generation")
            return self._generate_error_response("No code snippets available for analysis")
            
        # Prepare context from code snippets (limit to most relevant)
        relevant_snippets = sorted(code_snippets, key=lambda x: x.get('similarity', 0), reverse=True)[:2]
        code_context = "\n\n".join([
            f"File: {snippet.get('file_path', 'unknown')}\\n{snippet.get('code', '')}"
            for snippet in relevant_snippets
        ])
        
        system_prompt = """You are an expert code reviewer and fixer. Your task is to analyze code and provide fixes in a structured format.

IMPORTANT RESPONSE RULES:
1. Respond ONLY with a valid JSON object
2. DO NOT use markdown formatting
3. DO NOT use code blocks
4. DO NOT include any explanatory text outside the JSON
5. Ensure all strings are properly escaped
6. Use double quotes for all keys and string values
7. Use "\\n" for newlines in code

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
    ]
}"""

        fix_prompt = f"""Review and suggest improvements for this code:

Query: {query}

Code Context:
{code_context}

STRICT RESPONSE RULES:
1. Return ONLY a JSON object
2. NO markdown
3. NO code blocks
4. NO explanatory text
5. Use the exact format from the system prompt
6. Ensure all JSON is valid and properly escaped
7. Use double quotes for all strings
8. Use "\\n" for newlines in code
9. Include specific line-by-line changes in the "changes" array"""
        
        try:
            # Try each provider in sequence until one succeeds
            last_error = None
            for provider in self.llm_providers:
                try:
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": fix_prompt}
                    ]
                    
                    logger.debug(f"Trying code fix generation with provider: {provider.__class__.__name__}")
                    content = asyncio.run(provider.generate_completion(messages, temperature=0.2))
                    
                    # Clean and validate the response
                    cleaned_content = self._clean_json_response(content)
                    if not cleaned_content:
                        logger.warning(f"Provider {provider.__class__.__name__} returned empty response")
                        continue
                        
                    try:
                        code_fix = json.loads(cleaned_content)
                    except json.JSONDecodeError as je:
                        logger.error(f"JSON parsing error with {provider.__class__.__name__}: {str(je)}")
                        logger.debug(f"Problematic content: {cleaned_content[:200]}...")
                        last_error = je
                        continue
                        
                    # Validate and fix the response
                    code_fix = self._validate_and_fix_response(code_fix)
                    
                    # Check if code is syntactically valid
                    if code_fix.get('fixed_code'):
                        try:
                            compile(code_fix['fixed_code'], '<string>', 'exec')
                        except SyntaxError as se:
                            logger.warning(f"Syntax error in generated code: {str(se)}")
                            code_fix['syntax_check'] = {'status': 'failed', 'error': str(se)}
                        else:
                            code_fix['syntax_check'] = {'status': 'passed'}
                            
                    logger.info(f"Successfully generated code fix with {provider.__class__.__name__}")
                    return code_fix
                    
                except Exception as e:
                    logger.error(f"Error with provider {provider.__class__.__name__}: {str(e)}")
                    last_error = e
                    continue
            
            # All providers failed
            error_msg = f"All providers failed to generate code fix. Last error: {str(last_error)}"
            logger.error(error_msg)
            return self._generate_error_response(error_msg)
            
        except Exception as e:
            error_msg = f"Unexpected error generating code fix: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return self._generate_error_response(error_msg)

    def _clean_json_response(self, response: str) -> str:
        """Clean and prepare the response for JSON parsing."""
        if not response:
            logger.error("Empty response received")
            return "{}"
            
        try:
            # First try to find JSON content within code blocks
            code_block_pattern = r"```(?:json)?\s*(.*?)```"
            matches = re.findall(code_block_pattern, response, re.DOTALL)
            
            if matches:
                # Try each code block until we find valid JSON
                for block in matches:
                    try:
                        cleaned = block.strip()
                        json.loads(cleaned)  # Validate JSON
                        return cleaned
                    except json.JSONDecodeError:
                        continue
            
            # If no valid JSON in code blocks, try to extract JSON from the raw response
            # Find the outermost JSON object
            json_pattern = r"\{(?:[^{}]|(?R))*\}"
            matches = re.findall(json_pattern, response, re.DOTALL)
            
            if matches:
                for potential_json in matches:
                    try:
                        cleaned = potential_json.strip()
                        json.loads(cleaned)  # Validate JSON
                        return cleaned
                    except json.JSONDecodeError:
                        continue
            
            # If still no valid JSON, try to clean up the response
            response = response.strip()
            start = response.find('{')
            end = response.rfind('}')
            
            if start != -1 and end != -1:
                response = response[start:end + 1]
                # Clean up common issues
                response = re.sub(r'\\n', r'\\\\n', response)  # Fix newline escaping
                response = re.sub(r'(?<!\\)"', r'\"', response)  # Escape unescaped quotes
                response = re.sub(r',(\s*})', r'\1', response)  # Remove trailing commas
                
                try:
                    json.loads(response)  # Validate final JSON
                    return response
                except json.JSONDecodeError:
                    pass
            
            logger.error("Could not extract valid JSON from response")
            return "{}"
            
        except Exception as e:
            logger.error(f"Error cleaning JSON response: {str(e)}")
            return "{}"

    def _validate_and_fix_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and fix the response structure, ensuring all required fields are present."""
        if not isinstance(response, dict):
            logger.error("Response is not a dictionary")
            return self._generate_error_response("Invalid response format")
            
        # Required fields
        required_fields = ['explanation', 'fixed_code', 'file_path', 'changes']
        
        # Check and set defaults for missing fields
        for field in required_fields:
            if field not in response:
                logger.warning(f"Missing required field: {field}")
                response[field] = "" if field in ['explanation', 'fixed_code', 'file_path'] else []
                
        # Ensure changes is a list
        if not isinstance(response['changes'], list):
            logger.warning("Changes field is not a list")
            response['changes'] = []
            
        # Calculate quality score based on field completeness
        filled_fields = sum(1 for field in required_fields if response.get(field))
        response['quality_score'] = filled_fields / len(required_fields)
        
        # Add metadata
        response['metadata'] = {
            'timestamp': datetime.now().isoformat(),
            'provider': self.llm_providers[0].__class__.__name__ if self.llm_providers else 'unknown'
        }
        
        return response

    def _generate_error_response(self, error_message: str) -> Dict[str, Any]:
        """Generate a standardized error response."""
        return {
            'explanation': f"Error: {error_message}",
            'fixed_code': '',
            'file_path': '',
            'changes': [],
            'error': {
                'message': error_message,
                'timestamp': datetime.now().isoformat()
            },
            'quality_score': 0.0,
            'metadata': {
                'status': 'error',
                'provider': self.llm_providers[0].__class__.__name__ if self.llm_providers else 'unknown'
            }
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
        try:
            # Clear existing collection if requested
            if clear_existing:
                self._delete_collection()
                self._ensure_collection_exists()
            
            # Get embedding for the code content
            embedding = await self._get_embedding(content)
            
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
                            "code": content,
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

    async def process_query(self, query: str) -> Dict:
        """Process a user query to return relevant code snippets with explanations"""
        try:
            # Search for relevant code snippets
            search_results = await self.search_code(query)
            
            if not search_results:
                return {
                    "status": "success",
                    "message": "No relevant code snippets found",
                    "results": []
                }
            
            # Generate explanations for each result
            processed_results = []
            for result in search_results:
                # Generate explanation for the code
                explanation = await self.explain_code(result['code'], result['language'])
                
                processed_results.append({
                    "score": result['similarity'],
                    "file_path": result['file_path'],
                    "code": result['code'],
                    "language": result['language'],
                    "explanation": explanation
                })
            
            return {
                "status": "success",
                "message": f"Found {len(processed_results)} relevant code snippets",
                "results": processed_results
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {
                "status": "error",
                "message": f"Error processing query: {str(e)}",
                "results": []
            }

    async def generate_code_fix(self, query: str, code_snippets: List[Dict[str, Any]], error_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate a code fix based on the query, relevant code snippets, and optional error info"""
        try:
            # Add line numbers to code snippets
            snippets_with_lines = []
            for snippet in code_snippets:
                code_lines = snippet['code'].split('\n')
                snippets_with_lines.append({
                    'code': snippet['code'],
                    'file_path': snippet['file_path'],
                    'language': snippet['language'],
                    'start_line': 1,
                    'end_line': len(code_lines),
                    'similarity': snippet.get('similarity', 0)
                })
            
            # Prepare context from code snippets
            context = "\n\n".join([
                f"Code snippet {i+1}:\n{snippet['code']}\nFile: {snippet['file_path']}\nLines: {snippet['start_line']}-{snippet['end_line']}"
                for i, snippet in enumerate(snippets_with_lines)
            ])
            
            # Prepare error context if available
            error_context = ""
            if error_info:
                error_context = f"\nError message: {error_info.get('message', '')}\nError type: {error_info.get('type', '')}\nStack trace: {error_info.get('stack_trace', '')}"
            
            messages = [
                {
                    "role": "system",
                    "content": "You are an expert code assistant. Analyze the code snippets and error information, then suggest a fix that addresses the user's query."
                },
                {
                    "role": "user",
                    "content": f"Query: {query}\n\nRelevant code:\n{context}{error_context}\n\nPlease suggest a code fix that addresses the query and any errors."
                }
            ]
            
            fix_suggestion = await self.generate_completion(messages)
            
            return {
                "status": "success",
                "explanation": fix_suggestion,
                "code_snippets": snippets_with_lines
            }
            
        except Exception as e:
            logger.error(f"Error generating code fix: {str(e)}")
            return {
                "status": "error",
                "message": f"Error generating code fix: {str(e)}"
            }
            
    async def analyze_code(self, code: str) -> str:
        """Analyze code snippets and provide insights"""
        try:
            # Construct message for code analysis
            messages = [
                {
                    "role": "system",
                    "content": "You are an expert code analyst. Analyze the provided code snippets and provide insights about their functionality, patterns, and potential improvements."
                },
                {
                    "role": "user", 
                    "content": f"Please analyze the following code and provide insights:\n\n{code}"
                }
            ]
            
            # Generate analysis using LLM
            response = await self.provider.generate_completion(messages)
            return response
            
        except Exception as e:
            logger.error(f"Error analyzing code: {str(e)}")
            return "Error occurred while analyzing the code."

    async def generate_test(self, code: str, test_type: str = "unit") -> Dict[str, Any]:
        """Generate test cases for the given code"""
        try:
            messages = [
                {
                    "role": "system",
                    "content": f"You are an expert in {test_type} testing. Generate comprehensive test cases for the provided code."
                },
                {
                    "role": "user",
                    "content": f"Please generate {test_type} tests for this code:\n\n{code}\n\nInclude:\n1. Test cases covering main functionality\n2. Edge cases and error conditions\n3. Clear test descriptions and assertions"
                }
            ]
            
            test_code = await self.generate_completion(messages)
            
            return {
                "status": "success",
                "test_code": test_code
            }
            
        except Exception as e:
            logger.error(f"Error generating test: {str(e)}")
            return {
                "status": "error",
                "message": f"Error generating test: {str(e)}"
            }
            
    async def explain_code(self, code: str, language: str) -> str:
        """Generate an explanation for a code snippet"""
        try:
            messages = [
                {
                    "role": "system",
                    "content": "You are an expert code assistant. Analyze the code snippet and provide a clear, concise explanation."
                },
                {
                    "role": "user",
                    "content": f"Please explain this {language} code snippet:\n\n{code}"
                }
            ]
            
            explanation = await self.generate_completion(messages)
            return explanation
            
        except Exception as e:
            logger.error(f"Error generating explanation: {str(e)}")
            return "Error generating explanation"

    def _delete_collection(self):
        """Delete the existing collection if it exists"""
        try:
            collections = self.qdrant_client.get_collections().collections
            if any(c.name == self.collection_name for c in collections):
                self.qdrant_client.delete_collection(collection_name=self.collection_name)
                logger.info(f"Deleted collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error deleting collection: {str(e)}")
            raise

    async def run(self, query: str) -> Dict[str, Any]:
        """Run the agent with a query and return results"""
        try:
            # Search for relevant code snippets
            code_snippets = await self.search_code(query)
            
            if not code_snippets:
                return {
                    "status": "success",
                    "message": "No relevant code snippets found",
                    "code_snippets": [],
                    "analysis": "No code snippets to analyze",
                    "code_fix": {"explanation": "No code to fix"}
                }
            
            # Analyze the code snippets
            code_text = "\n\n".join([
                f"File: {snippet['file_path']}\n{snippet['code']}"
                for snippet in code_snippets
            ])
            
            # Get analysis as string
            analysis = await self.analyze_code(code_text)
            
            # Generate code fix suggestions
            code_fix = await self.generate_code_fix(query, code_snippets)
            
            return {
                "status": "success",
                "message": "Analysis completed successfully",
                "code_snippets": code_snippets,
                "analysis": analysis,  # Now correctly handling string result
                "code_fix": code_fix
            }
            
        except Exception as e:
            logger.error(f"Error running agent: {str(e)}")
            return {
                "status": "error",
                "message": f"Error running agent: {str(e)}",
                "code_snippets": [],
                "analysis": f"Error occurred: {str(e)}",
                "code_fix": {"status": "error", "message": str(e)}
            } 