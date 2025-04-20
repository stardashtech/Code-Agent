import logging
from typing import List, Optional, Dict, Any

import httpx
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import settings
from app.services.providers.base import BaseProvider
from app.services.providers.ollama import OllamaProvider
from app.services.providers.vllm import VLLMProvider

logger = logging.getLogger(__name__)

class OpenRouterProvider(BaseProvider):
    """OpenRouter API provider for LLM interactions"""

    def __init__(self):
        """Initialize the OpenRouter provider with API key and base URL"""
        super().__init__()
        self.api_key = settings.openrouter_api_key
        self.base_url = settings.openrouter_base_url
        self.model = settings.openrouter_model
        self.embedding_provider = self._initialize_embedding_provider()
        
        # Initialize HTTP client for OpenRouter API calls
        self.client = httpx.Client(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "HTTP-Referer": "http://localhost:3000",
            },
            timeout=60.0
        )

    def _initialize_embedding_provider(self) -> BaseProvider:
        """Initialize the embedding provider based on configuration"""
        provider = settings.openrouter_embedding_provider.lower()
        
        if provider == "openai":
            return OpenAI(api_key=settings.openai_api_key)
        elif provider == "ollama":
            return OllamaProvider()
        elif provider == "vllm":
            return VLLMProvider()
        else:
            raise ValueError(f"Unsupported embedding provider: {provider}")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embeddings for the given text using the configured provider
        
        Args:
            text: The text to generate embeddings for
            
        Returns:
            List[float]: The generated embedding vector
            
        Raises:
            Exception: If embedding generation fails
        """
        try:
            if not text:
                raise ValueError("Empty text provided for embedding generation")

            provider = settings.openrouter_embedding_provider.lower()
            
            if provider == "openai":
                response = self.embedding_provider.embeddings.create(
                    model=settings.openai_embedding_model,
                    input=text
                )
                embedding = response.data[0].embedding
                
            elif provider == "ollama":
                embedding = self.embedding_provider.generate_embedding(text)
                
            elif provider == "vllm":
                embedding = self.embedding_provider.generate_embedding(text)
                
            else:
                raise ValueError(f"Unsupported embedding provider: {provider}")
            
            # Verify embedding dimension
            if len(embedding) != settings.embedding_dimension:
                logger.warning(f"Embedding dimension mismatch from {provider}: got {len(embedding)}, expected {settings.embedding_dimension}")
                # Always truncate to match expected dimension
                # This preserves the most significant dimensions while ensuring consistent size
                embedding = embedding[:settings.embedding_dimension]
                logger.info(f"Truncated embedding to {len(embedding)} dimensions")
            
            return embedding
                
        except Exception as e:
            logger.error(f"Error generating embedding with {provider}: {str(e)}")
            return [0.0] * settings.embedding_dimension

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def generate_completion(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        **kwargs: Dict[str, Any]
    ) -> str:
        """Generate a completion using the OpenRouter API
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature (default: 0.7)
            max_tokens: Maximum tokens to generate
            stop: Optional list of stop sequences
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            str: The generated completion text
            
        Raises:
            Exception: If the API call fails
        """
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = self.client.post(
                "/chat/completions",
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "stop": stop,
                    **kwargs
                }
            )
            response.raise_for_status()
            
            return response.json()["choices"][0]["message"]["content"]
            
        except Exception as e:
            logger.error(f"Error generating completion: {str(e)}")
            raise 