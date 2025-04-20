import logging
from typing import List, Optional, Dict, Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import settings
from app.services.providers.base import BaseProvider

logger = logging.getLogger(__name__)

class VLLMProvider(BaseProvider):
    """vLLM API provider for LLM interactions"""

    def __init__(self):
        """Initialize the vLLM provider with API URL and model settings"""
        super().__init__()
        self.api_url = settings.vllm_url
        self.model = settings.vllm_model
        self.embedding_model = settings.vllm_embedding_model
        
        # Initialize HTTP client for vLLM API calls
        self.client = httpx.Client(
            base_url=self.api_url,
            timeout=60.0
        )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embeddings for the given text using the vLLM embedding model
        
        Args:
            text: The text to generate embeddings for
            
        Returns:
            List[float]: The generated embedding vector (768 dimensions)
            
        Raises:
            Exception: If embedding generation fails
        """
        try:
            if not text:
                raise ValueError("Empty text provided for embedding generation")

            response = self.client.post(
                "/embeddings",
                json={
                    "model": self.embedding_model,
                    "input": text
                }
            )
            response.raise_for_status()
            
            embedding = response.json()["data"][0]["embedding"]
            if len(embedding) != settings.embedding_dimension:
                logger.warning(f"Unexpected embedding dimension from vLLM: got {len(embedding)}, expected {settings.embedding_dimension}")
                
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
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
        """Generate a completion using the vLLM API
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature (default: 0.7)
            max_tokens: Maximum tokens to generate (default: None)
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
                    "max_tokens": max_tokens or settings.vllm_max_tokens,
                    "top_p": settings.vllm_top_p,
                    "stop": stop,
                    **kwargs
                }
            )
            response.raise_for_status()
            
            return response.json()["choices"][0]["message"]["content"]
            
        except Exception as e:
            logger.error(f"Error generating completion: {str(e)}")
            raise 