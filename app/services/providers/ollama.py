import logging
from typing import List, Optional, Dict, Any
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import settings
from app.services.providers.base import BaseProvider

logger = logging.getLogger(__name__)

class OllamaProvider(BaseProvider):
    """Ollama API provider for LLM interactions"""

    def __init__(self):
        """Initialize the Ollama provider with base URL and model"""
        super().__init__()
        self.base_url = settings.ollama_base_url
        self.model = settings.ollama_model
        self.embedding_model = settings.ollama_embedding_model or self.model
        
        # Initialize HTTP client for Ollama API calls
        self.client = httpx.Client(
            base_url=self.base_url,
            timeout=60.0
        )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embeddings for the given text
        
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

            response = self.client.post(
                "/api/embed",
                json={
                    "model": self.embedding_model,
                    "prompt": text
                }
            )
            response.raise_for_status()
            
            result = response.json()
            if "embedding" not in result:
                raise ValueError(f"No embedding found in Ollama response: {result}")
                
            embedding = result["embedding"]
            
            # Verify and adjust embedding dimension
            if len(embedding) != settings.embedding_dimension:
                logger.warning(f"Embedding dimension mismatch from Ollama: got {len(embedding)}, expected {settings.embedding_dimension}")
                # Always truncate to match expected dimension
                # This preserves the most significant dimensions while ensuring consistent size
                embedding = embedding[:settings.embedding_dimension]
                logger.info(f"Truncated embedding to {len(embedding)} dimensions")
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding with Ollama: {str(e)}")
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
        """Generate a completion using the Ollama API
        
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
            data = {
                "model": self.model,
                "prompt": prompt,
                "temperature": temperature,
                **kwargs
            }
            
            if system_prompt:
                data["system"] = system_prompt
            if max_tokens:
                data["max_tokens"] = max_tokens
            if stop:
                data["stop"] = stop
            
            response = self.client.post("/api/generate", json=data)
            response.raise_for_status()
            
            return response.json()["response"]
            
        except Exception as e:
            logger.error(f"Error generating completion: {str(e)}")
            raise 