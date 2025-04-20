from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any

class BaseProvider(ABC):
    """Base class for LLM providers defining the common interface"""

    @abstractmethod
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embeddings for the given text
        
        Args:
            text: The text to generate embeddings for
            
        Returns:
            List[float]: The generated embedding vector
            
        Raises:
            NotImplementedError: If not implemented by child class
        """
        raise NotImplementedError()

    @abstractmethod
    def generate_completion(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        **kwargs: Dict[str, Any]
    ) -> str:
        """Generate a completion for the given prompt
        
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
            NotImplementedError: If not implemented by child class
        """
        raise NotImplementedError() 