import openai
import ollama
import logging
import time
from abc import ABC, abstractmethod
from typing import Optional
from config import settings # Import the settings object

logger = logging.getLogger(__name__)

class LlmInterface(ABC):
    """Abstract interface for LLM providers."""
    
    @abstractmethod
    def generate(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        """
        Generate a response from the LLM.
        
        Args:
            prompt: The input prompt to send to the LLM
            max_tokens: Optional maximum number of tokens to generate
            
        Returns:
            The generated text response
        """
        pass

class OpenAIClient(LlmInterface):
    """OpenAI implementation of the LLM interface."""
    
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        
    def generate(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a production-ready advanced code assistant. Respond with safe, clear and thorough answers."},
                {"role": "user", "content": prompt}
            ],
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=max_tokens or settings.LLM_MAX_TOKENS
        )
        return response.choices[0].message.content

class OllamaClient(LlmInterface):
    """Ollama implementation of the LLM interface."""
    
    def __init__(self, host: str, model: str = "llama2"):
        self.client = ollama.Client(host=host)
        self.model = model
        
    def generate(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        response = self.client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a production-ready advanced code assistant. Respond with safe, clear and thorough answers."},
                {"role": "user", "content": prompt}
            ],
            options={
                'temperature': settings.LLM_TEMPERATURE,
                # Ollama doesn't have a direct max_tokens equivalent in the same way
            }
        )
        return response['message']['content']

# Initialize LLM client based on provider
llm_client = None

def get_llm_client() -> Optional[LlmInterface]:
    """
    Returns an initialized LLM client based on configuration settings.
    
    Returns:
        An implementation of LlmInterface or None if initialization fails
    """
    global llm_client
    
    if llm_client:
        return llm_client
        
    if settings.LLM_PROVIDER == "openai":
        if not settings.OPENAI_API_KEY:
            logger.error("OpenAI LLM provider selected, but OPENAI_API_KEY is not set.")
            return None
        try:
            client = OpenAIClient(api_key=settings.OPENAI_API_KEY, model=settings.LLM_MODEL)
            logger.info("Successfully created OpenAI LLM client.")
            llm_client = client
            return client
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI LLM client: {e}")
            return None
            
    elif settings.LLM_PROVIDER == "ollama":
        try:
            client = OllamaClient(host=settings.LLM_BASE_URL, model=settings.LLM_MODEL)
            logger.info(f"Successfully created Ollama LLM client at {settings.LLM_BASE_URL}.")
            llm_client = client
            return client
        except Exception as e:
            logger.error(f"Failed to initialize Ollama LLM client: {e}")
            return None
            
    else:
        logger.error(f"Unsupported LLM_PROVIDER: {settings.LLM_PROVIDER}. Only 'openai' and 'ollama' are supported.")
        return None

def call_llm(prompt: str, max_retries=2, retry_delay=3) -> str:
    """
    Calls the configured LLM provider (OpenAI or Ollama) with the given prompt.
    Includes safety directives and basic retry logic for transient errors.
    Raises RuntimeError if the call fails after retries or provider is unavailable.
    """
    client = get_llm_client()
    if not client:
        raise RuntimeError(f"LLM provider '{settings.LLM_PROVIDER}' is not available or failed to initialize.")

    safe_prompt = f"{settings.PROMPT_SAFETY_DIRECTIVE}\n{prompt}"

    for attempt in range(max_retries + 1):
        try:
            return client.generate(safe_prompt)
        except openai.AuthenticationError as e:
            logger.error(f"OpenAI Authentication Error: {e}. Check API key.")
            raise RuntimeError(f"OpenAI Authentication Failed: {e}") from e
        except openai.RateLimitError as e:
            logger.warning(f"OpenAI Rate Limit Error: {e}. Retrying in {retry_delay}s... (Attempt {attempt + 1}/{max_retries + 1})")
            if attempt == max_retries:
                raise RuntimeError(f"OpenAI Rate Limit Exceeded after retries: {e}") from e
            time.sleep(retry_delay)
        except (openai.APIConnectionError, openai.APITimeoutError, TimeoutError) as e:
            logger.warning(f"LLM Connection/Timeout Error: {e}. Retrying in {retry_delay}s... (Attempt {attempt + 1}/{max_retries + 1})")
            if attempt == max_retries:
                raise RuntimeError(f"LLM Connection Failed after retries: {e}") from e
            time.sleep(retry_delay)
        except ollama.ResponseError as e:
             # Ollama errors might include model not found, connection issues, etc.
            logger.error(f"Ollama Error: {e.error} (Status code: {e.status_code})")
            # Decide if retry is appropriate based on error type, e.g., not for model not found
            if "connection refused" in e.error.lower() and attempt < max_retries:
                 logger.warning(f"Ollama connection refused. Retrying in {retry_delay}s... (Attempt {attempt + 1}/{max_retries + 1})")
                 time.sleep(retry_delay)
            else:
                raise RuntimeError(f"Ollama API call failed: {e.error}") from e
        except Exception as e:
            logger.error(f"Unexpected LLM call error ({type(e).__name__}): {e}")
            # Generic retry for unexpected errors
            if attempt < max_retries:
                logger.warning(f"Retrying unexpected error in {retry_delay}s... (Attempt {attempt + 1}/{max_retries + 1})")
                time.sleep(retry_delay)
            else:
                raise RuntimeError(f"LLM call failed due to unexpected error after retries: {e}") from e
                
    # Should not be reachable if logic is correct, but acts as a safeguard
    raise RuntimeError("LLM call failed after all retries.") 