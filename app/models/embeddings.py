import openai
import ollama
import numpy as np
import logging
from config import settings # Import the settings object

logger = logging.getLogger(__name__)

# Initialize clients based on provider
openai_client = None
ollama_client = None

if settings.EMBEDDING_PROVIDER == "openai":
    # Use specific embedding key if provided, else fallback to general OPENAI_API_KEY
    api_key = settings.OPENAI_API_KEY # Pydantic handles the None case
    if not api_key:
        logger.warning("OpenAI embedding provider selected, but OPENAI_API_KEY is not set in config/env.")
    else:
        openai_client = openai.OpenAI(api_key=api_key)
elif settings.EMBEDDING_PROVIDER == "ollama":
    try:
        # Use the effective_embedding_base_url which falls back to LLM_BASE_URL
        ollama_host = settings.effective_embedding_base_url
        ollama_client = ollama.Client(host=ollama_host)
        ollama_client.list() 
        logger.info(f"Connected to Ollama embedding provider at {ollama_host}")
    except Exception as e:
        logger.error(f"Failed to connect to Ollama at {settings.effective_embedding_base_url}: {e}")
        ollama_client = None
else:
    logger.warning(f"Unsupported EMBEDDING_PROVIDER: {settings.EMBEDDING_PROVIDER}. Only 'openai' and 'ollama' are currently supported.")

def get_embedding(text: str) -> np.ndarray:
    """
    Converts text to a vector using the configured embedding provider and model.
    Returns a NumPy array of float32.
    Raises RuntimeError if embedding fails or provider is not configured correctly.
    """
    embedding = None
    try:
        if settings.EMBEDDING_PROVIDER == "openai" and openai_client:
            resp = openai_client.embeddings.create(model=settings.EMBEDDING_MODEL, input=text)
            embedding = resp.data[0].embedding
        elif settings.EMBEDDING_PROVIDER == "ollama" and ollama_client:
            resp = ollama_client.embeddings(model=settings.EMBEDDING_MODEL, prompt=text)
            embedding = resp["embedding"]
        else:
            raise RuntimeError(f"Embedding provider '{settings.EMBEDDING_PROVIDER}' is not supported or client failed to initialize.")
        
        if embedding:
            embedding_array = np.array(embedding, dtype=np.float32)
            # Compare against configured dimension
            if embedding_array.shape[0] != settings.EMBEDDING_DIMENSION:
                 logger.warning(f"Warning: Embedding dimension mismatch for model {settings.EMBEDDING_MODEL}. Expected {settings.EMBEDDING_DIMENSION}, got {embedding_array.shape[0]}. Ensure config matches model output.")
            return embedding_array
        else:
            raise RuntimeError("Embedding generation resulted in an empty response.")

    except Exception as e:
        logger.error(f"Embedding call failed for provider {settings.EMBEDDING_PROVIDER} with model {settings.EMBEDDING_MODEL}: {e}")
        raise RuntimeError(f"Embedding API call failed: {e}") from e 