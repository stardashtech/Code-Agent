import openai
import numpy as np
from config import OPENAI_API_KEY, EMBEDDING_MODEL, EMBEDDING_DIMENSION

openai.api_key = OPENAI_API_KEY

def get_embedding(text: str) -> np.ndarray:
    """
    Converts text to a vector of EMBEDDING_DIMENSION size using OpenAI Embedding API.
    """
    try:
        resp = openai.Embedding.create(model=EMBEDDING_MODEL, input=text)
        embedding = resp["data"][0]["embedding"]
        return np.array(embedding, dtype=np.float32)
    except Exception as e:
        raise RuntimeError(f"Embedding API call failed: {e}") 