import os

# OpenAI API information
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Please define the OPENAI_API_KEY environment variable.")

# LLM ChatCompletion Model
LLM_MODEL = "gpt-4"
LLM_TEMPERATURE = 0.2
LLM_MAX_TOKENS = 1024
PROMPT_SAFETY_DIRECTIVE = (
    "WARNING: Please do not generate any sensitive or dangerous content. "
    "Complete the given prompt instructions in accordance with safety rules."
)

# Embedding Model
EMBEDDING_MODEL = "text-embedding-ada-002"
EMBEDDING_DIMENSION = 1536  # for "ada-002"

# Code execution settings
CODE_EXECUTION_TIMEOUT = 5  # seconds
DOCKER_IMAGE = "python:3.9-slim"  # Docker image to use

# Qdrant Settings
QDRANT_HOST = os.getenv("QDRANT_HOST", "http://localhost:6333")
QDRANT_COLLECTION = "agent_docs"

# MeiliSearch Settings (for logging)
MEILISEARCH_HOST = os.getenv("MEILISEARCH_HOST", "http://localhost:7700")
MEILISEARCH_API_KEY = os.getenv("MEILISEARCH_API_KEY", "masterKey")
MEILISEARCH_INDEX = "agent_logs"

# Chunking settings
CHUNK_SIZE = 500  # Character limit per chunk

# Chain-of-thought logging (hidden reflection)
ENABLE_CHAIN_OF_THOUGHT_LOGGING = True 