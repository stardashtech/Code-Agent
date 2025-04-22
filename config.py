import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    """Manages application configuration using Pydantic BaseSettings."""
    
    # Load .env file if it exists
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')

    # --- API Keys --- 
    # Specific keys are optional; logic in modules might fallback or require them conditionally.
    OPENAI_API_KEY: Optional[str] = None
    GOOGLE_API_KEY: Optional[str] = None
    GOOGLE_CSE_ID: Optional[str] = None 
    MEILISEARCH_API_KEY: str = "masterKey" # Default key for local Meili
    # EMBEDDING_API_KEY is intentionally omitted here, let modules use OPENAI_API_KEY if needed
    # as embedding provider logic already handles this fallback.

    # --- LLM --- 
    LLM_PROVIDER: str = "ollama" # Supported: "ollama", "openai"
    LLM_MODEL: str = "gemma3:4b" # Using codellama model which is available in the system
    LLM_BASE_URL: str = "http://localhost:11434"
    LLM_TEMPERATURE: float = 0.2
    LLM_MAX_TOKENS: int = 1024
    PROMPT_SAFETY_DIRECTIVE: str = (
        "WARNING: Please do not generate any sensitive or dangerous content. "
        "Complete the given prompt instructions in accordance with safety rules."
    )

    # --- Embeddings --- 
    EMBEDDING_PROVIDER: str = "ollama" # Supported: "ollama", "openai"
    EMBEDDING_MODEL: str = "nomic-embed-text"
    # EMBEDDING_BASE_URL defaults to LLM_BASE_URL if not explicitly set
    EMBEDDING_BASE_URL: Optional[str] = None 
    EMBEDDING_DIMENSION: int = 768 # Default, adjust if using different models
    
    # --- Vector Store (Qdrant) --- 
    QDRANT_HOST: str = "http://localhost:6333"
    QDRANT_COLLECTION: str = "agent_docs"

    # --- Knowledge Graph (Neo4j) ---
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "password"

    # --- Logging Store (MeiliSearch) --- 
    MEILISEARCH_HOST: str = "http://localhost:7700"
    MEILISEARCH_INDEX: str = "agent_logs"

    # --- Redis Cache --- 
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_CACHE_TTL: int = 86400 # 24 hours

    # --- Code Execution --- 
    CODE_EXECUTION_TIMEOUT: int = 5 # seconds
    DOCKER_IMAGE: str = "python:3.9-slim"
    # DOCKER_HOST is read directly via os.getenv in CodeExecutor as needed

    # --- Processing --- 
    CHUNK_SIZE: int = 500

    # --- Agent Behavior --- 
    ENABLE_CHAIN_OF_THOUGHT_LOGGING: bool = True
    RISK_CONFIRMATION_THRESHOLD: str = 'MEDIUM' # Risk level (e.g., LOW, MEDIUM, HIGH) at which user confirmation is required before applying changes.

    # --- Helper method to get embedding base url --- 
    @property
    def effective_embedding_base_url(self) -> str:
        return self.EMBEDDING_BASE_URL or self.LLM_BASE_URL

# Create a single instance of the settings to be imported by other modules
try:
    settings = Settings()
except Exception as e:
    # Handle potential validation errors during settings loading
    print(f"FATAL ERROR: Failed to load configuration settings: {e}")
    # Depending on the application context, you might exit or raise further
    raise SystemExit(f"Configuration error: {e}") from e 