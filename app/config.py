import os
from typing import Optional
from dotenv import load_dotenv
from dataclasses import dataclass
from pydantic_settings import BaseSettings
from urllib.parse import urlparse

# Load environment variables from .env file
load_dotenv()

class Settings(BaseSettings):
    """Application settings"""
    
    # LLM Provider Configuration
    llm_provider: str = "openrouter"  # openai, openrouter, ollama, or vllm
    
    # Embedding Configuration
    embedding_dimension: int = 1536  # Default dimension for text-embedding-3-small
    
    # OpenAI Configuration
    openai_api_key: str = ""
    openai_model: str = "gpt-4-1106-preview"
    openai_embedding_model: str = "text-embedding-3-large"
    
    # OpenRouter Configuration
    openrouter_api_key: str = "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"
    openrouter_model: str = "deepseek/deepseek-r1:free"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_embedding_provider: str = "ollama"  # openai, ollama, or vllm
    
    # Ollama Configuration
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "gemma3:4b"
    ollama_embedding_model: str = "nomic-embed-text"
    
    # vLLM Configuration
    vllm_url: str = "http://localhost:8000"
    vllm_model: str = "codellama/CodeLlama-7b-Instruct-hf"
    vllm_embedding_model: str = "BAAI/bge-large-en-v1.5"
    vllm_max_tokens: int = 2048
    vllm_top_p: float = 0.95
    
    # GitHub Configuration
    github_token: Optional[str] = None
    
    # Stack Overflow Configuration
    stackoverflow_key: Optional[str] = None
    
    # Tavily Search Configuration
    tavily_api_key: Optional[str] = "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"
    
    # Qdrant Configuration
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: Optional[str] = None
    qdrant_collection: str = "code_snippets"  # Base name without dimensions
    
    # Redis Configuration
    redis_url: str = "redis://localhost:6379"
    redis_password: Optional[str] = None
    redis_db: int = 0
    redis_cache_ttl: int = 86400  # 24 hours
    
    # Neo4j Configuration
    neo4j_url: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"
    
    # MeiliSearch Configuration
    meilisearch_url: str = "http://localhost:7700"
    meilisearch_api_key: str = ""
    meilisearch_index: str = "code_search"
    
    # Google Search Configuration
    google_api_key: Optional[str] = None
    google_cse_id: Optional[str] = None
    
    # Application Settings
    debug: bool = False
    worker_count: int = 4
    
    # Model Settings
    max_context_length: int = 4096
    temperature: float = 0.2
    
    # Search Settings
    min_search_score: float = 0.3
    max_search_results: int = 10
    search_timeout: int = 30
    
    # Cache Settings
    enable_response_cache: bool = True
    enable_embedding_cache: bool = True
    enable_search_cache: bool = True
    
    @property
    def redis_host(self) -> str:
        """Extract host from redis_url"""
        parsed = urlparse(self.redis_url)
        return parsed.hostname or "localhost"
    
    @property
    def redis_port(self) -> int:
        """Extract port from redis_url"""
        parsed = urlparse(self.redis_url)
        return parsed.port or 6379
    
    @property
    def cache_ttl(self) -> int:
        """Alias for redis_cache_ttl"""
        return self.redis_cache_ttl
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "allow"  # Allow extra fields from environment variables

# Create a global settings instance
settings = Settings() 