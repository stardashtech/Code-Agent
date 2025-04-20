import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings
from typing import Optional

# Load .env file explicitly
load_dotenv()

class Settings(BaseSettings):
    """Application settings"""
    app_name: str = "Code Agent API"
    enable_mcp: bool = True
    debug: bool = False
    workers: int = 4
    
    # API keys and endpoints
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o")
    
    # Vector database settings
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key: Optional[str] = os.getenv("QDRANT_API_KEY")
    qdrant_collection: str = os.getenv("QDRANT_COLLECTION", "code_snippets")
    
    # MeiliSearch settings
    meili_url: str = os.getenv("MEILI_URL", "http://localhost:7700")
    meili_api_key: Optional[str] = os.getenv("MEILI_API_KEY")
    
    # Neo4j settings
    neo4j_url: str = os.getenv("NEO4J_URL", "bolt://localhost:7687")
    neo4j_user: str = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password: Optional[str] = os.getenv("NEO4J_PASSWORD")
    
    # Google Custom Search API settings
    google_api_key: Optional[str] = os.getenv("GOOGLE_API_KEY")
    google_cse_id: Optional[str] = os.getenv("GOOGLE_CSE_ID")
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# Create settings instance
settings = Settings() 