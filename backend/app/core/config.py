from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List, Optional, TYPE_CHECKING
from functools import lru_cache

if TYPE_CHECKING:
    from app.ai.config import AIConfig


class Settings(BaseSettings):
    # Database
    neo4j_uri: str
    neo4j_user: str
    neo4j_password: str
    database_url: str
    redis_url: str = "redis://localhost:6379"  # Optional - for caching
    
    # Security
    secret_key: str
    
    # Application
    debug: bool = False
    environment: str = "development"
    cors_allow_origins: List[str] = [
        "http://localhost:5173",
        "http://localhost:5174",
        "http://localhost:5175",
        "http://localhost:5176",
        "http://localhost:5177",
        "http://localhost:5178",
        "http://localhost:5179",
        "http://localhost:3000",
    ]
    
    # LLM (legacy - use AIConfig for new AI features)
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
    )
    
    @property
    def ai_config(self) -> "AIConfig":
        """Get AI configuration.
        
        Provides access to AI-specific settings like embedding models,
        LLM configuration, and confidence thresholds.
        """
        from app.ai.config import get_ai_config
        return get_ai_config()
    
    @property
    def is_ai_enabled(self) -> bool:
        """Check if AI features are enabled (API key configured)."""
        return self.ai_config.is_configured


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
