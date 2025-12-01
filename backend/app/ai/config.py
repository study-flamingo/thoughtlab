"""AI configuration settings for ThoughtLab.

This module provides configuration for AI/LLM integration including:
- OpenAI API settings
- Model selection
- Confidence thresholds for relationship suggestions
- Processing parameters
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class AIConfig(BaseSettings):
    """Configuration for AI/LLM integration.
    
    All settings can be overridden via environment variables with
    the THOUGHTLAB_ prefix (e.g., THOUGHTLAB_OPENAI_API_KEY).
    """
    
    # OpenAI API
    openai_api_key: str = ""
    
    # Models
    llm_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536
    
    # Confidence thresholds for relationship handling
    # >= auto_create_threshold: Create relationship automatically
    # >= suggest_threshold: Create suggestion for user review
    # < suggest_threshold: Discard silently
    auto_create_threshold: float = 0.8
    suggest_threshold: float = 0.6
    
    # Similarity search settings
    similarity_min_score: float = 0.5
    max_similar_nodes: int = 20
    
    # Processing settings
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    # LLM settings
    llm_temperature: float = 0.1  # Low temperature for consistent classification
    
    model_config = SettingsConfigDict(
        env_prefix="THOUGHTLAB_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # Ignore extra fields from parent config
    )
    
    @property
    def is_configured(self) -> bool:
        """Check if AI is properly configured with an API key."""
        return bool(self.openai_api_key and self.openai_api_key.strip())
    
    def validate_config(self) -> tuple[bool, Optional[str]]:
        """Validate configuration and return status with optional error message."""
        if not self.is_configured:
            return False, "Missing THOUGHTLAB_OPENAI_API_KEY environment variable"
        
        if self.auto_create_threshold <= self.suggest_threshold:
            return False, (
                f"auto_create_threshold ({self.auto_create_threshold}) must be "
                f"greater than suggest_threshold ({self.suggest_threshold})"
            )
        
        if not 0 <= self.similarity_min_score <= 1:
            return False, f"similarity_min_score must be between 0 and 1"
        
        return True, None


@lru_cache()
def get_ai_config() -> AIConfig:
    """Get cached AI configuration.
    
    Uses lru_cache to ensure we only load config once.
    Clear cache with get_ai_config.cache_clear() if needed.
    """
    config = AIConfig()
    
    if config.is_configured:
        logger.info(
            f"AI configured: model={config.llm_model}, "
            f"embeddings={config.embedding_model}"
        )
    else:
        logger.warning(
            "AI not configured: Set THOUGHTLAB_OPENAI_API_KEY to enable "
            "embedding generation and relationship discovery"
        )
    
    return config

