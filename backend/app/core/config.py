from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List
from functools import lru_cache


class Settings(BaseSettings):
    # Database
    neo4j_uri: str
    neo4j_user: str
    neo4j_password: str
    database_url: str
    redis_url: str
    
    # Security
    secret_key: str
    
    # Application
    debug: bool = False
    environment: str = "development"
    cors_allow_origins: List[str] = [
        "http://localhost:5173",
        "http://localhost:3000",
    ]
    
    # LLM
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
