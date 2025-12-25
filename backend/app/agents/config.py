"""Agent configuration for ThoughtLab LangGraph agents."""

from pydantic import BaseModel, Field
from typing import Optional
import os


class AgentConfig(BaseModel):
    """Configuration for ThoughtLab agent.

    This configuration is separate from the backend AI config to maintain
    modularity between the agent layer and backend services.
    """

    # LLM Configuration
    model_name: str = Field(
        default="gpt-4o-mini",
        description="OpenAI model to use for agent reasoning"
    )
    temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="Temperature for LLM responses (lower = more deterministic)"
    )
    max_tokens: Optional[int] = Field(
        default=None,
        description="Maximum tokens in LLM response"
    )

    # API Configuration
    api_base_url: str = Field(
        default="http://localhost:8000/api/v1",
        description="Base URL for ThoughtLab backend API"
    )
    api_timeout: float = Field(
        default=30.0,
        description="Timeout for API requests in seconds"
    )

    # Agent Behavior
    max_iterations: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum iterations for agent reasoning loop"
    )
    verbose: bool = Field(
        default=False,
        description="Enable verbose logging"
    )

    # OpenAI API Key
    openai_api_key: Optional[str] = Field(
        default=None,
        description="OpenAI API key (defaults to THOUGHTLAB_OPENAI_API_KEY env var)"
    )

    def __init__(self, **kwargs):
        """Initialize config with environment variable defaults."""
        # Load OpenAI API key from environment if not provided
        if "openai_api_key" not in kwargs or kwargs["openai_api_key"] is None:
            kwargs["openai_api_key"] = os.getenv("THOUGHTLAB_OPENAI_API_KEY", "")

        # Load API base URL from environment if provided
        if "api_base_url" not in kwargs:
            env_url = os.getenv("THOUGHTLAB_API_BASE_URL")
            if env_url:
                kwargs["api_base_url"] = env_url

        super().__init__(**kwargs)

    @property
    def is_configured(self) -> bool:
        """Check if agent is properly configured."""
        return bool(self.openai_api_key)

    def model_dump_safe(self) -> dict:
        """Dump config without sensitive information."""
        data = self.model_dump()
        if data.get("openai_api_key"):
            data["openai_api_key"] = "***"
        return data
