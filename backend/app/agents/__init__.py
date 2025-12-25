"""LangGraph agent layer for ThoughtLab.

This module provides LangGraph agents that interact with the backend API
via HTTP calls, maintaining complete separation between the AI layer and
backend logic.
"""

from app.agents.tools import get_thoughtlab_tools
from app.agents.agent import create_thoughtlab_agent
from app.agents.config import AgentConfig

__all__ = [
    "get_thoughtlab_tools",
    "create_thoughtlab_agent",
    "AgentConfig",
]
