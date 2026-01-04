"""LangGraph agent layer for ThoughtLab.

This module provides LangGraph agents that:
- Call ToolService directly (in-process, no HTTP)
- Save results to ReportService for later viewing
- Handle dangerous tool confirmation via Activity Feed
"""

from app.agents.agent_tools import get_agent_tools, get_thoughtlab_tools
from app.agents.agent import create_thoughtlab_agent
from app.agents.config import AgentConfig

__all__ = [
    "get_agent_tools",
    "get_thoughtlab_tools",  # Legacy alias
    "create_thoughtlab_agent",
    "AgentConfig",
]
