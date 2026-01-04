"""Tool service package.

Provides LLM-powered graph operations organized by domain:
- Node analysis (find related, summarize, confidence)
- Node modification (reclassify, merge, web evidence)
- Relationship analysis (summarize, reclassify, confidence)

Usage:
    from app.services.tools import get_tool_service, ToolService

    tool_service = get_tool_service()
    result = await tool_service.summarize_node("node-123")
"""

from app.services.tools.service import ToolService, get_tool_service

__all__ = [
    "ToolService",
    "get_tool_service",
]
