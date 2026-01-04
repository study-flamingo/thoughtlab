"""Tools package - Shared tool definitions for MCP and LangGraph.

This package provides:
- ToolDefinition: Pydantic model for tool metadata
- TOOL_DEFINITIONS: List of all available tools
- ToolRegistry: Centralized registry for tool discovery

Usage:
    from app.tools import get_tool_registry, ToolDefinition, ToolCategory

    registry = get_tool_registry()
    tools = registry.list_tools(category=ToolCategory.NODE_ANALYSIS)

    # For MCP registration
    schemas = registry.get_mcp_tool_schemas(include_dangerous=False)

    # For LangGraph binding
    schemas = registry.get_langgraph_tool_schemas()
"""

from app.tools.tool_definitions import (
    ToolDefinition,
    ToolCategory,
    MCPExecutionMode,
    ToolParameter,
    TOOL_DEFINITIONS,
    get_tool_by_name,
    get_tools_by_category,
    get_mcp_tools,
    get_langgraph_tools,
    get_sync_tools,
    get_async_tools,
    get_dangerous_tools,
)
from app.tools.tool_registry import (
    ToolRegistry,
    get_tool_registry,
)

__all__ = [
    # Tool definitions
    "ToolDefinition",
    "ToolCategory",
    "MCPExecutionMode",
    "ToolParameter",
    "TOOL_DEFINITIONS",
    # Lookup functions
    "get_tool_by_name",
    "get_tools_by_category",
    "get_mcp_tools",
    "get_langgraph_tools",
    "get_sync_tools",
    "get_async_tools",
    "get_dangerous_tools",
    # Registry
    "ToolRegistry",
    "get_tool_registry",
]
