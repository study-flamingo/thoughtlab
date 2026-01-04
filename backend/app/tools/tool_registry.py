"""Tool registry for programmatic tool access.

Provides a registry pattern for tools that can be used by:
- MCP server to register tools
- LangGraph to create tool bindings
- API to list available tools and capabilities
"""

from typing import Dict, List, Optional, Callable, Any
import logging

from app.tools.tool_definitions import (
    ToolDefinition,
    ToolCategory,
    MCPExecutionMode,
    TOOL_DEFINITIONS,
    get_tool_by_name,
    get_mcp_tools,
    get_langgraph_tools,
)

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Registry for tool definitions and implementations.

    This class provides a centralized way to:
    - Discover available tools
    - Get tool metadata for MCP/LangGraph registration
    - Map tools to their service implementations
    """

    def __init__(self):
        """Initialize the registry with all tool definitions."""
        self._tools: Dict[str, ToolDefinition] = {
            tool.name: tool for tool in TOOL_DEFINITIONS
        }
        self._implementations: Dict[str, Callable] = {}

    def get_tool(self, name: str) -> Optional[ToolDefinition]:
        """Get a tool definition by name."""
        return self._tools.get(name)

    def list_tools(
        self,
        category: Optional[ToolCategory] = None,
        mcp_only: bool = False,
        langgraph_only: bool = False,
        include_dangerous: bool = True,
    ) -> List[ToolDefinition]:
        """List tools with optional filters.

        Args:
            category: Filter by category
            mcp_only: Only return MCP-enabled tools
            langgraph_only: Only return LangGraph-enabled tools
            include_dangerous: Include dangerous tools (default True)

        Returns:
            List of matching tool definitions
        """
        tools = list(self._tools.values())

        if category:
            tools = [t for t in tools if t.category == category]

        if mcp_only:
            tools = [t for t in tools if t.mcp_enabled]

        if langgraph_only:
            tools = [t for t in tools if t.langgraph_enabled]

        if not include_dangerous:
            tools = [t for t in tools if not t.is_dangerous]

        return tools

    def register_implementation(
        self,
        tool_name: str,
        implementation: Callable,
    ) -> None:
        """Register an implementation function for a tool.

        Args:
            tool_name: Name of the tool
            implementation: Async function that implements the tool
        """
        if tool_name not in self._tools:
            logger.warning(f"Registering implementation for unknown tool: {tool_name}")
        self._implementations[tool_name] = implementation

    def get_implementation(self, tool_name: str) -> Optional[Callable]:
        """Get the implementation function for a tool."""
        return self._implementations.get(tool_name)

    def get_capabilities_dict(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get capabilities in the format expected by the /tools/capabilities endpoint.

        Returns:
            Dict organized by category with tool information
        """
        result: Dict[str, List[Dict[str, Any]]] = {}

        for tool in self._tools.values():
            category_name = tool.category.value
            if category_name not in result:
                result[category_name] = []

            result[category_name].append({
                "operation": tool.name,
                "endpoint": self._get_endpoint_for_tool(tool),
                "description": tool.description,
                "is_dangerous": tool.is_dangerous,
                "mcp_mode": tool.mcp_mode.value,
            })

        return result

    def _get_endpoint_for_tool(self, tool: ToolDefinition) -> str:
        """Generate the API endpoint path for a tool."""
        if tool.requires_node_id:
            return f"POST /tools/nodes/{{node_id}}/{tool.name.replace('_', '-')}"
        elif tool.requires_edge_id:
            return f"POST /tools/relationships/{{edge_id}}/{tool.name.replace('_', '-')}"
        else:
            return f"POST /tools/{tool.name.replace('_', '-')}"

    def get_mcp_tool_schemas(self, include_dangerous: bool = False) -> List[Dict[str, Any]]:
        """Generate tool schemas for MCP server registration.

        Args:
            include_dangerous: Include dangerous tools (requires admin mode)

        Returns:
            List of tool schemas in MCP format
        """
        tools = get_mcp_tools(include_dangerous=include_dangerous)
        schemas = []

        for tool in tools:
            schema = {
                "name": tool.name,
                "description": tool.description,
                "inputSchema": self._generate_input_schema(tool),
            }
            schemas.append(schema)

        return schemas

    def get_langgraph_tool_schemas(self) -> List[Dict[str, Any]]:
        """Generate tool schemas for LangGraph agent binding.

        Returns:
            List of tool schemas suitable for LangChain tools
        """
        tools = get_langgraph_tools()
        schemas = []

        for tool in tools:
            schema = {
                "name": tool.name,
                "description": tool.description,
                "parameters": self._generate_parameters_schema(tool),
                "is_dangerous": tool.is_dangerous,
                "service_method": tool.service_method,
            }
            schemas.append(schema)

        return schemas

    def _generate_input_schema(self, tool: ToolDefinition) -> Dict[str, Any]:
        """Generate JSON Schema for tool inputs (MCP format)."""
        properties = {}
        required = []

        for param in tool.parameters:
            prop: Dict[str, Any] = {
                "type": self._json_type(param.type),
                "description": param.description,
            }

            if param.default is not None:
                prop["default"] = self._parse_default(param.default, param.type)

            properties[param.name] = prop

            if param.required:
                required.append(param.name)

        return {
            "type": "object",
            "properties": properties,
            "required": required,
        }

    def _generate_parameters_schema(self, tool: ToolDefinition) -> Dict[str, Any]:
        """Generate parameters schema for LangChain tools."""
        return self._generate_input_schema(tool)

    def _json_type(self, param_type: str) -> str:
        """Convert parameter type to JSON Schema type."""
        type_map = {
            "string": "string",
            "integer": "integer",
            "float": "number",
            "boolean": "boolean",
            "array": "array",
        }
        return type_map.get(param_type, "string")

    def _parse_default(self, default: str, param_type: str) -> Any:
        """Parse default value string to appropriate type."""
        if param_type == "boolean":
            return default.lower() == "true"
        elif param_type == "integer":
            return int(default)
        elif param_type == "float":
            return float(default)
        return default


# Global registry instance
_registry: Optional[ToolRegistry] = None


def get_tool_registry() -> ToolRegistry:
    """Get the global tool registry instance."""
    global _registry
    if _registry is None:
        _registry = ToolRegistry()
    return _registry
