"""Shared tool definitions for MCP and LangGraph.

This module defines all available tools with their metadata, execution modes,
and flags. Both MCP server and LangGraph agents consume these definitions
to ensure consistency.

Key Concepts:
- MCPExecutionMode: How MCP server handles the tool (SYNC returns immediately,
  ASYNC queues job and client polls with check_job_status)
- is_dangerous: Requires user confirmation (LangGraph) or admin mode (MCP)
- mcp_enabled/langgraph_enabled: Control availability per context
"""

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class ToolCategory(str, Enum):
    """Categories for organizing tools."""
    NODE_ANALYSIS = "node_analysis"
    NODE_MODIFICATION = "node_modification"
    RELATIONSHIP_ANALYSIS = "relationship_analysis"
    JOB_MANAGEMENT = "job_management"


class MCPExecutionMode(str, Enum):
    """Execution mode for MCP server.

    Note: LangGraph ALWAYS queues - this only affects MCP behavior.

    SYNC: MCP executes immediately and returns result to client
    ASYNC: MCP queues job, client polls with check_job_status
    """
    SYNC = "sync"
    ASYNC = "async"


class ToolParameter(BaseModel):
    """Definition of a tool parameter."""
    name: str
    type: str  # "string", "integer", "float", "boolean", "array"
    description: str
    required: bool = False
    default: Optional[str] = None


class ToolDefinition(BaseModel):
    """Definition of a single tool.

    This definition is shared between MCP and LangGraph to ensure
    consistent tool behavior across all consumers.
    """
    name: str = Field(..., description="Unique tool identifier")
    description: str = Field(..., description="Human-readable description")
    category: ToolCategory = Field(..., description="Tool category")
    mcp_mode: MCPExecutionMode = Field(
        default=MCPExecutionMode.ASYNC,
        description="How MCP handles this tool (only affects MCP)"
    )
    is_dangerous: bool = Field(
        default=False,
        description="Requires confirmation (LangGraph) or admin mode (MCP)"
    )
    mcp_enabled: bool = Field(
        default=True,
        description="Available in MCP server"
    )
    langgraph_enabled: bool = Field(
        default=True,
        description="Available in LangGraph"
    )
    service_method: str = Field(
        ...,
        description="Method name on ToolService"
    )
    parameters: List[ToolParameter] = Field(
        default_factory=list,
        description="Tool parameters"
    )
    requires_node_id: bool = Field(
        default=False,
        description="Requires a node ID parameter"
    )
    requires_edge_id: bool = Field(
        default=False,
        description="Requires an edge ID parameter"
    )


# =============================================================================
# Tool Definitions
# =============================================================================

TOOL_DEFINITIONS: List[ToolDefinition] = [
    # -------------------------------------------------------------------------
    # Node Analysis Tools
    # -------------------------------------------------------------------------
    ToolDefinition(
        name="find_related_nodes",
        description="Find semantically similar nodes using vector embeddings. "
                    "Discovers connections between research and identifies patterns.",
        category=ToolCategory.NODE_ANALYSIS,
        mcp_mode=MCPExecutionMode.ASYNC,  # May take time for similarity search
        service_method="find_related_nodes",
        requires_node_id=True,
        parameters=[
            ToolParameter(
                name="node_id",
                type="string",
                description="The node to find similar nodes for",
                required=True
            ),
            ToolParameter(
                name="limit",
                type="integer",
                description="Maximum number of results (1-50)",
                default="10"
            ),
            ToolParameter(
                name="min_similarity",
                type="float",
                description="Minimum similarity score 0.0-1.0",
                default="0.5"
            ),
            ToolParameter(
                name="node_types",
                type="array",
                description="Filter by node types (e.g., ['Observation', 'Hypothesis'])"
            ),
            ToolParameter(
                name="auto_link",
                type="boolean",
                description="Automatically create relationships for high-confidence matches",
                default="false"
            ),
        ],
    ),
    ToolDefinition(
        name="summarize_node",
        description="Generate an AI summary of a node's content. "
                    "Provides quick understanding and key points.",
        category=ToolCategory.NODE_ANALYSIS,
        mcp_mode=MCPExecutionMode.SYNC,  # Quick LLM call, returns immediately
        service_method="summarize_node",
        requires_node_id=True,
        parameters=[
            ToolParameter(
                name="node_id",
                type="string",
                description="The node to summarize",
                required=True
            ),
            ToolParameter(
                name="max_length",
                type="integer",
                description="Maximum summary length in characters (50-1000)",
                default="200"
            ),
            ToolParameter(
                name="style",
                type="string",
                description="Summary style: 'concise', 'detailed', or 'bullet_points'",
                default="concise"
            ),
        ],
    ),
    ToolDefinition(
        name="summarize_node_with_context",
        description="Generate a context-aware summary including relationships. "
                    "Shows supporting/contradicting evidence and synthesis.",
        category=ToolCategory.NODE_ANALYSIS,
        mcp_mode=MCPExecutionMode.SYNC,  # Quick LLM call with context
        service_method="summarize_node_with_context",
        requires_node_id=True,
        parameters=[
            ToolParameter(
                name="node_id",
                type="string",
                description="The node to summarize",
                required=True
            ),
            ToolParameter(
                name="depth",
                type="integer",
                description="Relationship hops to include (1-2)",
                default="1"
            ),
            ToolParameter(
                name="relationship_types",
                type="array",
                description="Filter by relationship types (e.g., ['SUPPORTS', 'CONTRADICTS'])"
            ),
            ToolParameter(
                name="max_length",
                type="integer",
                description="Maximum summary length in characters (100-1000)",
                default="300"
            ),
        ],
    ),
    ToolDefinition(
        name="recalculate_node_confidence",
        description="Re-analyze a node's confidence based on graph context. "
                    "Considers content quality and connected evidence.",
        category=ToolCategory.NODE_ANALYSIS,
        mcp_mode=MCPExecutionMode.ASYNC,  # Updates database
        service_method="recalculate_node_confidence",
        requires_node_id=True,
        parameters=[
            ToolParameter(
                name="node_id",
                type="string",
                description="The node to recalculate confidence for",
                required=True
            ),
            ToolParameter(
                name="factor_in_relationships",
                type="boolean",
                description="Consider supporting/contradicting relationships",
                default="true"
            ),
        ],
    ),

    # -------------------------------------------------------------------------
    # Node Modification Tools
    # -------------------------------------------------------------------------
    ToolDefinition(
        name="reclassify_node",
        description="Change a node's type (e.g., Observation → Hypothesis). "
                    "Preserves properties and relationships.",
        category=ToolCategory.NODE_MODIFICATION,
        mcp_mode=MCPExecutionMode.ASYNC,  # Modifies database
        service_method="reclassify_node",
        requires_node_id=True,
        parameters=[
            ToolParameter(
                name="node_id",
                type="string",
                description="The node to reclassify",
                required=True
            ),
            ToolParameter(
                name="new_type",
                type="string",
                description="New type: Observation, Hypothesis, Question, Source, or Note",
                required=True
            ),
            ToolParameter(
                name="preserve_relationships",
                type="boolean",
                description="Keep existing relationships",
                default="true"
            ),
        ],
    ),
    ToolDefinition(
        name="search_web_evidence",
        description="Search the web for evidence related to a node. "
                    "Requires TAVILY_API_KEY configuration.",
        category=ToolCategory.NODE_MODIFICATION,
        mcp_mode=MCPExecutionMode.ASYNC,  # External API call
        service_method="search_web_evidence",
        requires_node_id=True,
        parameters=[
            ToolParameter(
                name="node_id",
                type="string",
                description="The node to find evidence for",
                required=True
            ),
            ToolParameter(
                name="evidence_type",
                type="string",
                description="Type of evidence: 'supporting' or 'contradicting'",
                default="supporting"
            ),
            ToolParameter(
                name="max_results",
                type="integer",
                description="Maximum number of results (1-20)",
                default="5"
            ),
            ToolParameter(
                name="auto_create_sources",
                type="boolean",
                description="Automatically create Source nodes from results",
                default="false"
            ),
        ],
    ),
    ToolDefinition(
        name="merge_nodes",
        description="Combine two nodes into one, transferring relationships. "
                    "Requires nodes to be the same type. THIS IS DESTRUCTIVE.",
        category=ToolCategory.NODE_MODIFICATION,
        mcp_mode=MCPExecutionMode.ASYNC,
        is_dangerous=True,  # Destructive operation
        service_method="merge_nodes",
        parameters=[
            ToolParameter(
                name="primary_node_id",
                type="string",
                description="The node to keep (receives merged content)",
                required=True
            ),
            ToolParameter(
                name="secondary_node_id",
                type="string",
                description="The node to merge and delete",
                required=True
            ),
            ToolParameter(
                name="merge_strategy",
                type="string",
                description="How to handle content: 'keep_primary', 'keep_secondary', "
                           "'combine', or 'smart' (AI-powered merge)",
                default="combine"
            ),
        ],
    ),

    # -------------------------------------------------------------------------
    # Relationship Analysis Tools
    # -------------------------------------------------------------------------
    ToolDefinition(
        name="summarize_relationship",
        description="Explain the connection between two nodes in plain language. "
                    "Includes evidence and strength assessment.",
        category=ToolCategory.RELATIONSHIP_ANALYSIS,
        mcp_mode=MCPExecutionMode.SYNC,  # Quick LLM call
        service_method="summarize_relationship",
        requires_edge_id=True,
        parameters=[
            ToolParameter(
                name="edge_id",
                type="string",
                description="The relationship ID to summarize",
                required=True
            ),
            ToolParameter(
                name="include_evidence",
                type="boolean",
                description="Include supporting evidence in summary",
                default="true"
            ),
        ],
    ),
    ToolDefinition(
        name="recalculate_edge_confidence",
        description="Re-analyze a relationship's confidence based on connected nodes. "
                    "Considers content alignment and graph structure.",
        category=ToolCategory.RELATIONSHIP_ANALYSIS,
        mcp_mode=MCPExecutionMode.ASYNC,  # Updates database
        service_method="recalculate_edge_confidence",
        requires_edge_id=True,
        parameters=[
            ToolParameter(
                name="edge_id",
                type="string",
                description="The relationship ID to recalculate",
                required=True
            ),
            ToolParameter(
                name="consider_graph_structure",
                type="boolean",
                description="Factor in broader graph context",
                default="true"
            ),
        ],
    ),
    ToolDefinition(
        name="reclassify_relationship",
        description="Change a relationship's type. Can use AI to suggest best type. "
                    "Types: SUPPORTS, CONTRADICTS, RELATES_TO, DERIVED_FROM, CITES.",
        category=ToolCategory.RELATIONSHIP_ANALYSIS,
        mcp_mode=MCPExecutionMode.ASYNC,  # Modifies database
        service_method="reclassify_relationship",
        requires_edge_id=True,
        parameters=[
            ToolParameter(
                name="edge_id",
                type="string",
                description="The relationship ID to reclassify",
                required=True
            ),
            ToolParameter(
                name="new_type",
                type="string",
                description="New type (or null for AI suggestion): "
                           "SUPPORTS, CONTRADICTS, RELATES_TO, DERIVED_FROM, CITES"
            ),
            ToolParameter(
                name="preserve_notes",
                type="boolean",
                description="Keep existing relationship notes",
                default="true"
            ),
        ],
    ),

    # -------------------------------------------------------------------------
    # Job Management Tools (MCP-only)
    # -------------------------------------------------------------------------
    ToolDefinition(
        name="check_job_status",
        description="Check the status of an async job. Returns status and result when complete.",
        category=ToolCategory.JOB_MANAGEMENT,
        mcp_mode=MCPExecutionMode.SYNC,  # Always returns immediately
        mcp_enabled=True,
        langgraph_enabled=False,  # LangGraph uses Activity Feed instead
        service_method="check_job_status",
        parameters=[
            ToolParameter(
                name="job_id",
                type="string",
                description="The job ID to check",
                required=True
            ),
        ],
    ),
    ToolDefinition(
        name="list_pending_jobs",
        description="List all in-progress and pending jobs for the current session.",
        category=ToolCategory.JOB_MANAGEMENT,
        mcp_mode=MCPExecutionMode.SYNC,
        mcp_enabled=True,
        langgraph_enabled=False,
        service_method="list_pending_jobs",
        parameters=[],
    ),
]


def get_tool_by_name(name: str) -> Optional[ToolDefinition]:
    """Get a tool definition by name."""
    for tool in TOOL_DEFINITIONS:
        if tool.name == name:
            return tool
    return None


def get_tools_by_category(category: ToolCategory) -> List[ToolDefinition]:
    """Get all tools in a category."""
    return [t for t in TOOL_DEFINITIONS if t.category == category]


def get_mcp_tools(include_dangerous: bool = False) -> List[ToolDefinition]:
    """Get tools available for MCP server.

    Args:
        include_dangerous: If True, include dangerous tools (admin mode)

    Returns:
        List of tool definitions enabled for MCP
    """
    tools = [t for t in TOOL_DEFINITIONS if t.mcp_enabled]
    if not include_dangerous:
        tools = [t for t in tools if not t.is_dangerous]
    return tools


def get_langgraph_tools() -> List[ToolDefinition]:
    """Get tools available for LangGraph agents.

    Note: All tools including dangerous ones are returned.
    Dangerous tool confirmation is handled at execution time.
    """
    return [t for t in TOOL_DEFINITIONS if t.langgraph_enabled]


def get_sync_tools() -> List[ToolDefinition]:
    """Get tools that execute synchronously in MCP."""
    return [t for t in TOOL_DEFINITIONS if t.mcp_mode == MCPExecutionMode.SYNC]


def get_async_tools() -> List[ToolDefinition]:
    """Get tools that execute asynchronously in MCP."""
    return [t for t in TOOL_DEFINITIONS if t.mcp_mode == MCPExecutionMode.ASYNC]


def get_dangerous_tools() -> List[ToolDefinition]:
    """Get tools marked as dangerous (require confirmation)."""
    return [t for t in TOOL_DEFINITIONS if t.is_dangerous]
