"""MCP tool wrappers for ThoughtLab.

This module provides tool wrappers that:
- Call ToolService directly (in-process, no HTTP)
- Handle sync/async execution based on tool definition
- Gate dangerous tools based on admin mode
- Queue async jobs for polling via check_job_status

Usage:
    from app.mcp.mcp_tools import register_mcp_tools

    mcp = FastMCP("ThoughtLab")
    register_mcp_tools(mcp)
"""

import os
import logging
from typing import Any, Dict, List, Optional

from fastmcp import FastMCP

from app.tools import get_tool_registry, ToolCategory, MCPExecutionMode
from app.services.tools import get_tool_service
from app.services.job_service import get_job_service
from app.models.job_models import JobStatus

logger = logging.getLogger(__name__)

# Admin mode enables dangerous tools
ADMIN_MODE = os.getenv("THOUGHTLAB_MCP_ADMIN_MODE", "false").lower() == "true"


def register_mcp_tools(mcp: FastMCP) -> None:
    """Register all MCP tools from tool definitions.

    Args:
        mcp: FastMCP server instance
    """
    registry = get_tool_registry()
    tool_service = get_tool_service()
    job_service = get_job_service()

    # Get MCP-enabled tools
    tools = registry.list_tools(mcp_only=True, include_dangerous=ADMIN_MODE)

    logger.info(f"Registering {len(tools)} MCP tools (admin_mode={ADMIN_MODE})")

    # Register each tool
    for tool_def in tools:
        _register_tool(mcp, tool_def, tool_service, job_service)

    # Always register job management tools
    _register_job_tools(mcp, job_service)


def _register_tool(
    mcp: FastMCP,
    tool_def,
    tool_service,
    job_service,
) -> None:
    """Register a single tool with the MCP server.

    Args:
        mcp: FastMCP server instance
        tool_def: Tool definition
        tool_service: ToolService instance
        job_service: JobService instance
    """
    # Skip job management tools (registered separately)
    if tool_def.category == ToolCategory.JOB_MANAGEMENT:
        return

    tool_name = tool_def.name
    is_sync = tool_def.mcp_mode == MCPExecutionMode.SYNC

    # Create the tool function dynamically
    if tool_name == "find_related_nodes":
        @mcp.tool(name=tool_name, description=tool_def.description)
        async def find_related_nodes(
            node_id: str,
            limit: int = 10,
            min_similarity: float = 0.5,
            node_types: Optional[List[str]] = None,
            auto_link: bool = False,
        ) -> str:
            """Find semantically similar nodes in the knowledge graph."""
            if is_sync:
                result = await tool_service.find_related_nodes(
                    node_id=node_id,
                    limit=limit,
                    min_similarity=min_similarity,
                    node_types=node_types,
                    auto_link=auto_link,
                )
                return _format_find_related_result(result, node_id, min_similarity)
            else:
                job_id = await job_service.queue_job(
                    "find_related_nodes",
                    {"node_id": node_id, "limit": limit, "min_similarity": min_similarity,
                     "node_types": node_types, "auto_link": auto_link}
                )
                return f"Job queued: {job_id}. Use check_job_status('{job_id}') to get results."

    elif tool_name == "summarize_node":
        @mcp.tool(name=tool_name, description=tool_def.description)
        async def summarize_node(
            node_id: str,
            max_length: int = 200,
            style: str = "concise",
        ) -> str:
            """Generate a summary of a node's content using AI."""
            result = await tool_service.summarize_node(
                node_id=node_id,
                max_length=max_length,
                style=style,
            )
            return _format_summarize_result(result, node_id)

    elif tool_name == "summarize_node_with_context":
        @mcp.tool(name=tool_name, description=tool_def.description)
        async def summarize_node_with_context(
            node_id: str,
            depth: int = 1,
            relationship_types: Optional[List[str]] = None,
            max_length: int = 300,
        ) -> str:
            """Generate a context-aware summary including relationships."""
            result = await tool_service.summarize_node_with_context(
                node_id=node_id,
                depth=depth,
                relationship_types=relationship_types,
                max_length=max_length,
            )
            return _format_summarize_with_context_result(result, node_id)

    elif tool_name == "recalculate_node_confidence":
        @mcp.tool(name=tool_name, description=tool_def.description)
        async def recalculate_node_confidence(
            node_id: str,
            factor_in_relationships: bool = True,
        ) -> str:
            """Recalculate a node's confidence score based on current context."""
            if is_sync:
                result = await tool_service.recalculate_node_confidence(
                    node_id=node_id,
                    factor_in_relationships=factor_in_relationships,
                )
                return _format_confidence_result(result, node_id)
            else:
                job_id = await job_service.queue_job(
                    "recalculate_node_confidence",
                    {"node_id": node_id, "factor_in_relationships": factor_in_relationships}
                )
                return f"Job queued: {job_id}. Use check_job_status('{job_id}') to get results."

    elif tool_name == "reclassify_node":
        @mcp.tool(name=tool_name, description=tool_def.description)
        async def reclassify_node(
            node_id: str,
            new_type: str,
            preserve_relationships: bool = True,
        ) -> str:
            """Change a node's type (e.g., Observation to Hypothesis)."""
            if is_sync:
                result = await tool_service.reclassify_node(
                    node_id=node_id,
                    new_type=new_type,
                    preserve_relationships=preserve_relationships,
                )
                return _format_reclassify_node_result(result, node_id)
            else:
                job_id = await job_service.queue_job(
                    "reclassify_node",
                    {"node_id": node_id, "new_type": new_type, "preserve_relationships": preserve_relationships}
                )
                return f"Job queued: {job_id}. Use check_job_status('{job_id}') to get results."

    elif tool_name == "search_web_evidence":
        @mcp.tool(name=tool_name, description=tool_def.description)
        async def search_web_evidence(
            node_id: str,
            evidence_type: str = "supporting",
            max_results: int = 5,
            auto_create_sources: bool = False,
        ) -> str:
            """Search the web for evidence related to a node."""
            if is_sync:
                result = await tool_service.search_web_evidence(
                    node_id=node_id,
                    evidence_type=evidence_type,
                    max_results=max_results,
                    auto_create_sources=auto_create_sources,
                )
                return _format_web_evidence_result(result, node_id)
            else:
                job_id = await job_service.queue_job(
                    "search_web_evidence",
                    {"node_id": node_id, "evidence_type": evidence_type,
                     "max_results": max_results, "auto_create_sources": auto_create_sources}
                )
                return f"Job queued: {job_id}. Use check_job_status('{job_id}') to get results."

    elif tool_name == "merge_nodes":
        @mcp.tool(name=tool_name, description=tool_def.description)
        async def merge_nodes(
            primary_node_id: str,
            secondary_node_id: str,
            merge_strategy: str = "combine",
        ) -> str:
            """Combine two nodes into one, transferring relationships. DESTRUCTIVE."""
            if is_sync:
                result = await tool_service.merge_nodes(
                    primary_node_id=primary_node_id,
                    secondary_node_id=secondary_node_id,
                    merge_strategy=merge_strategy,
                )
                return _format_merge_result(result)
            else:
                job_id = await job_service.queue_job(
                    "merge_nodes",
                    {"primary_node_id": primary_node_id, "secondary_node_id": secondary_node_id,
                     "merge_strategy": merge_strategy}
                )
                return f"Job queued: {job_id}. Use check_job_status('{job_id}') to get results."

    elif tool_name == "summarize_relationship":
        @mcp.tool(name=tool_name, description=tool_def.description)
        async def summarize_relationship(
            edge_id: str,
            include_evidence: bool = True,
        ) -> str:
            """Explain the connection between two nodes in plain language."""
            result = await tool_service.summarize_relationship(
                edge_id=edge_id,
                include_evidence=include_evidence,
            )
            return _format_relationship_summary(result, edge_id)

    elif tool_name == "recalculate_edge_confidence":
        @mcp.tool(name=tool_name, description=tool_def.description)
        async def recalculate_edge_confidence(
            edge_id: str,
            consider_graph_structure: bool = True,
        ) -> str:
            """Recalculate a relationship's confidence score."""
            if is_sync:
                result = await tool_service.recalculate_edge_confidence(
                    edge_id=edge_id,
                    consider_graph_structure=consider_graph_structure,
                )
                return _format_edge_confidence_result(result, edge_id)
            else:
                job_id = await job_service.queue_job(
                    "recalculate_edge_confidence",
                    {"edge_id": edge_id, "consider_graph_structure": consider_graph_structure}
                )
                return f"Job queued: {job_id}. Use check_job_status('{job_id}') to get results."

    elif tool_name == "reclassify_relationship":
        @mcp.tool(name=tool_name, description=tool_def.description)
        async def reclassify_relationship(
            edge_id: str,
            new_type: Optional[str] = None,
            preserve_notes: bool = True,
        ) -> str:
            """Change a relationship's type. Can use AI to suggest best type."""
            if is_sync:
                result = await tool_service.reclassify_relationship(
                    edge_id=edge_id,
                    new_type=new_type,
                    preserve_notes=preserve_notes,
                )
                return _format_reclassify_relationship_result(result, edge_id)
            else:
                job_id = await job_service.queue_job(
                    "reclassify_relationship",
                    {"edge_id": edge_id, "new_type": new_type, "preserve_notes": preserve_notes}
                )
                return f"Job queued: {job_id}. Use check_job_status('{job_id}') to get results."

    else:
        logger.warning(f"Unknown tool: {tool_name}, skipping registration")


def _register_job_tools(mcp: FastMCP, job_service) -> None:
    """Register job management tools.

    Args:
        mcp: FastMCP server instance
        job_service: JobService instance
    """
    @mcp.tool(name="check_job_status", description="Check the status of an async job. Returns status and result when complete.")
    async def check_job_status(job_id: str) -> str:
        """Check the status of an async job."""
        job = await job_service.get_job(job_id)
        if not job:
            return f"Job not found: {job_id}"

        output = [f"Job Status: {job.status.value}\n"]
        output.append(f"Tool: {job.tool_name}\n")
        output.append(f"Created: {job.created_at.isoformat()}\n")

        if job.started_at:
            output.append(f"Started: {job.started_at.isoformat()}\n")
        if job.completed_at:
            output.append(f"Completed: {job.completed_at.isoformat()}\n")

        if job.status == JobStatus.COMPLETED and job.result:
            output.append(f"\nResult:\n{_format_job_result(job.tool_name, job.result)}")
        elif job.status == JobStatus.FAILED and job.error:
            output.append(f"\nError: {job.error}")
        elif job.status == JobStatus.WAITING_APPROVAL:
            output.append("\nWaiting for user approval...")

        return "".join(output)

    @mcp.tool(name="list_pending_jobs", description="List all in-progress and pending jobs.")
    async def list_pending_jobs() -> str:
        """List all pending and in-progress jobs."""
        result = await job_service.list_pending_jobs()

        if not result.jobs:
            return "No pending jobs."

        output = [f"Pending Jobs ({result.total}):\n\n"]
        for job in result.jobs:
            output.append(f"- {job.id}: {job.tool_name} ({job.status.value})\n")

        return "".join(output)


# ============================================================================
# Result Formatters
# ============================================================================

def _format_find_related_result(result, node_id: str, min_similarity: float) -> str:
    """Format find_related_nodes result for MCP output."""
    if not result.success:
        return f"Error: {result.error or result.message}"

    if not result.related_nodes:
        return f"No related nodes found for {node_id} (min similarity: {min_similarity})"

    output = [f"Found {len(result.related_nodes)} related nodes for {node_id}:\n"]

    for i, node in enumerate(result.related_nodes, 1):
        content_preview = node.content[:100] + "..." if len(node.content) > 100 else node.content
        output.append(
            f"\n{i}. {node.type} ({node.id}) - Similarity: {node.similarity_score:.2f}\n"
            f"   Suggested: {node.suggested_relationship}\n"
            f"   Reasoning: {node.reasoning}\n"
            f"   Preview: {content_preview}\n"
        )

    if result.links_created and result.links_created > 0:
        output.append(f"\nCreated {result.links_created} relationship(s)")

    return "".join(output)


def _format_summarize_result(result, node_id: str) -> str:
    """Format summarize_node result for MCP output."""
    if not result.success:
        return f"Error: {result.error or 'Unknown error'}"

    output = [f"Summary of {node_id}:\n\n{result.summary}"]

    if result.key_points:
        output.append("\n\nKey Points:")
        for point in result.key_points:
            output.append(f"\n- {point}")

    output.append(f"\n\n(Word count: {result.word_count})")

    return "".join(output)


def _format_summarize_with_context_result(result, node_id: str) -> str:
    """Format summarize_node_with_context result for MCP output."""
    if not result.success:
        return f"Error: {result.error or 'Unknown error'}"

    output = [f"Context-aware summary of {node_id}:\n\n{result.summary}"]

    if result.context:
        if result.context.supports:
            output.append(f"\n\nSupporting Evidence ({len(result.context.supports)}):")
            for item in result.context.supports[:3]:
                output.append(f"\n  - {item}")

        if result.context.contradicts:
            output.append(f"\n\nContradicting Evidence ({len(result.context.contradicts)}):")
            for item in result.context.contradicts[:3]:
                output.append(f"\n  - {item}")

        if result.context.related:
            output.append(f"\n\nRelated ({len(result.context.related)}):")
            for item in result.context.related[:3]:
                output.append(f"\n  - {item}")

    if result.synthesis:
        output.append(f"\n\nSynthesis:\n{result.synthesis}")

    output.append(f"\n\n(Total relationships: {result.relationship_count})")

    return "".join(output)


def _format_confidence_result(result, node_id: str) -> str:
    """Format recalculate_node_confidence result for MCP output."""
    if not result.success:
        return f"Error: {result.error or 'Unknown error'}"

    change = result.new_confidence - result.old_confidence
    change_str = f"+{change:.2f}" if change > 0 else f"{change:.2f}"

    output = [
        f"Confidence recalculation for {node_id}:\n\n",
        f"Old confidence: {result.old_confidence:.2f}\n",
        f"New confidence: {result.new_confidence:.2f}\n",
        f"Change: {change_str}\n\n",
        f"Reasoning:\n{result.reasoning}",
    ]

    if result.factors:
        output.append("\n\nFactors:")
        for factor in result.factors:
            output.append(f"\n- {factor.factor}: {factor.impact}")

    return "".join(output)


def _format_reclassify_node_result(result, node_id: str) -> str:
    """Format reclassify_node result for MCP output."""
    if not result.success:
        return f"Error: {result.error or result.message}"

    output = [
        f"Reclassified node {node_id}:\n\n",
        f"Old type: {result.old_type}\n",
        f"New type: {result.new_type}\n",
        f"Relationships preserved: {result.relationships_preserved}\n",
    ]

    if result.properties_preserved:
        output.append(f"Properties preserved: {', '.join(result.properties_preserved)}\n")

    output.append(f"\n{result.message}")

    return "".join(output)


def _format_web_evidence_result(result, node_id: str) -> str:
    """Format search_web_evidence result for MCP output."""
    if not result.success:
        return f"Error: {result.error or result.message}"

    if not result.results:
        return f"No web evidence found for {node_id}. Query: {result.query_used}"

    output = [
        f"Web evidence search for {node_id}:\n",
        f"Query: {result.query_used}\n\n",
        f"Found {len(result.results)} results:\n",
    ]

    for i, item in enumerate(result.results, 1):
        output.append(
            f"\n{i}. {item.title}\n"
            f"   URL: {item.url}\n"
            f"   Snippet: {item.snippet[:150]}...\n"
        )

    if result.sources_created > 0:
        output.append(f"\nCreated {result.sources_created} source node(s)")

    return "".join(output)


def _format_merge_result(result) -> str:
    """Format merge_nodes result for MCP output."""
    if not result.success:
        return f"Error: {result.error or result.message}"

    output = [
        f"Merged nodes:\n\n",
        f"Primary (kept): {result.primary_node_id}\n",
        f"Secondary (deleted): {result.secondary_node_id}\n",
        f"Relationships transferred: {result.relationships_transferred}\n",
    ]

    if result.merged_properties:
        output.append(f"Properties merged: {', '.join(result.merged_properties)}\n")

    output.append(f"\n{result.message}")

    return "".join(output)


def _format_relationship_summary(result, edge_id: str) -> str:
    """Format summarize_relationship result for MCP output."""
    if not result.success:
        return f"Error: {result.error or 'Unknown error'}"

    output = [
        f"Relationship Summary ({edge_id}):\n\n",
        f"From: {result.from_node.type} ({result.from_node.id})\n",
        f"To: {result.to_node.type} ({result.to_node.id})\n",
        f"Type: {result.relationship_type}\n",
        f"Strength: {result.strength_assessment}\n\n",
        f"Explanation:\n{result.summary}\n",
    ]

    if result.evidence:
        output.append("\nEvidence:")
        for item in result.evidence:
            output.append(f"\n- {item}")

    return "".join(output)


def _format_edge_confidence_result(result, edge_id: str) -> str:
    """Format recalculate_edge_confidence result for MCP output."""
    if not result.success:
        return f"Error: {result.error or 'Unknown error'}"

    change = result.new_confidence - result.old_confidence
    change_str = f"+{change:.2f}" if change > 0 else f"{change:.2f}"

    output = [
        f"Edge confidence recalculation for {edge_id}:\n\n",
        f"Old confidence: {result.old_confidence:.2f}\n",
        f"New confidence: {result.new_confidence:.2f}\n",
        f"Change: {change_str}\n\n",
        f"Reasoning:\n{result.reasoning}",
    ]

    if result.factors:
        output.append("\n\nFactors:")
        for factor in result.factors:
            output.append(f"\n- {factor.factor}: {factor.impact}")

    return "".join(output)


def _format_reclassify_relationship_result(result, edge_id: str) -> str:
    """Format reclassify_relationship result for MCP output."""
    if not result.success:
        return f"Error: {result.error or 'Unknown error'}"

    output = [
        f"Reclassified relationship {edge_id}:\n\n",
        f"Old type: {result.old_type}\n",
        f"New type: {result.new_type}\n",
    ]

    if result.suggested_by_ai:
        output.append("(Suggested by AI)\n")

    if result.reasoning:
        output.append(f"\nReasoning: {result.reasoning}")

    return "".join(output)


def _format_job_result(tool_name: str, result: Dict[str, Any]) -> str:
    """Format job result based on tool type."""
    # Convert dict back to a simple string representation
    if result.get("success"):
        if "summary" in result:
            return result["summary"]
        elif "related_nodes" in result:
            nodes = result["related_nodes"]
            return f"Found {len(nodes)} related nodes"
        elif "new_confidence" in result:
            return f"New confidence: {result['new_confidence']:.2f}"
        elif "new_type" in result:
            return f"Reclassified to: {result['new_type']}"
        elif "message" in result:
            return result["message"]
    else:
        return result.get("error", "Unknown error")

    return str(result)
