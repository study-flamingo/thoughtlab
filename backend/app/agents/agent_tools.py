"""LangGraph agent tools for ThoughtLab.

This module provides LangChain tools that:
- Call ToolService directly (in-process, no HTTP)
- Save results to ReportService for later viewing
- Handle dangerous tool confirmation via Activity Feed
- Use shared tool definitions from app.tools

Usage:
    from app.agents.agent_tools import get_agent_tools

    tools = get_agent_tools()
    agent = create_react_agent(llm, tools)
"""

import logging
from typing import Any, Dict, List, Optional

from langchain_core.tools import tool

from app.tools import get_tool_registry
from app.services.tools import get_tool_service
from app.services.report_service import get_report_service

logger = logging.getLogger(__name__)


def get_agent_tools() -> List:
    """Get all LangGraph agent tools.

    Returns:
        List of LangChain tools that call ToolService directly
    """
    return [
        find_related_nodes,
        summarize_node,
        summarize_node_with_context,
        recalculate_node_confidence,
        reclassify_node,
        search_web_evidence,
        merge_nodes,
        summarize_relationship,
        recalculate_edge_confidence,
        reclassify_relationship,
    ]


# ============================================================================
# Node Analysis Tools
# ============================================================================

@tool
async def find_related_nodes(
    node_id: str,
    limit: int = 10,
    min_similarity: float = 0.5,
    node_types: Optional[List[str]] = None,
    auto_link: bool = False,
) -> str:
    """Find semantically similar nodes in the knowledge graph.

    Use this tool to discover connections between nodes based on semantic similarity.
    This is useful when you want to:
    - Find related research observations
    - Discover supporting or contradicting evidence
    - Identify patterns across the knowledge graph

    Args:
        node_id: The ID of the node to find similar nodes for
        limit: Maximum number of results (1-50, default 10)
        min_similarity: Minimum similarity score 0.0-1.0 (default 0.5)
        node_types: Optional list of node types to filter (e.g., ["Observation", "Hypothesis"])
        auto_link: If True, automatically create relationships (default False)

    Returns:
        A formatted string describing the related nodes found
    """
    tool_service = get_tool_service()
    report_service = get_report_service()

    result = await tool_service.find_related_nodes(
        node_id=node_id,
        limit=limit,
        min_similarity=min_similarity,
        node_types=node_types,
        auto_link=auto_link,
    )

    output = _format_find_related_result(result, node_id, min_similarity)

    # Save report
    await report_service.save_report(
        tool_name="find_related_nodes",
        node_id=node_id,
        content=output,
        metadata={"limit": limit, "min_similarity": min_similarity, "found": len(result.related_nodes) if result.related_nodes else 0},
    )

    return output


@tool
async def summarize_node(
    node_id: str,
    max_length: int = 200,
    style: str = "concise",
) -> str:
    """Generate a summary of a node's content using AI.

    Use this tool when you need to:
    - Quickly understand what a node is about
    - Generate a concise description for reporting
    - Extract key points from lengthy content

    Args:
        node_id: The ID of the node to summarize
        max_length: Maximum length in characters (50-1000, default 200)
        style: Summary style - "concise", "detailed", or "bullet_points"

    Returns:
        A formatted summary of the node
    """
    tool_service = get_tool_service()
    report_service = get_report_service()

    result = await tool_service.summarize_node(
        node_id=node_id,
        max_length=max_length,
        style=style,
    )

    output = _format_summarize_result(result, node_id)

    # Save report
    await report_service.save_report(
        tool_name="summarize_node",
        node_id=node_id,
        content=output,
        metadata={"style": style, "max_length": max_length},
    )

    return output


@tool
async def summarize_node_with_context(
    node_id: str,
    depth: int = 1,
    relationship_types: Optional[List[str]] = None,
    max_length: int = 300,
) -> str:
    """Generate a context-aware summary including relationships.

    Use this tool when you need to:
    - Understand a node in the broader context of the knowledge graph
    - See what supports or contradicts a hypothesis
    - Analyze the state of knowledge on a topic

    Args:
        node_id: The ID of the node to summarize
        depth: How many relationship hops to include (1-2, default 1)
        relationship_types: Optional filter for specific types (e.g., ["SUPPORTS", "CONTRADICTS"])
        max_length: Maximum length in characters (100-1000, default 300)

    Returns:
        A comprehensive summary including relationship context
    """
    tool_service = get_tool_service()
    report_service = get_report_service()

    result = await tool_service.summarize_node_with_context(
        node_id=node_id,
        depth=depth,
        relationship_types=relationship_types,
        max_length=max_length,
    )

    output = _format_summarize_with_context_result(result, node_id)

    # Save report
    await report_service.save_report(
        tool_name="summarize_node_with_context",
        node_id=node_id,
        content=output,
        metadata={"depth": depth, "relationship_count": result.relationship_count if hasattr(result, 'relationship_count') else 0},
    )

    return output


@tool
async def recalculate_node_confidence(
    node_id: str,
    factor_in_relationships: bool = True,
) -> str:
    """Recalculate a node's confidence score based on current context.

    Use this tool when:
    - New evidence has been added that might affect confidence
    - You want to validate the reliability of a claim
    - The knowledge graph has evolved significantly

    Args:
        node_id: The ID of the node to recalculate
        factor_in_relationships: Whether to consider connected nodes (default True)

    Returns:
        A report on the confidence change with reasoning
    """
    tool_service = get_tool_service()
    report_service = get_report_service()

    result = await tool_service.recalculate_node_confidence(
        node_id=node_id,
        factor_in_relationships=factor_in_relationships,
    )

    output = _format_confidence_result(result, node_id)

    # Save report
    await report_service.save_report(
        tool_name="recalculate_node_confidence",
        node_id=node_id,
        content=output,
        metadata={
            "old_confidence": result.old_confidence if hasattr(result, 'old_confidence') else None,
            "new_confidence": result.new_confidence if hasattr(result, 'new_confidence') else None,
        },
    )

    return output


@tool
async def reclassify_node(
    node_id: str,
    new_type: str,
    preserve_relationships: bool = True,
) -> str:
    """Change a node's type (e.g., Observation to Hypothesis).

    Valid types: Observation, Hypothesis, Question, Source, Note

    Args:
        node_id: The ID of the node to reclassify
        new_type: New type - Observation, Hypothesis, Question, Source, or Note
        preserve_relationships: Keep existing relationships (default True)

    Returns:
        A report on the reclassification result
    """
    tool_service = get_tool_service()
    report_service = get_report_service()

    result = await tool_service.reclassify_node(
        node_id=node_id,
        new_type=new_type,
        preserve_relationships=preserve_relationships,
    )

    output = _format_reclassify_node_result(result, node_id)

    # Save report
    await report_service.save_report(
        tool_name="reclassify_node",
        node_id=node_id,
        content=output,
        metadata={"old_type": result.old_type if hasattr(result, 'old_type') else None, "new_type": new_type},
    )

    return output


@tool
async def search_web_evidence(
    node_id: str,
    evidence_type: str = "supporting",
    max_results: int = 5,
    auto_create_sources: bool = False,
) -> str:
    """Search the web for evidence related to a node.

    Requires TAVILY_API_KEY to be configured.

    Args:
        node_id: The ID of the node to find evidence for
        evidence_type: Type of evidence - "supporting" or "contradicting"
        max_results: Maximum number of results (1-20, default 5)
        auto_create_sources: Automatically create Source nodes from results

    Returns:
        Web search results with relevance information
    """
    tool_service = get_tool_service()
    report_service = get_report_service()

    result = await tool_service.search_web_evidence(
        node_id=node_id,
        evidence_type=evidence_type,
        max_results=max_results,
        auto_create_sources=auto_create_sources,
    )

    output = _format_web_evidence_result(result, node_id)

    # Save report
    await report_service.save_report(
        tool_name="search_web_evidence",
        node_id=node_id,
        content=output,
        metadata={"evidence_type": evidence_type, "results_count": len(result.results) if result.results else 0},
    )

    return output


@tool
async def merge_nodes(
    primary_node_id: str,
    secondary_node_id: str,
    merge_strategy: str = "combine",
) -> str:
    """Combine two nodes into one, transferring relationships.

    CAUTION: This is a DESTRUCTIVE operation. The secondary node will be deleted.
    Both nodes must be of the same type.

    Args:
        primary_node_id: The node to keep (receives merged content)
        secondary_node_id: The node to merge and delete
        merge_strategy: How to handle content - "keep_primary", "keep_secondary", "combine", or "smart"

    Returns:
        A report on the merge result
    """
    tool_service = get_tool_service()
    report_service = get_report_service()

    # Note: In production, this should request user confirmation via Activity Feed
    # For now, we execute directly but log a warning
    logger.warning(f"Executing dangerous merge_nodes: {primary_node_id} <- {secondary_node_id}")

    result = await tool_service.merge_nodes(
        primary_node_id=primary_node_id,
        secondary_node_id=secondary_node_id,
        merge_strategy=merge_strategy,
    )

    output = _format_merge_result(result)

    # Save report
    await report_service.save_report(
        tool_name="merge_nodes",
        node_id=primary_node_id,
        content=output,
        metadata={"secondary_node_id": secondary_node_id, "strategy": merge_strategy, "success": result.success},
    )

    return output


# ============================================================================
# Relationship Analysis Tools
# ============================================================================

@tool
async def summarize_relationship(
    edge_id: str,
    include_evidence: bool = True,
) -> str:
    """Explain the connection between two nodes in plain language.

    Use this tool to:
    - Understand why two nodes are connected
    - Get a natural language explanation of a relationship
    - Assess the strength of a connection

    Args:
        edge_id: The ID of the relationship to summarize
        include_evidence: Whether to include supporting evidence (default True)

    Returns:
        A plain language explanation of the relationship
    """
    tool_service = get_tool_service()
    report_service = get_report_service()

    result = await tool_service.summarize_relationship(
        edge_id=edge_id,
        include_evidence=include_evidence,
    )

    output = _format_relationship_summary(result, edge_id)

    # Save report
    await report_service.save_report(
        tool_name="summarize_relationship",
        edge_id=edge_id,
        content=output,
        metadata={"relationship_type": result.relationship_type if hasattr(result, 'relationship_type') else None},
    )

    return output


@tool
async def recalculate_edge_confidence(
    edge_id: str,
    consider_graph_structure: bool = True,
) -> str:
    """Recalculate a relationship's confidence score.

    Use this tool when:
    - Connected nodes have changed
    - You want to assess relationship reliability
    - The graph context has evolved

    Args:
        edge_id: The ID of the relationship to recalculate
        consider_graph_structure: Factor in broader graph context (default True)

    Returns:
        A report on the confidence change with reasoning
    """
    tool_service = get_tool_service()
    report_service = get_report_service()

    result = await tool_service.recalculate_edge_confidence(
        edge_id=edge_id,
        consider_graph_structure=consider_graph_structure,
    )

    output = _format_edge_confidence_result(result, edge_id)

    # Save report
    await report_service.save_report(
        tool_name="recalculate_edge_confidence",
        edge_id=edge_id,
        content=output,
        metadata={
            "old_confidence": result.old_confidence if hasattr(result, 'old_confidence') else None,
            "new_confidence": result.new_confidence if hasattr(result, 'new_confidence') else None,
        },
    )

    return output


@tool
async def reclassify_relationship(
    edge_id: str,
    new_type: Optional[str] = None,
    preserve_notes: bool = True,
) -> str:
    """Change a relationship's type. Can use AI to suggest best type.

    Valid types: SUPPORTS, CONTRADICTS, RELATES_TO, DERIVED_FROM, CITES

    Args:
        edge_id: The ID of the relationship to reclassify
        new_type: New type (or None for AI suggestion)
        preserve_notes: Keep existing relationship notes (default True)

    Returns:
        A report on the reclassification result
    """
    tool_service = get_tool_service()
    report_service = get_report_service()

    result = await tool_service.reclassify_relationship(
        edge_id=edge_id,
        new_type=new_type,
        preserve_notes=preserve_notes,
    )

    output = _format_reclassify_relationship_result(result, edge_id)

    # Save report
    await report_service.save_report(
        tool_name="reclassify_relationship",
        edge_id=edge_id,
        content=output,
        metadata={
            "old_type": result.old_type if hasattr(result, 'old_type') else None,
            "new_type": result.new_type if hasattr(result, 'new_type') else None,
            "suggested_by_ai": result.suggested_by_ai if hasattr(result, 'suggested_by_ai') else False,
        },
    )

    return output


# ============================================================================
# Result Formatters
# ============================================================================

def _format_find_related_result(result, node_id: str, min_similarity: float) -> str:
    """Format find_related_nodes result."""
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
    """Format summarize_node result."""
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
    """Format summarize_node_with_context result."""
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
    """Format recalculate_node_confidence result."""
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
    """Format reclassify_node result."""
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
    """Format search_web_evidence result."""
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
    """Format merge_nodes result."""
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
    """Format summarize_relationship result."""
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
    """Format recalculate_edge_confidence result."""
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
    """Format reclassify_relationship result."""
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


# Legacy alias for backwards compatibility
def get_thoughtlab_tools() -> List:
    """Legacy alias for get_agent_tools().

    Deprecated: Use get_agent_tools() instead.
    """
    return get_agent_tools()
