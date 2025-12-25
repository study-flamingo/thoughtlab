"""LangGraph tools that call ThoughtLab backend API endpoints.

These tools are thin HTTP clients that maintain separation between
the AI agent layer and the backend logic. All operations are performed
via REST API calls.
"""

from typing import Optional, List, Literal
from langchain_core.tools import tool
import httpx
import logging

logger = logging.getLogger(__name__)


# API base URL (configurable via environment)
API_BASE_URL = "http://localhost:8000/api/v1"


class ThoughtLabAPIError(Exception):
    """Error communicating with ThoughtLab API."""
    pass


async def _api_request(
    method: str,
    endpoint: str,
    json_data: Optional[dict] = None,
    timeout: float = 30.0,
) -> dict:
    """Make an HTTP request to the ThoughtLab API.

    Args:
        method: HTTP method (GET, POST, etc.)
        endpoint: API endpoint path
        json_data: Optional JSON data for request body
        timeout: Request timeout in seconds

    Returns:
        Response JSON data

    Raises:
        ThoughtLabAPIError: If request fails
    """
    url = f"{API_BASE_URL}{endpoint}"

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.request(
                method=method,
                url=url,
                json=json_data,
            )
            response.raise_for_status()
            return response.json()
    except httpx.HTTPError as e:
        logger.error(f"API request failed: {method} {url} - {e}")
        raise ThoughtLabAPIError(f"API request failed: {e}") from e


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
    try:
        result = await _api_request(
            "POST",
            f"/tools/nodes/{node_id}/find-related",
            json_data={
                "limit": limit,
                "min_similarity": min_similarity,
                "node_types": node_types,
                "auto_link": auto_link,
            }
        )

        if not result.get("success"):
            return f"Error: {result.get('error', 'Unknown error')}"

        related = result.get("related_nodes", [])
        if not related:
            return f"No related nodes found for {node_id} (min similarity: {min_similarity})"

        output = [f"Found {len(related)} related nodes for {node_id}:\n"]

        for i, node in enumerate(related, 1):
            output.append(
                f"{i}. {node['type']} ({node['id']}) - Similarity: {node['similarity_score']:.2f}\n"
                f"   Suggested Relationship: {node['suggested_relationship']}\n"
                f"   Reasoning: {node['reasoning']}\n"
                f"   Preview: {node['content'][:100]}...\n"
            )

        if result.get("links_created", 0) > 0:
            output.append(f"\n✓ Created {result['links_created']} relationship(s)")

        return "".join(output)

    except ThoughtLabAPIError as e:
        return f"Error calling API: {e}"


@tool
async def summarize_node(
    node_id: str,
    max_length: int = 200,
    style: Literal["concise", "detailed", "bullet_points"] = "concise",
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
    try:
        result = await _api_request(
            "POST",
            f"/tools/nodes/{node_id}/summarize",
            json_data={
                "max_length": max_length,
                "style": style,
            }
        )

        if not result.get("success"):
            return f"Error: {result.get('error', 'Unknown error')}"

        output = [f"Summary of {node_id}:\n\n"]
        output.append(result["summary"])

        if result.get("key_points"):
            output.append("\n\nKey Points:")
            for point in result["key_points"]:
                output.append(f"\n- {point}")

        output.append(f"\n\n(Word count: {result.get('word_count', 0)})")

        return "".join(output)

    except ThoughtLabAPIError as e:
        return f"Error calling API: {e}"


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
    try:
        result = await _api_request(
            "POST",
            f"/tools/nodes/{node_id}/summarize-with-context",
            json_data={
                "depth": depth,
                "relationship_types": relationship_types,
                "max_length": max_length,
            }
        )

        if not result.get("success"):
            return f"Error: {result.get('error', 'Unknown error')}"

        output = [f"Context-aware summary of {node_id}:\n\n"]
        output.append(result["summary"])

        context = result.get("context", {})

        if context.get("supports"):
            output.append(f"\n\n✓ Supporting Evidence ({len(context['supports'])}):")
            for item in context["supports"][:3]:  # Show top 3
                output.append(f"\n  - {item}")

        if context.get("contradicts"):
            output.append(f"\n\n✗ Contradicting Evidence ({len(context['contradicts'])}):")
            for item in context["contradicts"][:3]:
                output.append(f"\n  - {item}")

        if context.get("related"):
            output.append(f"\n\n~ Related ({len(context['related'])}):")
            for item in context["related"][:3]:
                output.append(f"\n  - {item}")

        if result.get("synthesis"):
            output.append(f"\n\nSynthesis:\n{result['synthesis']}")

        output.append(f"\n\n(Total relationships: {result.get('relationship_count', 0)})")

        return "".join(output)

    except ThoughtLabAPIError as e:
        return f"Error calling API: {e}"


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
    try:
        result = await _api_request(
            "POST",
            f"/tools/nodes/{node_id}/recalculate-confidence",
            json_data={
                "factor_in_relationships": factor_in_relationships,
            }
        )

        if not result.get("success"):
            return f"Error: {result.get('error', 'Unknown error')}"

        old = result.get("old_confidence", 0)
        new = result.get("new_confidence", 0)
        change = new - old

        output = [f"Confidence recalculation for {node_id}:\n\n"]
        output.append(f"Old confidence: {old:.2f}\n")
        output.append(f"New confidence: {new:.2f}\n")

        if change > 0:
            output.append(f"Change: +{change:.2f} ↑\n")
        elif change < 0:
            output.append(f"Change: {change:.2f} ↓\n")
        else:
            output.append("Change: No change\n")

        output.append(f"\nReasoning:\n{result.get('reasoning', 'N/A')}")

        factors = result.get("factors", [])
        if factors:
            output.append("\n\nFactors:")
            for factor in factors:
                output.append(f"\n- {factor['factor']}: {factor['impact']}")

        return "".join(output)

    except ThoughtLabAPIError as e:
        return f"Error calling API: {e}"


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
    try:
        result = await _api_request(
            "POST",
            f"/tools/relationships/{edge_id}/summarize",
            json_data={
                "include_evidence": include_evidence,
            }
        )

        if not result.get("success"):
            return f"Error: {result.get('error', 'Unknown error')}"

        from_node = result.get("from_node", {})
        to_node = result.get("to_node", {})
        rel_type = result.get("relationship_type", "UNKNOWN")
        strength = result.get("strength_assessment", "unknown")

        output = [
            f"Relationship Summary ({edge_id}):\n\n",
            f"From: {from_node.get('type')} ({from_node.get('id')})\n",
            f"To: {to_node.get('type')} ({to_node.get('id')})\n",
            f"Type: {rel_type}\n",
            f"Strength: {strength}\n\n",
            f"Explanation:\n{result.get('summary', 'N/A')}\n",
        ]

        evidence = result.get("evidence", [])
        if evidence:
            output.append("\nEvidence:")
            for item in evidence:
                output.append(f"\n- {item}")

        return "".join(output)

    except ThoughtLabAPIError as e:
        return f"Error calling API: {e}"


# ============================================================================
# Tool Registry
# ============================================================================

def get_thoughtlab_tools() -> list:
    """Get all ThoughtLab tools for LangGraph agent.

    Returns:
        List of LangChain tools that call ThoughtLab API
    """
    return [
        find_related_nodes,
        summarize_node,
        summarize_node_with_context,
        recalculate_node_confidence,
        summarize_relationship,
    ]
