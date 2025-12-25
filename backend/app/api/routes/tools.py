"""API routes for LLM-powered graph operation tools.

These endpoints provide on-demand operations that can be invoked by:
- LangGraph agents
- Frontend components
- MCP server
- CLI tools

All endpoints are stateless and independently testable.
"""

from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel, Field
from typing import Optional, List, Literal

from app.services.tool_service import (
    get_tool_service,
    FindRelatedNodesResponse,
    SummarizeNodeResponse,
    SummarizeNodeWithContextResponse,
    RecalculateConfidenceResponse,
    SummarizeRelationshipResponse,
)

router = APIRouter(prefix="/tools", tags=["tools"])
tool_service = get_tool_service()


# ============================================================================
# Request Models
# ============================================================================

class FindRelatedNodesRequest(BaseModel):
    """Request for finding related nodes."""
    limit: int = Field(default=10, ge=1, le=50)
    min_similarity: float = Field(default=0.5, ge=0.0, le=1.0)
    node_types: Optional[List[str]] = None
    auto_link: bool = False


class SummarizeNodeRequest(BaseModel):
    """Request for summarizing a node."""
    max_length: int = Field(default=200, ge=50, le=1000)
    style: Literal["concise", "detailed", "bullet_points"] = "concise"


class SummarizeNodeWithContextRequest(BaseModel):
    """Request for summarizing a node with context."""
    depth: int = Field(default=1, ge=1, le=2)
    relationship_types: Optional[List[str]] = None
    max_length: int = Field(default=300, ge=100, le=1000)


class RecalculateConfidenceRequest(BaseModel):
    """Request for recalculating node confidence."""
    factor_in_relationships: bool = True


class SummarizeRelationshipRequest(BaseModel):
    """Request for summarizing a relationship."""
    include_evidence: bool = True


# ============================================================================
# Node Analysis Endpoints
# ============================================================================

@router.post("/nodes/{node_id}/find-related", response_model=FindRelatedNodesResponse)
async def find_related_nodes(
    node_id: str,
    request: FindRelatedNodesRequest = Body(...),
) -> FindRelatedNodesResponse:
    """Find semantically similar nodes using vector embeddings.

    This endpoint:
    - Uses vector similarity search to find related content
    - Classifies the type of relationship between nodes
    - Optionally creates relationships automatically

    Args:
        node_id: The node to find similar nodes for
        request: Search parameters

    Returns:
        List of related nodes with similarity scores and suggested relationships

    Example:
        POST /api/v1/tools/nodes/obs-123/find-related
        {
            "limit": 10,
            "min_similarity": 0.6,
            "node_types": ["Observation", "Hypothesis"],
            "auto_link": false
        }
    """
    result = await tool_service.find_related_nodes(
        node_id=node_id,
        limit=request.limit,
        min_similarity=request.min_similarity,
        node_types=request.node_types,
        auto_link=request.auto_link,
    )

    if not result.success and result.error:
        status_code = 404 if "not found" in result.error.lower() else 500
        raise HTTPException(status_code=status_code, detail=result.error)

    return result


@router.post("/nodes/{node_id}/summarize", response_model=SummarizeNodeResponse)
async def summarize_node(
    node_id: str,
    request: SummarizeNodeRequest = Body(...),
) -> SummarizeNodeResponse:
    """Generate an LLM-powered summary of a node's content.

    This endpoint:
    - Extracts the main content from a node
    - Uses LLM to generate a concise summary
    - Identifies key points

    Args:
        node_id: The node to summarize
        request: Summary parameters

    Returns:
        Summary text, key points, and metadata

    Example:
        POST /api/v1/tools/nodes/obs-123/summarize
        {
            "max_length": 200,
            "style": "concise"
        }
    """
    result = await tool_service.summarize_node(
        node_id=node_id,
        max_length=request.max_length,
        style=request.style,
    )

    if not result.success and result.error:
        status_code = 404 if "not found" in result.error.lower() else 500
        raise HTTPException(status_code=status_code, detail=result.error)

    return result


@router.post(
    "/nodes/{node_id}/summarize-with-context",
    response_model=SummarizeNodeWithContextResponse
)
async def summarize_node_with_context(
    node_id: str,
    request: SummarizeNodeWithContextRequest = Body(...),
) -> SummarizeNodeWithContextResponse:
    """Generate a context-aware summary including relationships.

    This endpoint:
    - Summarizes the node's content
    - Includes information about connected nodes
    - Organizes context by relationship type (supports, contradicts, relates)
    - Provides a synthesis of the overall state

    Args:
        node_id: The node to summarize
        request: Summary parameters

    Returns:
        Summary with relationship context and synthesis

    Example:
        POST /api/v1/tools/nodes/hyp-456/summarize-with-context
        {
            "depth": 1,
            "relationship_types": ["SUPPORTS", "CONTRADICTS"],
            "max_length": 300
        }
    """
    result = await tool_service.summarize_node_with_context(
        node_id=node_id,
        depth=request.depth,
        relationship_types=request.relationship_types,
        max_length=request.max_length,
    )

    if not result.success and result.error:
        status_code = 404 if "not found" in result.error.lower() else 500
        raise HTTPException(status_code=status_code, detail=result.error)

    return result


@router.post(
    "/nodes/{node_id}/recalculate-confidence",
    response_model=RecalculateConfidenceResponse
)
async def recalculate_node_confidence(
    node_id: str,
    request: RecalculateConfidenceRequest = Body(...),
) -> RecalculateConfidenceResponse:
    """Recalculate a node's confidence based on current graph context.

    This endpoint:
    - Analyzes the node's content quality
    - Considers supporting and contradicting evidence
    - Uses LLM to evaluate overall confidence
    - Updates the confidence value in the database

    Args:
        node_id: The node to recalculate confidence for
        request: Recalculation parameters

    Returns:
        Old and new confidence scores with reasoning

    Example:
        POST /api/v1/tools/nodes/obs-123/recalculate-confidence
        {
            "factor_in_relationships": true
        }
    """
    result = await tool_service.recalculate_node_confidence(
        node_id=node_id,
        factor_in_relationships=request.factor_in_relationships,
    )

    if not result.success and result.error:
        status_code = 404 if "not found" in result.error.lower() else 500
        raise HTTPException(status_code=status_code, detail=result.error)

    return result


# ============================================================================
# Relationship Analysis Endpoints
# ============================================================================

@router.post(
    "/relationships/{edge_id}/summarize",
    response_model=SummarizeRelationshipResponse
)
async def summarize_relationship(
    edge_id: str,
    request: SummarizeRelationshipRequest = Body(...),
) -> SummarizeRelationshipResponse:
    """Explain the connection between two nodes in plain language.

    This endpoint:
    - Retrieves both connected nodes
    - Uses LLM to explain the relationship
    - Identifies supporting evidence
    - Assesses relationship strength

    Args:
        edge_id: The relationship ID to summarize
        request: Summary parameters

    Returns:
        Plain language explanation with evidence and strength assessment

    Example:
        POST /api/v1/tools/relationships/rel-abc/summarize
        {
            "include_evidence": true
        }
    """
    result = await tool_service.summarize_relationship(
        edge_id=edge_id,
        include_evidence=request.include_evidence,
    )

    if not result.success and result.error:
        status_code = 404 if "not found" in result.error.lower() else 500
        raise HTTPException(status_code=status_code, detail=result.error)

    return result


# ============================================================================
# Utility Endpoints
# ============================================================================

@router.get("/health")
async def tools_health():
    """Check if tool service is properly configured.

    Returns:
        Health status including AI configuration status
    """
    config = tool_service.config

    return {
        "status": "healthy" if config.is_configured else "degraded",
        "ai_configured": config.is_configured,
        "llm_model": config.llm_model,
        "embedding_model": config.embedding_model,
        "message": "Tools ready" if config.is_configured else "OpenAI API key not configured"
    }


@router.get("/capabilities")
async def list_capabilities():
    """List all available tool capabilities.

    Returns:
        List of available operations with descriptions
    """
    return {
        "node_analysis": [
            {
                "operation": "find_related_nodes",
                "endpoint": "POST /tools/nodes/{node_id}/find-related",
                "description": "Find semantically similar nodes using vector embeddings"
            },
            {
                "operation": "summarize_node",
                "endpoint": "POST /tools/nodes/{node_id}/summarize",
                "description": "Generate LLM summary of a node's content"
            },
            {
                "operation": "summarize_node_with_context",
                "endpoint": "POST /tools/nodes/{node_id}/summarize-with-context",
                "description": "Summarize node including its relationships"
            },
            {
                "operation": "recalculate_confidence",
                "endpoint": "POST /tools/nodes/{node_id}/recalculate-confidence",
                "description": "Re-analyze node confidence based on graph context"
            },
        ],
        "relationship_analysis": [
            {
                "operation": "summarize_relationship",
                "endpoint": "POST /tools/relationships/{edge_id}/summarize",
                "description": "Explain connection between nodes in plain language"
            },
        ],
        "coming_soon": [
            "search_web_for_evidence",
            "recalculate_edge_confidence",
            "reclassify_relationship",
            "reclassify_node",
            "merge_nodes",
        ]
    }
