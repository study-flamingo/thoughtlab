"""API routes for LLM-powered graph operation tools.

These endpoints provide on-demand operations that can be invoked by:
- LangGraph agents
- Frontend components
- MCP server
- CLI tools

All endpoints are stateless and independently testable.
"""

from fastapi import APIRouter, HTTPException, Body

from app.services.tools import get_tool_service
from app.models.tool_models import (
    # Request models
    FindRelatedNodesRequest,
    SummarizeNodeRequest,
    SummarizeNodeWithContextRequest,
    RecalculateConfidenceRequest,
    SummarizeRelationshipRequest,
    ReclassifyNodeRequest,
    SearchWebEvidenceRequest,
    RecalculateEdgeConfidenceRequest,
    ReclassifyRelationshipRequest,
    MergeNodesRequest,
    # Response models
    FindRelatedNodesResponse,
    SummarizeNodeResponse,
    SummarizeNodeWithContextResponse,
    RecalculateConfidenceResponse,
    SummarizeRelationshipResponse,
    ReclassifyNodeResponse,
    SearchWebEvidenceResponse,
    RecalculateEdgeConfidenceResponse,
    ReclassifyRelationshipResponse,
    MergeNodesResponse,
)

router = APIRouter(prefix="/tools", tags=["tools"])
tool_service = get_tool_service()


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
# Node Modification Endpoints
# ============================================================================

@router.post("/nodes/{node_id}/reclassify", response_model=ReclassifyNodeResponse)
async def reclassify_node(
    node_id: str,
    request: ReclassifyNodeRequest = Body(...),
) -> ReclassifyNodeResponse:
    """Reclassify a node to a different type.

    This endpoint:
    - Changes the node's type/label
    - Preserves all properties
    - Optionally preserves relationships

    Args:
        node_id: The node to reclassify
        request: Reclassification parameters

    Returns:
        Result with old type, new type, and preserved items

    Example:
        POST /api/v1/tools/nodes/obs-123/reclassify
        {
            "new_type": "Hypothesis",
            "preserve_relationships": true
        }
    """
    result = await tool_service.reclassify_node(
        node_id=node_id,
        new_type=request.new_type,
        preserve_relationships=request.preserve_relationships,
    )

    if not result.success and result.error:
        status_code = 404 if "not found" in result.error.lower() else 400 if "invalid" in result.error.lower() else 500
        raise HTTPException(status_code=status_code, detail=result.error)

    return result


@router.post(
    "/nodes/{node_id}/search-web-evidence",
    response_model=SearchWebEvidenceResponse
)
async def search_web_evidence(
    node_id: str,
    request: SearchWebEvidenceRequest = Body(...),
) -> SearchWebEvidenceResponse:
    """Search the web for evidence related to a node.

    Note: This endpoint requires TAVILY_API_KEY to be configured.
    Without it, returns a placeholder message.

    This endpoint:
    - Extracts content from the node
    - Searches the web for related evidence
    - Optionally creates Source nodes from results

    Args:
        node_id: The node to find evidence for
        request: Search parameters

    Returns:
        Search results with titles, URLs, and snippets

    Example:
        POST /api/v1/tools/nodes/hyp-456/search-web-evidence
        {
            "evidence_type": "supporting",
            "max_results": 5,
            "auto_create_sources": false
        }
    """
    result = await tool_service.search_web_evidence(
        node_id=node_id,
        evidence_type=request.evidence_type,
        max_results=request.max_results,
        auto_create_sources=request.auto_create_sources,
    )

    # Note: We don't raise HTTPException for "not configured" - it's a valid response
    if not result.success and result.error and "not found" in result.error.lower():
        raise HTTPException(status_code=404, detail=result.error)

    return result


@router.post("/nodes/merge", response_model=MergeNodesResponse)
async def merge_nodes(
    request: MergeNodesRequest = Body(...),
) -> MergeNodesResponse:
    """Merge two nodes into one.

    This endpoint:
    - Combines properties from both nodes based on strategy
    - Transfers all relationships from secondary to primary
    - Deletes the secondary node

    Args:
        request: Merge parameters including both node IDs and strategy

    Returns:
        Result with merged properties and transferred relationships

    Example:
        POST /api/v1/tools/nodes/merge
        {
            "primary_node_id": "obs-123",
            "secondary_node_id": "obs-456",
            "merge_strategy": "combine"
        }
    """
    result = await tool_service.merge_nodes(
        primary_node_id=request.primary_node_id,
        secondary_node_id=request.secondary_node_id,
        merge_strategy=request.merge_strategy,
    )

    if not result.success and result.error:
        status_code = 404 if "not found" in result.error.lower() else 400 if "mismatch" in result.error.lower() else 500
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


@router.post(
    "/relationships/{edge_id}/recalculate-confidence",
    response_model=RecalculateEdgeConfidenceResponse
)
async def recalculate_edge_confidence(
    edge_id: str,
    request: RecalculateEdgeConfidenceRequest = Body(...),
) -> RecalculateEdgeConfidenceResponse:
    """Recalculate a relationship's confidence based on connected nodes.

    This endpoint:
    - Analyzes the content of both connected nodes
    - Evaluates how well they support the relationship type
    - Uses LLM to determine confidence level
    - Updates the confidence value in the database

    Args:
        edge_id: The relationship ID to recalculate confidence for
        request: Recalculation parameters

    Returns:
        Old and new confidence scores with reasoning

    Example:
        POST /api/v1/tools/relationships/rel-abc/recalculate-confidence
        {
            "consider_graph_structure": true
        }
    """
    result = await tool_service.recalculate_edge_confidence(
        edge_id=edge_id,
        consider_graph_structure=request.consider_graph_structure,
    )

    if not result.success and result.error:
        status_code = 404 if "not found" in result.error.lower() else 500
        raise HTTPException(status_code=status_code, detail=result.error)

    return result


@router.post(
    "/relationships/{edge_id}/reclassify",
    response_model=ReclassifyRelationshipResponse
)
async def reclassify_relationship(
    edge_id: str,
    request: ReclassifyRelationshipRequest = Body(...),
) -> ReclassifyRelationshipResponse:
    """Reclassify a relationship to a different type.

    This endpoint:
    - Changes the relationship type
    - Optionally lets AI suggest the best type
    - Preserves notes and confidence if requested

    Args:
        edge_id: The relationship to reclassify
        request: Reclassification parameters

    Returns:
        Result with old type, new type, and reasoning

    Example:
        POST /api/v1/tools/relationships/rel-abc/reclassify
        {
            "new_type": "SUPPORTS",
            "preserve_notes": true
        }

        Or let AI suggest:
        POST /api/v1/tools/relationships/rel-abc/reclassify
        {
            "new_type": null,
            "preserve_notes": true
        }
    """
    result = await tool_service.reclassify_relationship(
        edge_id=edge_id,
        new_type=request.new_type,
        preserve_notes=request.preserve_notes,
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
            {
                "operation": "reclassify_node",
                "endpoint": "POST /tools/nodes/{node_id}/reclassify",
                "description": "Change a node's type (Observation, Hypothesis, etc.)"
            },
            {
                "operation": "search_web_evidence",
                "endpoint": "POST /tools/nodes/{node_id}/search-web-evidence",
                "description": "Search the web for evidence (requires TAVILY_API_KEY)"
            },
        ],
        "node_modification": [
            {
                "operation": "merge_nodes",
                "endpoint": "POST /tools/nodes/merge",
                "description": "Merge two nodes into one, transferring relationships"
            },
        ],
        "relationship_analysis": [
            {
                "operation": "summarize_relationship",
                "endpoint": "POST /tools/relationships/{edge_id}/summarize",
                "description": "Explain connection between nodes in plain language"
            },
            {
                "operation": "recalculate_edge_confidence",
                "endpoint": "POST /tools/relationships/{edge_id}/recalculate-confidence",
                "description": "Re-analyze relationship confidence based on connected nodes"
            },
            {
                "operation": "reclassify_relationship",
                "endpoint": "POST /tools/relationships/{edge_id}/reclassify",
                "description": "Change relationship type (SUPPORTS, CONTRADICTS, etc.)"
            },
        ],
    }
