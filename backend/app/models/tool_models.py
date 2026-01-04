"""Pydantic models for tool requests and responses.

This module contains all request and response models for the LLM-powered
tool operations. These models are used by:
- API routes (validation)
- Tool service (return types)
- LangGraph agents (tool schemas)
- MCP server (tool definitions)
"""

from typing import Optional, List, Literal
from pydantic import BaseModel, Field


# ============================================================================
# Shared/Helper Models
# ============================================================================

class RelatedNodeResult(BaseModel):
    """A single related node result from similarity search."""
    id: str
    type: str
    content: str
    similarity_score: float
    suggested_relationship: str
    reasoning: str


class NodeInfo(BaseModel):
    """Basic node information for relationship summaries."""
    id: str
    type: str
    content: str


class NodeContextSummary(BaseModel):
    """Context summary organizing relationships by type."""
    supports: List[str] = Field(default_factory=list)
    contradicts: List[str] = Field(default_factory=list)
    related: List[str] = Field(default_factory=list)


class ConfidenceFactor(BaseModel):
    """A factor affecting confidence calculation."""
    factor: str
    impact: str


class SearchWebEvidenceResult(BaseModel):
    """A single web search result."""
    title: str
    url: str
    snippet: str
    relevance_score: float


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


class ReclassifyNodeRequest(BaseModel):
    """Request for reclassifying a node."""
    new_type: str = Field(
        ...,
        description="New node type (Observation, Hypothesis, Question, Source, Note)"
    )
    preserve_relationships: bool = True


class SearchWebEvidenceRequest(BaseModel):
    """Request for searching web evidence."""
    evidence_type: str = Field(
        default="supporting",
        description="Type of evidence to search for"
    )
    max_results: int = Field(default=5, ge=1, le=20)
    auto_create_sources: bool = False


class RecalculateEdgeConfidenceRequest(BaseModel):
    """Request for recalculating edge confidence."""
    consider_graph_structure: bool = True


class ReclassifyRelationshipRequest(BaseModel):
    """Request for reclassifying a relationship."""
    new_type: Optional[str] = Field(
        default=None,
        description="New relationship type (SUPPORTS, CONTRADICTS, RELATES_TO, DERIVED_FROM, CITES). If null, AI suggests."
    )
    preserve_notes: bool = True


class MergeNodesRequest(BaseModel):
    """Request for merging two nodes."""
    primary_node_id: str = Field(
        ...,
        description="The node to keep (receives merged content)"
    )
    secondary_node_id: str = Field(
        ...,
        description="The node to merge and delete"
    )
    merge_strategy: Literal["keep_primary", "keep_secondary", "combine", "smart"] = Field(
        default="combine",
        description="How to handle conflicting properties. 'smart' uses AI to intelligently merge text content."
    )


# ============================================================================
# Response Models
# ============================================================================

class FindRelatedNodesResponse(BaseModel):
    """Response for find_related_nodes operation."""
    success: bool
    node_id: str
    related_nodes: List[RelatedNodeResult]
    links_created: int = 0
    message: str
    error: Optional[str] = None


class SummarizeNodeResponse(BaseModel):
    """Response for summarize_node operation."""
    success: bool
    node_id: str
    summary: str
    key_points: List[str]
    word_count: int
    error: Optional[str] = None


class SummarizeNodeWithContextResponse(BaseModel):
    """Response for summarize_node_with_context operation."""
    success: bool
    node_id: str
    summary: str
    context: NodeContextSummary
    synthesis: str
    relationship_count: int
    error: Optional[str] = None


class RecalculateConfidenceResponse(BaseModel):
    """Response for recalculate_node_confidence operation."""
    success: bool
    node_id: str
    old_confidence: float
    new_confidence: float
    reasoning: str
    factors: List[ConfidenceFactor]
    error: Optional[str] = None


class SummarizeRelationshipResponse(BaseModel):
    """Response for summarize_relationship operation."""
    success: bool
    edge_id: str
    from_node: NodeInfo
    to_node: NodeInfo
    relationship_type: str
    summary: str
    evidence: List[str]
    strength_assessment: Literal["strong", "moderate", "weak"]
    error: Optional[str] = None


class ReclassifyNodeResponse(BaseModel):
    """Response for reclassify_node operation."""
    success: bool
    node_id: str
    old_type: str
    new_type: str
    properties_preserved: List[str]
    relationships_preserved: int
    message: str
    error: Optional[str] = None


class SearchWebEvidenceResponse(BaseModel):
    """Response for search_web_evidence operation."""
    success: bool
    node_id: str
    query_used: str
    results: List[SearchWebEvidenceResult]
    sources_created: int = 0
    message: str
    error: Optional[str] = None


class RecalculateEdgeConfidenceResponse(BaseModel):
    """Response for recalculate_edge_confidence operation."""
    success: bool
    edge_id: str
    old_confidence: float
    new_confidence: float
    reasoning: str
    factors: List[ConfidenceFactor]
    error: Optional[str] = None


class ReclassifyRelationshipResponse(BaseModel):
    """Response for reclassify_relationship operation."""
    success: bool
    edge_id: str
    old_type: str
    new_type: str
    suggested_by_ai: bool
    reasoning: str
    notes_preserved: bool
    error: Optional[str] = None


class MergeNodesResponse(BaseModel):
    """Response for merge_nodes operation."""
    success: bool
    primary_node_id: str
    secondary_node_id: str
    merged_properties: List[str]
    relationships_transferred: int
    message: str
    error: Optional[str] = None
