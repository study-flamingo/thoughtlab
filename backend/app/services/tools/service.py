"""Tool service facade.

Provides the main ToolService class that composes all operation modules
into a unified interface for LLM-powered graph operations.
"""

from typing import Optional, List, Literal

from app.services.tools.operations.node_analysis import NodeAnalysisOperations
from app.services.tools.operations.node_modification import NodeModificationOperations
from app.services.tools.operations.relationship_analysis import RelationshipAnalysisOperations

from app.models.tool_models import (
    FindRelatedNodesResponse,
    SummarizeNodeResponse,
    SummarizeNodeWithContextResponse,
    RecalculateConfidenceResponse,
    ReclassifyNodeResponse,
    SearchWebEvidenceResponse,
    MergeNodesResponse,
    SummarizeRelationshipResponse,
    RecalculateEdgeConfidenceResponse,
    ReclassifyRelationshipResponse,
)


class ToolService:
    """Unified service for LLM-powered graph operations.

    This is the main entry point for tool operations, composing:
    - NodeAnalysisOperations: find_related, summarize, confidence
    - NodeModificationOperations: reclassify, merge, web_evidence
    - RelationshipAnalysisOperations: summarize, reclassify, confidence

    All operations are designed to be stateless and independently testable.
    """

    def __init__(self):
        """Initialize tool service with operation modules."""
        self._node_analysis = NodeAnalysisOperations()
        self._node_modification = NodeModificationOperations()
        self._relationship_analysis = RelationshipAnalysisOperations()

    @property
    def config(self):
        """Access AI configuration (from any operation module)."""
        return self._node_analysis.config

    # ========================================================================
    # Node Analysis Operations
    # ========================================================================

    async def find_related_nodes(
        self,
        node_id: str,
        limit: int = 10,
        min_similarity: float = 0.5,
        node_types: Optional[List[str]] = None,
        auto_link: bool = False,
    ) -> FindRelatedNodesResponse:
        """Find semantically similar nodes using vector embeddings.

        Args:
            node_id: The node to find similar nodes for
            limit: Maximum number of results
            min_similarity: Minimum similarity score (0-1)
            node_types: Optional filter for node types
            auto_link: If True, automatically create relationships

        Returns:
            FindRelatedNodesResponse with similar nodes
        """
        return await self._node_analysis.find_related_nodes(
            node_id=node_id,
            limit=limit,
            min_similarity=min_similarity,
            node_types=node_types,
            auto_link=auto_link,
        )

    async def summarize_node(
        self,
        node_id: str,
        max_length: int = 200,
        style: Literal["concise", "detailed", "bullet_points"] = "concise",
    ) -> SummarizeNodeResponse:
        """Generate LLM summary of a node's content.

        Args:
            node_id: The node to summarize
            max_length: Maximum length in characters
            style: Summary style

        Returns:
            SummarizeNodeResponse with summary and key points
        """
        return await self._node_analysis.summarize_node(
            node_id=node_id,
            max_length=max_length,
            style=style,
        )

    async def summarize_node_with_context(
        self,
        node_id: str,
        depth: int = 1,
        relationship_types: Optional[List[str]] = None,
        max_length: int = 300,
    ) -> SummarizeNodeWithContextResponse:
        """Summarize node including its relationships and connected nodes.

        Args:
            node_id: The node to summarize
            depth: How many hops to include (currently only supports 1)
            relationship_types: Optional filter for relationship types
            max_length: Maximum length in characters

        Returns:
            SummarizeNodeWithContextResponse with context-aware summary
        """
        return await self._node_analysis.summarize_node_with_context(
            node_id=node_id,
            depth=depth,
            relationship_types=relationship_types,
            max_length=max_length,
        )

    async def recalculate_node_confidence(
        self,
        node_id: str,
        factor_in_relationships: bool = True,
    ) -> RecalculateConfidenceResponse:
        """Re-analyze node confidence based on current graph context.

        Args:
            node_id: The node to recalculate confidence for
            factor_in_relationships: Whether to consider connected nodes

        Returns:
            RecalculateConfidenceResponse with new confidence and reasoning
        """
        return await self._node_analysis.recalculate_node_confidence(
            node_id=node_id,
            factor_in_relationships=factor_in_relationships,
        )

    # ========================================================================
    # Node Modification Operations
    # ========================================================================

    async def reclassify_node(
        self,
        node_id: str,
        new_type: str,
        preserve_relationships: bool = True,
    ) -> ReclassifyNodeResponse:
        """Reclassify a node to a new type.

        Args:
            node_id: The node to reclassify
            new_type: New node type (Observation, Hypothesis, etc.)
            preserve_relationships: Whether to preserve existing relationships

        Returns:
            ReclassifyNodeResponse with result
        """
        return await self._node_modification.reclassify_node(
            node_id=node_id,
            new_type=new_type,
            preserve_relationships=preserve_relationships,
        )

    async def search_web_evidence(
        self,
        node_id: str,
        evidence_type: str = "supporting",
        max_results: int = 5,
        auto_create_sources: bool = False,
    ) -> SearchWebEvidenceResponse:
        """Search the web for evidence related to a node.

        Args:
            node_id: The node to find evidence for
            evidence_type: Type of evidence to search for
            max_results: Maximum number of results
            auto_create_sources: Whether to auto-create Source nodes

        Returns:
            SearchWebEvidenceResponse with results
        """
        return await self._node_modification.search_web_evidence(
            node_id=node_id,
            evidence_type=evidence_type,
            max_results=max_results,
            auto_create_sources=auto_create_sources,
        )

    async def merge_nodes(
        self,
        primary_node_id: str,
        secondary_node_id: str,
        merge_strategy: Literal["keep_primary", "keep_secondary", "combine", "smart"] = "combine",
    ) -> MergeNodesResponse:
        """Merge two nodes into one.

        Args:
            primary_node_id: The node to keep
            secondary_node_id: The node to merge into primary and delete
            merge_strategy: How to handle conflicting properties

        Returns:
            MergeNodesResponse with result
        """
        return await self._node_modification.merge_nodes(
            primary_node_id=primary_node_id,
            secondary_node_id=secondary_node_id,
            merge_strategy=merge_strategy,
        )

    # ========================================================================
    # Relationship Analysis Operations
    # ========================================================================

    async def summarize_relationship(
        self,
        edge_id: str,
        include_evidence: bool = True,
    ) -> SummarizeRelationshipResponse:
        """Explain the connection between two nodes in plain language.

        Args:
            edge_id: The relationship ID to summarize
            include_evidence: Whether to include supporting evidence

        Returns:
            SummarizeRelationshipResponse with explanation
        """
        return await self._relationship_analysis.summarize_relationship(
            edge_id=edge_id,
            include_evidence=include_evidence,
        )

    async def recalculate_edge_confidence(
        self,
        edge_id: str,
        consider_graph_structure: bool = True,
    ) -> RecalculateEdgeConfidenceResponse:
        """Recalculate edge confidence based on connected nodes and context.

        Args:
            edge_id: The relationship ID to recalculate
            consider_graph_structure: Whether to factor in graph structure

        Returns:
            RecalculateEdgeConfidenceResponse with new confidence and reasoning
        """
        return await self._relationship_analysis.recalculate_edge_confidence(
            edge_id=edge_id,
            consider_graph_structure=consider_graph_structure,
        )

    async def reclassify_relationship(
        self,
        edge_id: str,
        new_type: Optional[str] = None,
        preserve_notes: bool = True,
    ) -> ReclassifyRelationshipResponse:
        """Reclassify a relationship to a new type.

        Args:
            edge_id: The relationship ID to reclassify
            new_type: New relationship type (if None, AI suggests best type)
            preserve_notes: Whether to preserve existing notes

        Returns:
            ReclassifyRelationshipResponse with result
        """
        return await self._relationship_analysis.reclassify_relationship(
            edge_id=edge_id,
            new_type=new_type,
            preserve_notes=preserve_notes,
        )


# Global service instance
_tool_service: Optional[ToolService] = None


def get_tool_service() -> ToolService:
    """Get the global tool service instance."""
    global _tool_service
    if _tool_service is None:
        _tool_service = ToolService()
    return _tool_service
