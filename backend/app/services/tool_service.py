"""Tool service for LLM-powered graph operations.

This service provides on-demand operations that can be invoked by:
- LangGraph agents (via API)
- Frontend (via API)
- MCP server (via API)
- CLI tools (via API)

All operations are designed to be stateless and independently testable.
"""

from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field
import logging

from app.ai.config import get_ai_config
from app.ai.similarity import get_similarity_search
from app.ai.classifier import get_relationship_classifier
from app.services.graph_service import graph_service
from app.db.neo4j import neo4j_conn
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)


# ============================================================================
# Response Models
# ============================================================================

class RelatedNodeResult(BaseModel):
    """A single related node result."""
    id: str
    type: str
    content: str
    similarity_score: float
    suggested_relationship: str
    reasoning: str


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


class NodeContextSummary(BaseModel):
    """Context summary for a node."""
    supports: List[str] = Field(default_factory=list)
    contradicts: List[str] = Field(default_factory=list)
    related: List[str] = Field(default_factory=list)


class SummarizeNodeWithContextResponse(BaseModel):
    """Response for summarize_node_with_context operation."""
    success: bool
    node_id: str
    summary: str
    context: NodeContextSummary
    synthesis: str
    relationship_count: int
    error: Optional[str] = None


class ConfidenceFactor(BaseModel):
    """A factor affecting confidence calculation."""
    factor: str
    impact: str


class RecalculateConfidenceResponse(BaseModel):
    """Response for recalculate_node_confidence operation."""
    success: bool
    node_id: str
    old_confidence: float
    new_confidence: float
    reasoning: str
    factors: List[ConfidenceFactor]
    error: Optional[str] = None


class NodeInfo(BaseModel):
    """Node information for relationship summary."""
    id: str
    type: str
    content: str


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


# ============================================================================
# Tool Service
# ============================================================================

class ToolService:
    """Service for LLM-powered graph operations."""

    def __init__(self):
        """Initialize tool service."""
        self.config = get_ai_config()
        self._similarity_search = None
        self._classifier = None
        self._llm = None

    @property
    def similarity_search(self):
        """Lazy initialization of similarity search."""
        if self._similarity_search is None:
            self._similarity_search = get_similarity_search()
        return self._similarity_search

    @property
    def classifier(self):
        """Lazy initialization of relationship classifier."""
        if self._classifier is None:
            self._classifier = get_relationship_classifier()
        return self._classifier

    @property
    def llm(self):
        """Lazy initialization of LLM for summarization."""
        if self._llm is None:
            self._llm = ChatOpenAI(
                model=self.config.llm_model,
                openai_api_key=self.config.openai_api_key,
                temperature=0.3,  # Slightly creative for summaries
            )
        return self._llm

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
        try:
            # Get the source node
            node = await self._get_node_by_id(node_id)
            if not node:
                return FindRelatedNodesResponse(
                    success=False,
                    node_id=node_id,
                    related_nodes=[],
                    message="Node not found",
                    error="Node not found"
                )

            # Extract content for similarity search
            content = self._extract_node_content(node)
            if not content:
                return FindRelatedNodesResponse(
                    success=False,
                    node_id=node_id,
                    related_nodes=[],
                    message="Node has no content to analyze",
                    error="No content available"
                )

            # Find similar nodes
            candidates = await self.similarity_search.find_similar(
                query_text=content,
                exclude_node_id=node_id,
                node_types=node_types,
                limit=limit,
                min_score=min_similarity,
            )

            # Classify relationships for each candidate
            related_nodes = []
            links_created = 0
            node_type = node.get("type", "Unknown")

            for candidate in candidates:
                # Classify the relationship
                classification = await self.classifier.classify(
                    source_content=content,
                    source_type=node_type,
                    target_content=candidate["content"] or "",
                    target_type=candidate["node_type"],
                )

                if classification and classification.is_valid:
                    related_node = RelatedNodeResult(
                        id=candidate["node_id"],
                        type=candidate["node_type"],
                        content=(candidate["content"] or "")[:200],  # Preview
                        similarity_score=candidate["score"],
                        suggested_relationship=classification.relationship_type,
                        reasoning=classification.reasoning,
                    )
                    related_nodes.append(related_node)

                    # Auto-link if requested and confidence is high
                    if auto_link and classification.confidence >= self.config.auto_create_threshold:
                        rel_id = await graph_service.create_relationship(
                            from_id=node_id,
                            to_id=candidate["node_id"],
                            rel_type=classification.relationship_type,
                            properties={
                                "confidence": classification.confidence,
                                "notes": classification.reasoning,
                            },
                            created_by="tool-find-related",
                        )
                        if rel_id:
                            links_created += 1

            return FindRelatedNodesResponse(
                success=True,
                node_id=node_id,
                related_nodes=related_nodes,
                links_created=links_created,
                message=f"Found {len(related_nodes)} related nodes" +
                        (f", created {links_created} links" if auto_link else "")
            )

        except Exception as e:
            logger.exception(f"Error finding related nodes for {node_id}")
            return FindRelatedNodesResponse(
                success=False,
                node_id=node_id,
                related_nodes=[],
                message="Error finding related nodes",
                error=str(e)
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
        try:
            # Get the node
            node = await self._get_node_by_id(node_id)
            if not node:
                return SummarizeNodeResponse(
                    success=False,
                    node_id=node_id,
                    summary="",
                    key_points=[],
                    word_count=0,
                    error="Node not found"
                )

            # Extract content
            content = self._extract_node_content(node)
            if not content:
                return SummarizeNodeResponse(
                    success=False,
                    node_id=node_id,
                    summary="",
                    key_points=[],
                    word_count=0,
                    error="Node has no content to summarize"
                )

            # Build prompt based on style
            node_type = node.get("type", "Unknown")

            if style == "concise":
                prompt = f"""Summarize this {node_type} in {max_length} characters or less. Be concise and focus on the main point.

{node_type} Content:
{content[:1000]}

Provide a brief summary:"""
            elif style == "detailed":
                prompt = f"""Provide a detailed summary of this {node_type} in approximately {max_length} characters.

{node_type} Content:
{content[:2000]}

Summary:"""
            else:  # bullet_points
                prompt = f"""Summarize this {node_type} as 3-5 bullet points. Each point should be concise.

{node_type} Content:
{content[:1000]}

Key Points:
-"""

            # Generate summary
            response = await self.llm.ainvoke(prompt)
            summary = response.content.strip()

            # Extract key points
            key_points = []
            if style == "bullet_points":
                # Parse bullet points
                lines = summary.split("\n")
                for line in lines:
                    line = line.strip()
                    if line.startswith("-") or line.startswith("•"):
                        key_points.append(line.lstrip("-• ").strip())
            else:
                # Generate key points from summary
                key_points_prompt = f"""From this summary, extract 2-3 key points as a bullet list:

Summary: {summary}

Key points:
-"""
                kp_response = await self.llm.ainvoke(key_points_prompt)
                lines = kp_response.content.strip().split("\n")
                for line in lines:
                    line = line.strip()
                    if line.startswith("-") or line.startswith("•"):
                        key_points.append(line.lstrip("-• ").strip())

            word_count = len(summary.split())

            return SummarizeNodeResponse(
                success=True,
                node_id=node_id,
                summary=summary,
                key_points=key_points,
                word_count=word_count
            )

        except Exception as e:
            logger.exception(f"Error summarizing node {node_id}")
            return SummarizeNodeResponse(
                success=False,
                node_id=node_id,
                summary="",
                key_points=[],
                word_count=0,
                error=str(e)
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
        try:
            # Get the node
            node = await self._get_node_by_id(node_id)
            if not node:
                return SummarizeNodeWithContextResponse(
                    success=False,
                    node_id=node_id,
                    summary="",
                    context=NodeContextSummary(),
                    synthesis="",
                    relationship_count=0,
                    error="Node not found"
                )

            # Get relationships and connected nodes
            relationships = await self._get_node_relationships(
                node_id,
                relationship_types
            )

            # Organize by relationship type
            context = NodeContextSummary()
            for rel in relationships:
                rel_type = rel["type"]
                connected_content = rel["connected_content"][:100]
                summary_line = f"{rel['connected_type']} {rel['connected_id']}: {connected_content}"

                if rel_type == "SUPPORTS":
                    context.supports.append(summary_line)
                elif rel_type == "CONTRADICTS":
                    context.contradicts.append(summary_line)
                else:
                    context.related.append(summary_line)

            # Get node content
            node_content = self._extract_node_content(node)
            node_type = node.get("type", "Unknown")

            # Build comprehensive prompt
            context_str = ""
            if context.supports:
                context_str += "\n\nSupporting nodes:\n" + "\n".join(f"- {s}" for s in context.supports)
            if context.contradicts:
                context_str += "\n\nContradicting nodes:\n" + "\n".join(f"- {c}" for c in context.contradicts)
            if context.related:
                context_str += "\n\nRelated nodes:\n" + "\n".join(f"- {r}" for r in context.related[:3])  # Limit

            prompt = f"""Summarize this {node_type} considering its relationships in the knowledge graph.

{node_type} Content:
{node_content[:1000]}
{context_str}

Provide:
1. A concise summary of the {node_type} itself
2. How it fits within the broader context of connected information

Summary (max {max_length} characters):"""

            # Generate context-aware summary
            response = await self.llm.ainvoke(prompt)
            summary = response.content.strip()

            # Generate synthesis
            synthesis_prompt = f"""Based on this {node_type} and its relationships, provide a brief synthesis (2-3 sentences) about the overall state of knowledge on this topic.

{node_type}: {node_content[:500]}

Supporting evidence: {len(context.supports)} items
Contradicting evidence: {len(context.contradicts)} items
Related concepts: {len(context.related)} items

Synthesis:"""

            synthesis_response = await self.llm.ainvoke(synthesis_prompt)
            synthesis = synthesis_response.content.strip()

            return SummarizeNodeWithContextResponse(
                success=True,
                node_id=node_id,
                summary=summary,
                context=context,
                synthesis=synthesis,
                relationship_count=len(relationships)
            )

        except Exception as e:
            logger.exception(f"Error summarizing node with context {node_id}")
            return SummarizeNodeWithContextResponse(
                success=False,
                node_id=node_id,
                summary="",
                context=NodeContextSummary(),
                synthesis="",
                relationship_count=0,
                error=str(e)
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
        try:
            # Get the node
            node = await self._get_node_by_id(node_id)
            if not node:
                return RecalculateConfidenceResponse(
                    success=False,
                    node_id=node_id,
                    old_confidence=0.0,
                    new_confidence=0.0,
                    reasoning="",
                    factors=[],
                    error="Node not found"
                )

            old_confidence = node.get("confidence", 1.0)
            node_type = node.get("type", "Unknown")
            content = self._extract_node_content(node)

            # Get relationships if requested
            context_str = ""
            if factor_in_relationships:
                relationships = await self._get_node_relationships(node_id, None)
                supports = [r for r in relationships if r["type"] == "SUPPORTS"]
                contradicts = [r for r in relationships if r["type"] == "CONTRADICTS"]

                context_str = f"""
Relationship context:
- {len(supports)} supporting relationships
- {len(contradicts)} contradicting relationships
"""

            # Ask LLM to evaluate confidence
            prompt = f"""Evaluate the confidence level for this {node_type} on a scale of 0.0 to 1.0.

{node_type} Content:
{content[:1000]}
{context_str}

Consider:
1. Quality and clarity of the content
2. Internal consistency
3. Supporting and contradicting evidence (if provided)
4. Overall credibility

Provide:
1. A confidence score (0.0 to 1.0)
2. Reasoning for the score
3. Key factors that influenced the score

Response format:
CONFIDENCE: 0.XX
REASONING: ...
FACTORS:
- Factor 1: +/-0.XX
- Factor 2: +/-0.XX
"""

            response = await self.llm.ainvoke(prompt)
            content_response = response.content.strip()

            # Parse response
            new_confidence = old_confidence  # Default to old if parsing fails
            reasoning = ""
            factors = []

            for line in content_response.split("\n"):
                line = line.strip()
                if line.startswith("CONFIDENCE:"):
                    try:
                        new_confidence = float(line.split(":")[1].strip())
                        new_confidence = max(0.0, min(1.0, new_confidence))  # Clamp
                    except ValueError:
                        pass
                elif line.startswith("REASONING:"):
                    reasoning = line.split(":", 1)[1].strip()
                elif line.startswith("-") and ":" in line:
                    # Factor line
                    factor_text = line.lstrip("-• ").strip()
                    if ":" in factor_text:
                        factor_name, impact = factor_text.split(":", 1)
                        factors.append(ConfidenceFactor(
                            factor=factor_name.strip(),
                            impact=impact.strip()
                        ))

            # Update node confidence in database
            await self._update_node_confidence(node_id, new_confidence)

            return RecalculateConfidenceResponse(
                success=True,
                node_id=node_id,
                old_confidence=old_confidence,
                new_confidence=new_confidence,
                reasoning=reasoning or "Confidence recalculated based on content analysis",
                factors=factors
            )

        except Exception as e:
            logger.exception(f"Error recalculating confidence for {node_id}")
            return RecalculateConfidenceResponse(
                success=False,
                node_id=node_id,
                old_confidence=0.0,
                new_confidence=0.0,
                reasoning="",
                factors=[],
                error=str(e)
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
        try:
            # Get the relationship and connected nodes
            relationship = await self._get_relationship_by_id(edge_id)
            if not relationship:
                return SummarizeRelationshipResponse(
                    success=False,
                    edge_id=edge_id,
                    from_node=NodeInfo(id="", type="", content=""),
                    to_node=NodeInfo(id="", type="", content=""),
                    relationship_type="",
                    summary="",
                    evidence=[],
                    strength_assessment="weak",
                    error="Relationship not found"
                )

            from_node_id = relationship["from_id"]
            to_node_id = relationship["to_id"]
            rel_type = relationship["type"]
            rel_confidence = relationship.get("confidence", 1.0)
            rel_notes = relationship.get("notes", "")

            # Get both nodes
            from_node = await self._get_node_by_id(from_node_id)
            to_node = await self._get_node_by_id(to_node_id)

            if not from_node or not to_node:
                return SummarizeRelationshipResponse(
                    success=False,
                    edge_id=edge_id,
                    from_node=NodeInfo(id="", type="", content=""),
                    to_node=NodeInfo(id="", type="", content=""),
                    relationship_type=rel_type,
                    summary="",
                    evidence=[],
                    strength_assessment="weak",
                    error="Connected nodes not found"
                )

            from_content = self._extract_node_content(from_node)
            to_content = self._extract_node_content(to_node)
            from_type = from_node.get("type", "Unknown")
            to_type = to_node.get("type", "Unknown")

            # Build prompt
            evidence_str = f"\nRelationship notes: {rel_notes}" if rel_notes else ""

            prompt = f"""Explain the relationship between these two knowledge graph nodes in plain language.

From Node ({from_type}):
{from_content[:500]}

To Node ({to_type}):
{to_content[:500]}

Relationship Type: {rel_type}
Confidence: {rel_confidence:.2f}
{evidence_str}

Provide:
1. A clear explanation of how these nodes are connected (2-3 sentences)
2. {"3-5 pieces of specific evidence supporting this relationship" if include_evidence else ""}

Response:"""

            response = await self.llm.ainvoke(prompt)
            summary = response.content.strip()

            # Extract evidence if included
            evidence = []
            if include_evidence:
                # Parse evidence from response
                lines = summary.split("\n")
                in_evidence = False
                for line in lines:
                    line = line.strip()
                    if "evidence" in line.lower():
                        in_evidence = True
                        continue
                    if in_evidence and (line.startswith("-") or line.startswith("•") or line.startswith("1.")):
                        evidence.append(line.lstrip("-•123456789. ").strip())

            # Assess strength
            if rel_confidence >= 0.8:
                strength = "strong"
            elif rel_confidence >= 0.6:
                strength = "moderate"
            else:
                strength = "weak"

            return SummarizeRelationshipResponse(
                success=True,
                edge_id=edge_id,
                from_node=NodeInfo(
                    id=from_node_id,
                    type=from_type,
                    content=from_content[:200]
                ),
                to_node=NodeInfo(
                    id=to_node_id,
                    type=to_type,
                    content=to_content[:200]
                ),
                relationship_type=rel_type,
                summary=summary,
                evidence=evidence,
                strength_assessment=strength
            )

        except Exception as e:
            logger.exception(f"Error summarizing relationship {edge_id}")
            return SummarizeRelationshipResponse(
                success=False,
                edge_id=edge_id,
                from_node=NodeInfo(id="", type="", content=""),
                to_node=NodeInfo(id="", type="", content=""),
                relationship_type="",
                summary="",
                evidence=[],
                strength_assessment="weak",
                error=str(e)
            )

    # ========================================================================
    # Helper Methods
    # ========================================================================

    async def _get_node_by_id(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get a node by ID from Neo4j."""
        query = """
        MATCH (n {id: $node_id})
        RETURN n, labels(n)[0] as type
        """

        async with neo4j_conn.get_session() as session:
            result = await session.run(query, node_id=node_id)
            record = await result.single()
            if record:
                node_data = dict(record["n"])
                node_data["type"] = record["type"]
                return node_data
            return None

    def _extract_node_content(self, node: Dict[str, Any]) -> str:
        """Extract text content from a node."""
        # Try common content fields in order
        for field in ["text", "description", "content", "title", "name"]:
            if field in node and node[field]:
                return str(node[field])
        return ""

    async def _get_node_relationships(
        self,
        node_id: str,
        relationship_types: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Get all relationships for a node."""
        type_filter = ""
        if relationship_types:
            type_filter = f"AND type(r) IN {relationship_types}"

        query = f"""
        MATCH (n {{id: $node_id}})-[r]-(connected)
        {type_filter}
        RETURN
            r.id as id,
            type(r) as type,
            r.confidence as confidence,
            connected.id as connected_id,
            labels(connected)[0] as connected_type,
            COALESCE(
                connected.text,
                connected.description,
                connected.content,
                connected.title,
                connected.name,
                ''
            ) as connected_content
        """

        relationships = []
        async with neo4j_conn.get_session() as session:
            result = await session.run(query, node_id=node_id)
            async for record in result:
                relationships.append({
                    "id": record["id"],
                    "type": record["type"],
                    "confidence": record["confidence"] or 1.0,
                    "connected_id": record["connected_id"],
                    "connected_type": record["connected_type"],
                    "connected_content": record["connected_content"],
                })

        return relationships

    async def _get_relationship_by_id(self, edge_id: str) -> Optional[Dict[str, Any]]:
        """Get a relationship by ID."""
        query = """
        MATCH (from)-[r {id: $edge_id}]->(to)
        RETURN
            r.id as id,
            type(r) as type,
            r.confidence as confidence,
            r.notes as notes,
            from.id as from_id,
            to.id as to_id
        """

        async with neo4j_conn.get_session() as session:
            result = await session.run(query, edge_id=edge_id)
            record = await result.single()
            if record:
                return {
                    "id": record["id"],
                    "type": record["type"],
                    "confidence": record["confidence"],
                    "notes": record["notes"],
                    "from_id": record["from_id"],
                    "to_id": record["to_id"],
                }
            return None

    async def _update_node_confidence(self, node_id: str, new_confidence: float) -> bool:
        """Update node confidence in database."""
        query = """
        MATCH (n {id: $node_id})
        SET n.confidence = $confidence
        RETURN n.id as id
        """

        async with neo4j_conn.get_session() as session:
            result = await session.run(
                query,
                node_id=node_id,
                confidence=new_confidence
            )
            record = await result.single()
            return record is not None


# Global service instance
_tool_service: Optional[ToolService] = None


def get_tool_service() -> ToolService:
    """Get the global tool service instance."""
    global _tool_service
    if _tool_service is None:
        _tool_service = ToolService()
    return _tool_service
