"""Node analysis operations.

Operations for analyzing nodes in the knowledge graph:
- Find related nodes via semantic similarity
- Summarize node content
- Summarize with relationship context
- Recalculate confidence based on graph context
"""

from typing import Optional, List, Literal
import logging

from app.services.tools.base import ToolServiceBase
from app.services.graph_service import graph_service
from app.models.tool_models import (
    RelatedNodeResult,
    NodeContextSummary,
    ConfidenceFactor,
    FindRelatedNodesResponse,
    SummarizeNodeResponse,
    SummarizeNodeWithContextResponse,
    RecalculateConfidenceResponse,
)

logger = logging.getLogger(__name__)


class NodeAnalysisOperations(ToolServiceBase):
    """Operations for analyzing nodes in the knowledge graph."""

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
            node = await self.get_node_by_id(node_id)
            if not node:
                return FindRelatedNodesResponse(
                    success=False,
                    node_id=node_id,
                    related_nodes=[],
                    message="Node not found",
                    error="Node not found"
                )

            # Extract content for similarity search
            content = self.extract_node_content(node)
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
            node = await self.get_node_by_id(node_id)
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
            content = self.extract_node_content(node)
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
            node = await self.get_node_by_id(node_id)
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
            relationships = await self.get_node_relationships(
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
            node_content = self.extract_node_content(node)
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
            node = await self.get_node_by_id(node_id)
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
            content = self.extract_node_content(node)

            # Get relationships if requested
            context_str = ""
            if factor_in_relationships:
                relationships = await self.get_node_relationships(node_id, None)
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
            await self.update_node_confidence(node_id, new_confidence)

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
