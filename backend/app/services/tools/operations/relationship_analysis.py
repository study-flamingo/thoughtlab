"""Relationship analysis operations.

Operations for analyzing and modifying relationships in the knowledge graph:
- Summarize relationship
- Recalculate relationship confidence
- Reclassify relationship type
"""

from typing import Optional, Literal
import logging

from app.services.tools.base import ToolServiceBase
from app.db.neo4j import neo4j_conn
from app.models.tool_models import (
    NodeInfo,
    ConfidenceFactor,
    SummarizeRelationshipResponse,
    RecalculateEdgeConfidenceResponse,
    ReclassifyRelationshipResponse,
)

logger = logging.getLogger(__name__)


class RelationshipAnalysisOperations(ToolServiceBase):
    """Operations for analyzing relationships in the knowledge graph."""

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
            relationship = await self.get_relationship_by_id(edge_id)
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
            from_node = await self.get_node_by_id(from_node_id)
            to_node = await self.get_node_by_id(to_node_id)

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

            from_content = self.extract_node_content(from_node)
            to_content = self.extract_node_content(to_node)
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
                strength: Literal["strong", "moderate", "weak"] = "strong"
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
        try:
            # Get the relationship and connected nodes
            relationship = await self.get_relationship_by_id(edge_id)
            if not relationship:
                return RecalculateEdgeConfidenceResponse(
                    success=False,
                    edge_id=edge_id,
                    old_confidence=0.0,
                    new_confidence=0.0,
                    reasoning="",
                    factors=[],
                    error="Relationship not found"
                )

            old_confidence = relationship.get("confidence", 1.0) or 1.0
            rel_type = relationship["type"]
            rel_notes = relationship.get("notes", "")

            # Get both connected nodes
            from_node = await self.get_node_by_id(relationship["from_id"])
            to_node = await self.get_node_by_id(relationship["to_id"])

            if not from_node or not to_node:
                return RecalculateEdgeConfidenceResponse(
                    success=False,
                    edge_id=edge_id,
                    old_confidence=old_confidence,
                    new_confidence=old_confidence,
                    reasoning="",
                    factors=[],
                    error="Connected nodes not found"
                )

            from_content = self.extract_node_content(from_node)
            to_content = self.extract_node_content(to_node)
            from_type = from_node.get("type", "Unknown")
            to_type = to_node.get("type", "Unknown")

            # Build context for graph structure if requested
            structure_context = ""
            if consider_graph_structure:
                from_rels = await self.get_node_relationships(relationship["from_id"], None)
                to_rels = await self.get_node_relationships(relationship["to_id"], None)
                structure_context = f"""
Graph structure context:
- Source node has {len(from_rels)} total relationships
- Target node has {len(to_rels)} total relationships
"""

            # Ask LLM to evaluate relationship strength
            prompt = f"""Evaluate the strength of this relationship on a scale of 0.0 to 1.0.

Relationship Type: {rel_type}

From Node ({from_type}):
{from_content[:500]}

To Node ({to_type}):
{to_content[:500]}

Relationship Notes: {rel_notes or "None"}
{structure_context}

Consider:
1. How well the content of both nodes supports this relationship type
2. The logical connection between the nodes
3. {"The graph structure context" if consider_graph_structure else ""}

Provide:
1. A confidence score (0.0 to 1.0)
2. Reasoning for the score
3. Key factors that influenced the score

Response format:
CONFIDENCE: 0.XX
REASONING: ...
FACTORS:
- Factor 1: impact description
- Factor 2: impact description
"""

            response = await self.llm.ainvoke(prompt)
            content_response = response.content.strip()

            # Parse response
            new_confidence = old_confidence
            reasoning = ""
            factors = []

            for line in content_response.split("\n"):
                line = line.strip()
                if line.startswith("CONFIDENCE:"):
                    try:
                        new_confidence = float(line.split(":")[1].strip())
                        new_confidence = max(0.0, min(1.0, new_confidence))
                    except ValueError:
                        pass
                elif line.startswith("REASONING:"):
                    reasoning = line.split(":", 1)[1].strip()
                elif line.startswith("-") and ":" in line:
                    factor_text = line.lstrip("-• ").strip()
                    if ":" in factor_text:
                        factor_name, impact = factor_text.split(":", 1)
                        factors.append(ConfidenceFactor(
                            factor=factor_name.strip(),
                            impact=impact.strip()
                        ))

            # Update relationship confidence in database
            await self.update_relationship_confidence(edge_id, new_confidence)

            return RecalculateEdgeConfidenceResponse(
                success=True,
                edge_id=edge_id,
                old_confidence=old_confidence,
                new_confidence=new_confidence,
                reasoning=reasoning or "Confidence recalculated based on relationship analysis",
                factors=factors
            )

        except Exception as e:
            logger.exception(f"Error recalculating edge confidence for {edge_id}")
            return RecalculateEdgeConfidenceResponse(
                success=False,
                edge_id=edge_id,
                old_confidence=0.0,
                new_confidence=0.0,
                reasoning="",
                factors=[],
                error=str(e)
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
        try:
            # Get the relationship
            relationship = await self.get_relationship_by_id(edge_id)
            if not relationship:
                return ReclassifyRelationshipResponse(
                    success=False,
                    edge_id=edge_id,
                    old_type="",
                    new_type="",
                    suggested_by_ai=False,
                    reasoning="",
                    notes_preserved=False,
                    error="Relationship not found"
                )

            old_type = relationship["type"]
            rel_notes = relationship.get("notes", "")
            rel_confidence = relationship.get("confidence", 1.0) or 1.0

            # Get connected nodes for context
            from_node = await self.get_node_by_id(relationship["from_id"])
            to_node = await self.get_node_by_id(relationship["to_id"])

            if not from_node or not to_node:
                return ReclassifyRelationshipResponse(
                    success=False,
                    edge_id=edge_id,
                    old_type=old_type,
                    new_type="",
                    suggested_by_ai=False,
                    reasoning="",
                    notes_preserved=False,
                    error="Connected nodes not found"
                )

            from_content = self.extract_node_content(from_node)
            to_content = self.extract_node_content(to_node)
            from_type = from_node.get("type", "Unknown")
            to_type = to_node.get("type", "Unknown")

            suggested_by_ai = new_type is None
            reasoning = ""

            if new_type is None:
                # Ask LLM to suggest best relationship type
                prompt = f"""Analyze these two connected nodes and suggest the best relationship type.

From Node ({from_type}):
{from_content[:500]}

To Node ({to_type}):
{to_content[:500]}

Current relationship type: {old_type}
Relationship notes: {rel_notes or "None"}

Available relationship types:
- SUPPORTS: The source provides evidence supporting the target
- CONTRADICTS: The source contradicts or challenges the target
- RELATES_TO: General relationship or connection
- DERIVED_FROM: The source is derived from or based on the target
- CITES: The source cites or references the target

What is the most appropriate relationship type for this connection?

Response format:
TYPE: RELATIONSHIP_TYPE
REASONING: Why this type is most appropriate
"""

                response = await self.llm.ainvoke(prompt)
                content_response = response.content.strip()

                for line in content_response.split("\n"):
                    line = line.strip()
                    if line.startswith("TYPE:"):
                        suggested = line.split(":")[1].strip().upper()
                        valid_types = ["SUPPORTS", "CONTRADICTS", "RELATES_TO", "DERIVED_FROM", "CITES"]
                        if suggested in valid_types:
                            new_type = suggested
                    elif line.startswith("REASONING:"):
                        reasoning = line.split(":", 1)[1].strip()

                if new_type is None:
                    new_type = old_type
                    reasoning = "Could not determine a better relationship type"
            else:
                reasoning = f"Manually reclassified from {old_type} to {new_type}"

            # Update relationship type in database
            success = await self._update_relationship_type(
                edge_id,
                new_type,
                preserve_notes,
                rel_notes if preserve_notes else "",
                rel_confidence
            )

            return ReclassifyRelationshipResponse(
                success=success,
                edge_id=edge_id,
                old_type=old_type,
                new_type=new_type,
                suggested_by_ai=suggested_by_ai,
                reasoning=reasoning,
                notes_preserved=preserve_notes
            )

        except Exception as e:
            logger.exception(f"Error reclassifying relationship {edge_id}")
            return ReclassifyRelationshipResponse(
                success=False,
                edge_id=edge_id,
                old_type="",
                new_type="",
                suggested_by_ai=False,
                reasoning="",
                notes_preserved=False,
                error=str(e)
            )

    # ========================================================================
    # Private Helper Methods
    # ========================================================================

    async def _update_relationship_type(
        self,
        edge_id: str,
        new_type: str,
        preserve_notes: bool,
        notes: str,
        confidence: float,
    ) -> bool:
        """Update relationship type by recreating it with new type.

        Note: Neo4j doesn't allow changing relationship types directly,
        so we need to delete and recreate.
        """
        # Get current relationship details
        get_query = """
        MATCH (from)-[r {id: $edge_id}]->(to)
        RETURN from.id as from_id, to.id as to_id, r.id as id,
               r.confidence as confidence, r.notes as notes,
               r.created_at as created_at, r.created_by as created_by
        """

        async with neo4j_conn.get_session() as session:
            result = await session.run(get_query, edge_id=edge_id)
            record = await result.single()
            if not record:
                return False

            from_id = record["from_id"]
            to_id = record["to_id"]
            rel_id = record["id"]
            created_at = record["created_at"]
            created_by = record["created_by"]

            # Delete old relationship and create new one with new type
            delete_create_query = f"""
            MATCH (from {{id: $from_id}})-[r {{id: $edge_id}}]->(to {{id: $to_id}})
            DELETE r
            WITH from, to
            CREATE (from)-[new_r:{new_type} {{
                id: $rel_id,
                confidence: $confidence,
                notes: $notes,
                created_at: $created_at,
                created_by: $created_by,
                updated_at: datetime()
            }}]->(to)
            RETURN new_r.id as id
            """

            result = await session.run(
                delete_create_query,
                from_id=from_id,
                to_id=to_id,
                edge_id=edge_id,
                rel_id=rel_id,
                confidence=confidence,
                notes=notes if preserve_notes else "",
                created_at=created_at,
                created_by=created_by
            )
            record = await result.single()
            return record is not None
