"""Node modification operations.

Operations for modifying nodes in the knowledge graph:
- Reclassify node type
- Search web for evidence
- Merge two nodes
"""

from typing import Optional, List, Dict, Any, Literal
import os
import logging

from app.services.tools.base import ToolServiceBase
from app.db.neo4j import neo4j_conn
from app.models.tool_models import (
    ReclassifyNodeResponse,
    SearchWebEvidenceResponse,
    MergeNodesResponse,
)
from app.models.nodes import NodeType

logger = logging.getLogger(__name__)


class NodeModificationOperations(ToolServiceBase):
    """Operations for modifying nodes in the knowledge graph."""

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
        try:
            # Get the node
            node = await self.get_node_by_id(node_id)
            if not node:
                return ReclassifyNodeResponse(
                    success=False,
                    node_id=node_id,
                    old_type="",
                    new_type=new_type,
                    properties_preserved=[],
                    relationships_preserved=0,
                    message="Node not found",
                    error="Node not found"
                )

            old_type = node.get("type", "Unknown")

            # Validate new type against NodeType enum
            valid_types = [t.value for t in NodeType]
            if new_type not in valid_types:
                return ReclassifyNodeResponse(
                    success=False,
                    node_id=node_id,
                    old_type=old_type,
                    new_type=new_type,
                    properties_preserved=[],
                    relationships_preserved=0,
                    message=f"Invalid node type: {new_type}",
                    error=f"Type must be one of: {', '.join(valid_types)}"
                )

            if old_type == new_type:
                return ReclassifyNodeResponse(
                    success=True,
                    node_id=node_id,
                    old_type=old_type,
                    new_type=new_type,
                    properties_preserved=list(node.keys()),
                    relationships_preserved=0,
                    message="Node already has this type"
                )

            # Get current relationships if preserving
            relationships_count = 0
            if preserve_relationships:
                relationships = await self.get_node_relationships(node_id, None)
                relationships_count = len(relationships)

            # Update node label in Neo4j
            success, properties_preserved = await self._change_node_type(
                node_id, old_type, new_type
            )

            if not success:
                return ReclassifyNodeResponse(
                    success=False,
                    node_id=node_id,
                    old_type=old_type,
                    new_type=new_type,
                    properties_preserved=[],
                    relationships_preserved=0,
                    message="Failed to update node type",
                    error="Database update failed"
                )

            return ReclassifyNodeResponse(
                success=True,
                node_id=node_id,
                old_type=old_type,
                new_type=new_type,
                properties_preserved=properties_preserved,
                relationships_preserved=relationships_count,
                message=f"Successfully reclassified from {old_type} to {new_type}"
            )

        except Exception as e:
            logger.exception(f"Error reclassifying node {node_id}")
            return ReclassifyNodeResponse(
                success=False,
                node_id=node_id,
                old_type="",
                new_type=new_type,
                properties_preserved=[],
                relationships_preserved=0,
                message="Error reclassifying node",
                error=str(e)
            )

    async def search_web_evidence(
        self,
        node_id: str,
        evidence_type: str = "supporting",
        max_results: int = 5,
        auto_create_sources: bool = False,
    ) -> SearchWebEvidenceResponse:
        """Search the web for evidence related to a node.

        Note: This is a placeholder implementation. Full functionality requires
        a configured web search API (e.g., Tavily).

        Args:
            node_id: The node to find evidence for
            evidence_type: Type of evidence to search for
            max_results: Maximum number of results
            auto_create_sources: Whether to auto-create Source nodes

        Returns:
            SearchWebEvidenceResponse with results (or placeholder message)
        """
        try:
            # Check if web search is configured
            tavily_key = os.environ.get("TAVILY_API_KEY")

            if not tavily_key:
                # Get node for context in message
                node = await self.get_node_by_id(node_id)
                query_used = ""
                if node:
                    content = self.extract_node_content(node)
                    query_used = content[:100] if content else ""

                return SearchWebEvidenceResponse(
                    success=False,
                    node_id=node_id,
                    query_used=query_used,
                    results=[],
                    sources_created=0,
                    message="Web search not configured",
                    error="TAVILY_API_KEY environment variable not set. Web search functionality requires a Tavily API key."
                )

            # Get the node
            node = await self.get_node_by_id(node_id)
            if not node:
                return SearchWebEvidenceResponse(
                    success=False,
                    node_id=node_id,
                    query_used="",
                    results=[],
                    sources_created=0,
                    message="Node not found",
                    error="Node not found"
                )

            content = self.extract_node_content(node)
            if not content:
                return SearchWebEvidenceResponse(
                    success=False,
                    node_id=node_id,
                    query_used="",
                    results=[],
                    sources_created=0,
                    message="Node has no content to search for",
                    error="No content available"
                )

            # Build search query
            query = f"{evidence_type} evidence for: {content[:200]}"

            # TODO: Implement actual Tavily search when API key is available
            return SearchWebEvidenceResponse(
                success=False,
                node_id=node_id,
                query_used=query,
                results=[],
                sources_created=0,
                message="Web search not yet implemented",
                error="Full web search functionality coming soon. Configure TAVILY_API_KEY to enable."
            )

        except Exception as e:
            logger.exception(f"Error searching web evidence for {node_id}")
            return SearchWebEvidenceResponse(
                success=False,
                node_id=node_id,
                query_used="",
                results=[],
                sources_created=0,
                message="Error searching for evidence",
                error=str(e)
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
        try:
            # Get both nodes
            primary = await self.get_node_by_id(primary_node_id)
            secondary = await self.get_node_by_id(secondary_node_id)

            if not primary:
                return MergeNodesResponse(
                    success=False,
                    primary_node_id=primary_node_id,
                    secondary_node_id=secondary_node_id,
                    merged_properties=[],
                    relationships_transferred=0,
                    message="Primary node not found",
                    error="Primary node not found"
                )

            if not secondary:
                return MergeNodesResponse(
                    success=False,
                    primary_node_id=primary_node_id,
                    secondary_node_id=secondary_node_id,
                    merged_properties=[],
                    relationships_transferred=0,
                    message="Secondary node not found",
                    error="Secondary node not found"
                )

            primary_type = primary.get("type", "Unknown")
            secondary_type = secondary.get("type", "Unknown")

            if primary_type != secondary_type:
                return MergeNodesResponse(
                    success=False,
                    primary_node_id=primary_node_id,
                    secondary_node_id=secondary_node_id,
                    merged_properties=[],
                    relationships_transferred=0,
                    message=f"Cannot merge nodes of different types ({primary_type} vs {secondary_type})",
                    error="Node type mismatch"
                )

            # Merge properties based on strategy
            merged_properties = []
            smart_merged_content: Dict[str, str] = {}

            for key in secondary.keys():
                if key in ["id", "type", "created_at"]:
                    continue

                primary_value = primary.get(key)
                secondary_value = secondary.get(key)

                if secondary_value is not None:
                    if primary_value is None:
                        # Primary doesn't have this property, take from secondary
                        merged_properties.append(key)
                    elif merge_strategy == "keep_secondary":
                        merged_properties.append(key)
                    elif merge_strategy == "combine":
                        # For text fields, combine content
                        if key in ["text", "description", "content", "notes"]:
                            if primary_value != secondary_value:
                                merged_properties.append(key)
                    elif merge_strategy == "smart":
                        # For text fields, use AI to intelligently merge
                        if key in ["text", "description", "content", "notes"]:
                            if primary_value != secondary_value:
                                merged_properties.append(key)

            # For "smart" strategy, use LLM to merge text content
            if merge_strategy == "smart" and merged_properties:
                for key in merged_properties:
                    if key in ["text", "description", "content", "notes"]:
                        primary_value = primary.get(key, "")
                        secondary_value = secondary.get(key, "")
                        if primary_value and secondary_value:
                            merged_text = await self._smart_merge_text(
                                primary_value, secondary_value, primary_type
                            )
                            smart_merged_content[key] = merged_text

            # Transfer relationships and merge in database
            relationships_transferred = await self._merge_nodes_in_db(
                primary_node_id,
                secondary_node_id,
                merge_strategy,
                merged_properties,
                smart_merged_content
            )

            return MergeNodesResponse(
                success=True,
                primary_node_id=primary_node_id,
                secondary_node_id=secondary_node_id,
                merged_properties=merged_properties,
                relationships_transferred=relationships_transferred,
                message=f"Successfully merged nodes. Transferred {relationships_transferred} relationships."
            )

        except Exception as e:
            logger.exception(f"Error merging nodes {primary_node_id} and {secondary_node_id}")
            return MergeNodesResponse(
                success=False,
                primary_node_id=primary_node_id,
                secondary_node_id=secondary_node_id,
                merged_properties=[],
                relationships_transferred=0,
                message="Error merging nodes",
                error=str(e)
            )

    # ========================================================================
    # Private Helper Methods
    # ========================================================================

    async def _change_node_type(
        self,
        node_id: str,
        old_type: str,
        new_type: str,
    ) -> tuple[bool, List[str]]:
        """Change node type by updating labels.

        Returns tuple of (success, list of preserved property names)
        """
        # Get current node properties
        get_query = """
        MATCH (n {id: $node_id})
        RETURN properties(n) as props
        """

        async with neo4j_conn.get_session() as session:
            result = await session.run(get_query, node_id=node_id)
            record = await result.single()
            if not record:
                return False, []

            props = dict(record["props"])
            property_names = list(props.keys())

            # Remove old label and add new label
            update_query = f"""
            MATCH (n {{id: $node_id}})
            REMOVE n:{old_type}
            SET n:{new_type}
            SET n.updated_at = datetime()
            RETURN n.id as id
            """

            result = await session.run(update_query, node_id=node_id)
            record = await result.single()
            return record is not None, property_names

    async def _smart_merge_text(
        self,
        primary_text: str,
        secondary_text: str,
        node_type: str,
    ) -> str:
        """Use LLM to intelligently merge text from two nodes.

        Args:
            primary_text: Text content from the primary node
            secondary_text: Text content from the secondary node
            node_type: The type of node being merged (for context)

        Returns:
            Merged text that preserves all data points and removes duplicates
        """
        prompt = f"""Merge these two bodies of text from {node_type} nodes, preserving all data points and removing any duplicate information.

--- First Text ---
{primary_text}

--- Second Text ---
{secondary_text}

--- Instructions ---
1. Combine all unique information from both texts
2. Remove any redundant or duplicate statements
3. Maintain a coherent, logical flow
4. Preserve all factual data points
5. Keep the tone and style consistent

Provide the merged text directly without any preamble or explanation:"""

        try:
            response = await self.llm.ainvoke(prompt)
            merged_text = response.content.strip()
            return merged_text
        except Exception as e:
            logger.warning(f"Smart merge failed, falling back to concatenation: {e}")
            # Fallback to simple concatenation if LLM fails
            return f"{primary_text}\n\n---\n\n{secondary_text}"

    async def _merge_nodes_in_db(
        self,
        primary_id: str,
        secondary_id: str,
        merge_strategy: str,
        properties_to_merge: List[str],
        smart_merged_content: Optional[Dict[str, str]] = None,
    ) -> int:
        """Merge nodes in database: transfer relationships and delete secondary.

        Args:
            primary_id: The node to keep
            secondary_id: The node to merge and delete
            merge_strategy: How to handle properties
            properties_to_merge: List of property names to merge
            smart_merged_content: For "smart" strategy, pre-merged content from LLM

        Returns the number of relationships transferred.
        """
        async with neo4j_conn.get_session() as session:
            # First, get the secondary node's properties to merge
            if properties_to_merge:
                get_props_query = """
                MATCH (s {id: $secondary_id})
                RETURN properties(s) as props
                """
                result = await session.run(get_props_query, secondary_id=secondary_id)
                record = await result.single()
                if record:
                    secondary_props = dict(record["props"])

                    # Build property update based on strategy
                    if merge_strategy == "combine":
                        for prop in properties_to_merge:
                            if prop in secondary_props:
                                update_query = f"""
                                MATCH (p {{id: $primary_id}})
                                SET p.{prop} = CASE
                                    WHEN p.{prop} IS NULL THEN $value
                                    WHEN p.{prop} = $value THEN p.{prop}
                                    ELSE p.{prop} + '\\n\\n---\\n\\n' + $value
                                END
                                """
                                await session.run(
                                    update_query,
                                    primary_id=primary_id,
                                    value=str(secondary_props[prop])
                                )
                    elif merge_strategy == "keep_secondary":
                        for prop in properties_to_merge:
                            if prop in secondary_props:
                                update_query = f"""
                                MATCH (p {{id: $primary_id}})
                                SET p.{prop} = $value
                                """
                                await session.run(
                                    update_query,
                                    primary_id=primary_id,
                                    value=secondary_props[prop]
                                )
                    elif merge_strategy == "smart" and smart_merged_content:
                        for prop in properties_to_merge:
                            if prop in smart_merged_content:
                                update_query = f"""
                                MATCH (p {{id: $primary_id}})
                                SET p.{prop} = $value
                                """
                                await session.run(
                                    update_query,
                                    primary_id=primary_id,
                                    value=smart_merged_content[prop]
                                )
                            elif prop in secondary_props:
                                update_query = f"""
                                MATCH (p {{id: $primary_id}})
                                SET p.{prop} = $value
                                """
                                await session.run(
                                    update_query,
                                    primary_id=primary_id,
                                    value=secondary_props[prop]
                                )

            # Transfer relationships using manual approach
            transferred_count = 0
            get_rels_query = """
            MATCH (s {id: $secondary_id})-[r]-(other)
            WHERE other.id <> $primary_id
            RETURN
                CASE WHEN startNode(r) = s THEN 'outgoing' ELSE 'incoming' END as direction,
                type(r) as rel_type,
                properties(r) as props,
                other.id as other_id
            """
            result = await session.run(get_rels_query, secondary_id=secondary_id, primary_id=primary_id)
            relationships_to_transfer = [dict(record) async for record in result]

            for rel in relationships_to_transfer:
                try:
                    if rel["direction"] == "outgoing":
                        create_query = f"""
                        MATCH (p {{id: $primary_id}}), (other {{id: $other_id}})
                        CREATE (p)-[r:{rel['rel_type']}]->(other)
                        SET r = $props
                        """
                    else:
                        create_query = f"""
                        MATCH (p {{id: $primary_id}}), (other {{id: $other_id}})
                        CREATE (other)-[r:{rel['rel_type']}]->(p)
                        SET r = $props
                        """
                    await session.run(
                        create_query,
                        primary_id=primary_id,
                        other_id=rel["other_id"],
                        props=rel["props"]
                    )
                    transferred_count += 1
                except Exception as create_error:
                    logger.warning(f"Failed to transfer relationship: {create_error}")

            # Delete old relationships from secondary
            delete_old_rels_query = """
            MATCH (s {id: $secondary_id})-[r]-()
            DELETE r
            """
            await session.run(delete_old_rels_query, secondary_id=secondary_id)

            # Delete secondary node
            delete_query = """
            MATCH (s {id: $secondary_id})
            DELETE s
            """
            await session.run(delete_query, secondary_id=secondary_id)

            # Update primary node's updated_at
            update_primary_query = """
            MATCH (p {id: $primary_id})
            SET p.updated_at = datetime()
            """
            await session.run(update_primary_query, primary_id=primary_id)

            return transferred_count
