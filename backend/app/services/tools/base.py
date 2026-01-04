"""Base infrastructure for tool service operations.

Provides shared configuration, LLM initialization, and Neo4j helper methods
used by all operation modules.
"""

from typing import Optional, List, Dict, Any
import logging

from app.ai.config import get_ai_config, AIConfig
from app.ai.similarity import get_similarity_search, SimilaritySearch
from app.ai.classifier import get_relationship_classifier, RelationshipClassifier
from app.db.neo4j import neo4j_conn
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)


class ToolServiceBase:
    """Base class providing shared infrastructure for tool operations.

    This class handles lazy initialization of:
    - AI configuration
    - Similarity search service
    - Relationship classifier
    - LLM instance for summarization/analysis

    It also provides common helper methods for Neo4j operations.
    """

    def __init__(self):
        """Initialize base tool service."""
        self._config: Optional[AIConfig] = None
        self._similarity_search: Optional[SimilaritySearch] = None
        self._classifier: Optional[RelationshipClassifier] = None
        self._llm: Optional[ChatOpenAI] = None

    @property
    def config(self) -> AIConfig:
        """Lazy initialization of AI configuration."""
        if self._config is None:
            self._config = get_ai_config()
        return self._config

    @property
    def similarity_search(self) -> SimilaritySearch:
        """Lazy initialization of similarity search."""
        if self._similarity_search is None:
            self._similarity_search = get_similarity_search()
        return self._similarity_search

    @property
    def classifier(self) -> RelationshipClassifier:
        """Lazy initialization of relationship classifier."""
        if self._classifier is None:
            self._classifier = get_relationship_classifier()
        return self._classifier

    @property
    def llm(self) -> ChatOpenAI:
        """Lazy initialization of LLM for summarization."""
        if self._llm is None:
            self._llm = ChatOpenAI(
                model=self.config.llm_model,
                openai_api_key=self.config.openai_api_key,
                temperature=0.3,  # Slightly creative for summaries
            )
        return self._llm

    # ========================================================================
    # Neo4j Helper Methods
    # ========================================================================

    async def get_node_by_id(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get a node by ID from Neo4j.

        Args:
            node_id: The unique node identifier

        Returns:
            Node data dict with properties and type, or None if not found
        """
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

    def extract_node_content(self, node: Dict[str, Any]) -> str:
        """Extract text content from a node.

        Tries common content fields in order: text, description, content, title, name.

        Args:
            node: Node data dictionary

        Returns:
            The extracted content string, or empty string if no content found
        """
        for field in ["text", "description", "content", "title", "name"]:
            if field in node and node[field]:
                return str(node[field])
        return ""

    async def get_node_relationships(
        self,
        node_id: str,
        relationship_types: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Get all relationships for a node.

        Args:
            node_id: The node to get relationships for
            relationship_types: Optional filter for specific relationship types

        Returns:
            List of relationship dicts with id, type, confidence, and connected node info
        """
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

    async def get_relationship_by_id(self, edge_id: str) -> Optional[Dict[str, Any]]:
        """Get a relationship by ID.

        Args:
            edge_id: The relationship identifier

        Returns:
            Relationship data dict, or None if not found
        """
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

    async def update_node_confidence(self, node_id: str, new_confidence: float) -> bool:
        """Update node confidence in database.

        Args:
            node_id: The node to update
            new_confidence: New confidence value (0.0 to 1.0)

        Returns:
            True if update succeeded, False otherwise
        """
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

    async def update_relationship_confidence(self, edge_id: str, new_confidence: float) -> bool:
        """Update relationship confidence in database.

        Args:
            edge_id: The relationship to update
            new_confidence: New confidence value (0.0 to 1.0)

        Returns:
            True if update succeeded, False otherwise
        """
        query = """
        MATCH ()-[r {id: $edge_id}]->()
        SET r.confidence = $confidence
        RETURN r.id as id
        """

        async with neo4j_conn.get_session() as session:
            result = await session.run(
                query,
                edge_id=edge_id,
                confidence=new_confidence
            )
            record = await result.single()
            return record is not None
