"""Similarity search using Neo4j vector indexes.

This module provides vector similarity search capabilities:
- Find similar nodes based on text content
- Filter by node types
- Configure minimum similarity thresholds

Uses Neo4j's native vector index for efficient similarity search.
"""

from typing import List, Optional, Dict, Any
import logging

from langchain_openai import OpenAIEmbeddings

from app.db.neo4j import neo4j_conn
from app.ai.config import AIConfig, get_ai_config

logger = logging.getLogger(__name__)


class SimilaritySearch:
    """Vector similarity search using Neo4j indexes.
    
    Provides methods to find semantically similar nodes in the graph
    using vector embeddings and cosine similarity.
    """
    
    # Default vector index name (created in init.cypher)
    DEFAULT_INDEX_NAME = "node_embedding"
    
    def __init__(self, config: Optional[AIConfig] = None):
        """Initialize similarity search.
        
        Args:
            config: AI configuration. If None, uses global config.
        """
        self.config = config or get_ai_config()
        self._embeddings: Optional[OpenAIEmbeddings] = None
    
    @property
    def embeddings(self) -> OpenAIEmbeddings:
        """Lazy initialization of OpenAI embeddings client."""
        if self._embeddings is None:
            if not self.config.is_configured:
                raise RuntimeError(
                    "OpenAI API key not configured. "
                    "Set THOUGHTLAB_OPENAI_API_KEY environment variable."
                )
            
            self._embeddings = OpenAIEmbeddings(
                model=self.config.embedding_model,
                openai_api_key=self.config.openai_api_key,
            )
        
        return self._embeddings
    
    @property
    def is_ready(self) -> bool:
        """Check if similarity search is ready."""
        return self.config.is_configured
    
    async def find_similar(
        self,
        query_text: str,
        exclude_node_id: Optional[str] = None,
        node_types: Optional[List[str]] = None,
        limit: int = 20,
        min_score: Optional[float] = None,
        index_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Find nodes similar to the query text.
        
        Uses vector similarity search with Neo4j's native vector index.
        
        Args:
            query_text: Text to find similar content for.
            exclude_node_id: Node ID to exclude from results (usually the source node).
            node_types: Filter by node types (e.g., ["Observation", "Source"]).
                       If None, searches all node types.
            limit: Maximum number of results to return.
            min_score: Minimum similarity score (0-1). Defaults to config value.
            index_name: Neo4j vector index name. Defaults to "node_embedding".
            
        Returns:
            List of similar nodes with their similarity scores:
            [
                {
                    "node_id": "...",
                    "node_type": "Observation",
                    "content": "...",
                    "score": 0.85,
                }
            ]
        """
        if not self.is_ready:
            logger.warning("Similarity search not ready (AI not configured)")
            return []
        
        min_score = min_score if min_score is not None else self.config.similarity_min_score
        index_name = index_name or self.DEFAULT_INDEX_NAME
        
        try:
            # Generate query embedding
            query_embedding = await self.embeddings.aembed_query(query_text)
            
            # Build the Cypher query
            # We query more than needed to account for filtering
            query_limit = limit * 2 if node_types or exclude_node_id else limit
            
            # Build WHERE clause parts
            where_parts = []
            if exclude_node_id:
                where_parts.append("n.id <> $exclude_id")
            if node_types:
                # Create label filter: (n:Observation OR n:Source OR ...)
                label_checks = " OR ".join([f"n:{nt}" for nt in node_types])
                where_parts.append(f"({label_checks})")
            
            where_clause = ""
            if where_parts:
                where_clause = "WHERE " + " AND ".join(where_parts)
            
            # Query using Neo4j vector index
            query = f"""
            CALL db.index.vector.queryNodes($index_name, $query_limit, $embedding)
            YIELD node as n, score
            {where_clause}
            WITH n, score
            WHERE score >= $min_score
            RETURN n.id as node_id,
                   labels(n)[0] as node_type,
                   COALESCE(n.text, n.title, n.name, n.content, n.description) as content,
                   score
            ORDER BY score DESC
            LIMIT $limit
            """
            
            async with neo4j_conn.get_session() as session:
                result = await session.run(
                    query,
                    index_name=index_name,
                    embedding=query_embedding,
                    exclude_id=exclude_node_id or "",
                    min_score=min_score,
                    query_limit=query_limit,
                    limit=limit,
                )
                
                results = []
                async for record in result:
                    results.append({
                        "node_id": record["node_id"],
                        "node_type": record["node_type"],
                        "content": record["content"],
                        "score": record["score"],
                    })
                
                logger.debug(
                    f"Similarity search found {len(results)} results "
                    f"(min_score={min_score})"
                )
                return results
                
        except Exception as e:
            # Check if it's a "vector index not found" error
            error_str = str(e).lower()
            if "index" in error_str and "not found" in error_str:
                logger.warning(
                    f"Vector index '{index_name}' not found. "
                    "Run the Neo4j init script to create vector indexes."
                )
            else:
                logger.error(f"Similarity search failed: {e}")
            return []
    
    async def find_similar_by_embedding(
        self,
        embedding: List[float],
        exclude_node_id: Optional[str] = None,
        node_types: Optional[List[str]] = None,
        limit: int = 20,
        min_score: Optional[float] = None,
        index_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Find similar nodes using a pre-computed embedding.
        
        Use this when you already have an embedding and want to avoid
        re-computing it.
        
        Args:
            embedding: Pre-computed embedding vector.
            exclude_node_id: Node ID to exclude from results.
            node_types: Filter by node types.
            limit: Maximum results.
            min_score: Minimum similarity score.
            index_name: Neo4j vector index name.
            
        Returns:
            List of similar nodes with scores.
        """
        min_score = min_score if min_score is not None else self.config.similarity_min_score
        index_name = index_name or self.DEFAULT_INDEX_NAME
        
        try:
            query_limit = limit * 2 if node_types or exclude_node_id else limit
            
            where_parts = []
            if exclude_node_id:
                where_parts.append("n.id <> $exclude_id")
            if node_types:
                label_checks = " OR ".join([f"n:{nt}" for nt in node_types])
                where_parts.append(f"({label_checks})")
            
            where_clause = ""
            if where_parts:
                where_clause = "WHERE " + " AND ".join(where_parts)
            
            query = f"""
            CALL db.index.vector.queryNodes($index_name, $query_limit, $embedding)
            YIELD node as n, score
            {where_clause}
            WITH n, score
            WHERE score >= $min_score
            RETURN n.id as node_id,
                   labels(n)[0] as node_type,
                   COALESCE(n.text, n.title, n.name, n.content, n.description) as content,
                   score
            ORDER BY score DESC
            LIMIT $limit
            """
            
            async with neo4j_conn.get_session() as session:
                result = await session.run(
                    query,
                    index_name=index_name,
                    embedding=embedding,
                    exclude_id=exclude_node_id or "",
                    min_score=min_score,
                    query_limit=query_limit,
                    limit=limit,
                )
                
                results = []
                async for record in result:
                    results.append({
                        "node_id": record["node_id"],
                        "node_type": record["node_type"],
                        "content": record["content"],
                        "score": record["score"],
                    })
                
                return results
                
        except Exception as e:
            logger.error(f"Similarity search by embedding failed: {e}")
            return []


# Global instance (lazy initialization)
_similarity_search: Optional[SimilaritySearch] = None


def get_similarity_search() -> SimilaritySearch:
    """Get the global similarity search instance."""
    global _similarity_search
    if _similarity_search is None:
        _similarity_search = SimilaritySearch()
    return _similarity_search

