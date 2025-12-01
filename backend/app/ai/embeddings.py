"""Embedding manager for generating and storing vector embeddings.

This module provides:
- OpenAI embedding generation via LangChain
- Neo4j embedding storage
- Batch embedding support for efficiency

Uses langchain-openai for embedding generation.
"""

from typing import List, Optional
import logging

from langchain_openai import OpenAIEmbeddings

from app.db.neo4j import neo4j_conn
from app.ai.config import AIConfig, get_ai_config

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """Manages embedding generation and storage in Neo4j.
    
    Uses lazy initialization for the OpenAI embeddings client to avoid
    initialization errors when API key is not configured.
    """
    
    def __init__(self, config: Optional[AIConfig] = None):
        """Initialize embedding manager.
        
        Args:
            config: AI configuration. If None, uses global config.
        """
        self.config = config or get_ai_config()
        self._embeddings: Optional[OpenAIEmbeddings] = None
    
    @property
    def embeddings(self) -> OpenAIEmbeddings:
        """Lazy initialization of OpenAI embeddings client.
        
        Raises:
            RuntimeError: If OpenAI API key is not configured.
        """
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
            logger.info(f"Initialized OpenAI embeddings: {self.config.embedding_model}")
        
        return self._embeddings
    
    @property
    def is_ready(self) -> bool:
        """Check if embedding manager is ready to generate embeddings."""
        return self.config.is_configured
    
    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text.
        
        Args:
            text: Text to embed.
            
        Returns:
            List of floats representing the embedding vector.
            
        Raises:
            RuntimeError: If not configured.
        """
        return await self.embeddings.aembed_query(text)
    
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts (batch).
        
        More efficient than calling embed_text multiple times.
        
        Args:
            texts: List of texts to embed.
            
        Returns:
            List of embedding vectors.
        """
        return await self.embeddings.aembed_documents(texts)
    
    async def store_embedding(
        self,
        node_id: str,
        embedding: List[float],
    ) -> bool:
        """Store embedding on a Neo4j node.
        
        Updates the 'embedding' property on the node with the given ID.
        
        Args:
            node_id: The node's unique identifier.
            embedding: The embedding vector to store.
            
        Returns:
            True if successful, False otherwise.
        """
        query = """
        MATCH (n {id: $node_id})
        SET n.embedding = $embedding
        RETURN n.id as id
        """
        
        try:
            async with neo4j_conn.get_session() as session:
                result = await session.run(
                    query,
                    node_id=node_id,
                    embedding=embedding,
                )
                record = await result.single()
                
                if record:
                    logger.debug(f"Stored embedding for node {node_id}")
                    return True
                else:
                    logger.warning(f"Node not found for embedding storage: {node_id}")
                    return False
                    
        except Exception as e:
            logger.error(f"Failed to store embedding for {node_id}: {e}")
            return False
    
    async def embed_and_store(
        self,
        node_id: str,
        text: str,
    ) -> bool:
        """Generate embedding and store on node in one operation.
        
        Convenience method that combines embed_text and store_embedding.
        
        Args:
            node_id: The node's unique identifier.
            text: Text content to embed.
            
        Returns:
            True if both operations succeed, False otherwise.
        """
        if not self.is_ready:
            logger.warning(
                f"Embedding manager not ready, skipping embed for {node_id}"
            )
            return False
        
        try:
            embedding = await self.embed_text(text)
            return await self.store_embedding(node_id, embedding)
        except Exception as e:
            logger.error(f"Failed to embed and store for {node_id}: {e}")
            return False
    
    async def embed_and_store_batch(
        self,
        items: List[tuple[str, str]],
    ) -> int:
        """Embed and store multiple items efficiently.
        
        Uses batch embedding for efficiency, then stores each embedding.
        
        Args:
            items: List of (node_id, text) tuples.
            
        Returns:
            Number of successfully stored embeddings.
        """
        if not items:
            return 0
        
        if not self.is_ready:
            logger.warning("Embedding manager not ready, skipping batch embed")
            return 0
        
        try:
            # Extract texts for batch embedding
            node_ids = [item[0] for item in items]
            texts = [item[1] for item in items]
            
            # Batch embed
            embeddings = await self.embed_texts(texts)
            
            # Store each embedding
            success_count = 0
            for node_id, embedding in zip(node_ids, embeddings):
                if await self.store_embedding(node_id, embedding):
                    success_count += 1
            
            logger.info(
                f"Batch embedded {success_count}/{len(items)} items successfully"
            )
            return success_count
            
        except Exception as e:
            logger.error(f"Failed batch embed and store: {e}")
            return 0


# Global instance (lazy initialization)
_embedding_manager: Optional[EmbeddingManager] = None


def get_embedding_manager() -> EmbeddingManager:
    """Get the global embedding manager instance.
    
    Creates the instance on first call (lazy initialization).
    """
    global _embedding_manager
    if _embedding_manager is None:
        _embedding_manager = EmbeddingManager()
    return _embedding_manager

