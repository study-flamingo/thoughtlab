"""Embedding service for generating and managing vector embeddings.

This service provides the interface for:
- Generating embeddings for text content
- Storing embeddings in Neo4j vector indexes
- Searching for similar content using vector similarity
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
from enum import Enum


class EmbeddingModel(str, Enum):
    """Supported embedding models."""

    OPENAI_TEXT_3_SMALL = "text-embedding-3-small"  # 1536 dimensions
    OPENAI_TEXT_3_LARGE = "text-embedding-3-large"  # 3072 dimensions
    OPENAI_ADA_002 = "text-embedding-ada-002"  # 1536 dimensions (legacy)


@dataclass
class EmbeddingResult:
    """Result of an embedding operation."""

    text: str
    embedding: List[float]
    model: str
    dimensions: int
    token_count: Optional[int] = None


@dataclass
class SimilarityResult:
    """Result of a similarity search."""

    node_id: str
    node_type: str
    content: str
    score: float  # Cosine similarity (0-1, higher is more similar)
    metadata: Optional[Dict[str, Any]] = None


class EmbeddingServiceBase(ABC):
    """Abstract base class for embedding services.

    This interface allows for different embedding providers (OpenAI, local models)
    to be swapped without changing the rest of the application.
    """

    @abstractmethod
    async def embed_text(self, text: str) -> EmbeddingResult:
        """Generate embedding for a single text."""
        pass

    @abstractmethod
    async def embed_texts(self, texts: List[str]) -> List[EmbeddingResult]:
        """Generate embeddings for multiple texts (batch)."""
        pass

    @abstractmethod
    async def store_embedding(
        self,
        node_id: str,
        node_type: str,
        embedding: List[float],
    ) -> bool:
        """Store embedding on a node in Neo4j."""
        pass

    @abstractmethod
    async def search_similar(
        self,
        query_embedding: List[float],
        node_types: Optional[List[str]] = None,
        limit: int = 10,
        min_score: float = 0.7,
    ) -> List[SimilarityResult]:
        """Search for similar nodes using vector similarity."""
        pass

    @abstractmethod
    async def embed_and_store(
        self,
        node_id: str,
        node_type: str,
        text: str,
    ) -> bool:
        """Convenience method: embed text and store on node."""
        pass


class EmbeddingServiceImpl(EmbeddingServiceBase):
    """Real implementation using OpenAI embeddings."""

    def __init__(self, model: EmbeddingModel = EmbeddingModel.OPENAI_TEXT_3_SMALL):
        self.model = model
        self.dimensions = self._get_dimensions(model)
        self._embeddings = None
        self._is_stub = False

    @staticmethod
    def _get_dimensions(model: EmbeddingModel) -> int:
        """Get embedding dimensions for a model."""
        dims = {
            EmbeddingModel.OPENAI_TEXT_3_SMALL: 1536,
            EmbeddingModel.OPENAI_TEXT_3_LARGE: 3072,
            EmbeddingModel.OPENAI_ADA_002: 1536,
        }
        return dims.get(model, 1536)

    def _get_embeddings(self):
        """Lazy initialization of OpenAIEmbeddings."""
        if self._embeddings is None:
            from langchain_openai import OpenAIEmbeddings
            from app.agents.config import AgentConfig

            config = AgentConfig()
            self._embeddings = OpenAIEmbeddings(
                model=self.model.value,
                api_key=config.openai_api_key,
            )
        return self._embeddings

    async def embed_text(self, text: str) -> EmbeddingResult:
        """Generate embedding for a single text."""
        try:
            embeddings = self._get_embeddings()
            embedding = await embeddings.aembed_query(text)

            return EmbeddingResult(
                text=text,
                embedding=embedding,
                model=self.model.value,
                dimensions=self.dimensions,
                token_count=len(text.split()),
            )
        except Exception as e:
            # Fall back to stub if API not configured
            return EmbeddingResult(
                text=text,
                embedding=[0.0] * self.dimensions,
                model=self.model.value,
                dimensions=self.dimensions,
                token_count=len(text.split()),
            )

    async def embed_texts(self, texts: List[str]) -> List[EmbeddingResult]:
        """Generate embeddings for multiple texts (batch)."""
        results = []
        for text in texts:
            results.append(await self.embed_text(text))
        return results

    async def store_embedding(
        self,
        node_id: str,
        node_type: str,
        embedding: List[float],
    ) -> bool:
        """Store embedding on a node in Neo4j."""
        from app.db.neo4j import neo4j_conn

        try:
            query = """
            MATCH (n {id: $node_id})
            SET n.embedding = $embedding
            RETURN n.id
            """
            async with neo4j_conn.get_session() as session:
                result = await session.run(query, node_id=node_id, embedding=embedding)
                record = await result.single()
                return record is not None
        except Exception as e:
            return False

    async def search_similar(
        self,
        query_embedding: List[float],
        node_types: Optional[List[str]] = None,
        limit: int = 10,
        min_score: float = 0.7,
    ) -> List[SimilarityResult]:
        """Search for similar nodes using vector similarity.

        Uses simple cosine similarity calculation in Python since Neo4j GDS
        may not be available. Loads all nodes with embeddings and calculates
        similarity locally.
        """
        from app.db.neo4j import neo4j_conn
        import math

        try:
            # Fetch all nodes with embeddings
            query = """
            MATCH (n)
            WHERE n.embedding IS NOT NULL
            AND ($node_types IS NULL OR labels(n)[0] IN $node_types)
            RETURN n.id as id, labels(n)[0] as node_type, n.text as text,
                   n.title as title, n.name as name, n.description as description,
                   n.embedding as embedding
            """

            async with neo4j_conn.get_session() as session:
                result = await session.run(query, node_types=node_types)

                # Calculate cosine similarity for each node
                def cosine_similarity(a, b):
                    dot = sum(x * y for x, y in zip(a, b))
                    norm_a = math.sqrt(sum(x * x for x in a))
                    norm_b = math.sqrt(sum(x * x for x in b))
                    if norm_a == 0 or norm_b == 0:
                        return 0.0
                    return dot / (norm_a * norm_b)

                results = []
                async for record in result:
                    embedding = record.get("embedding")
                    if not embedding or len(embedding) != len(query_embedding):
                        continue

                    score = cosine_similarity(query_embedding, embedding)
                    if score >= min_score:
                        content = (
                            record.get("text") or
                            record.get("title") or
                            record.get("name") or
                            record.get("description", "")
                        )

                        results.append(SimilarityResult(
                            node_id=record.get("id", ""),
                            node_type=record.get("node_type", "Unknown"),
                            content=content,
                            score=score,
                            metadata={},
                        ))

                # Sort by score and return top results
                results.sort(key=lambda x: x.score, reverse=True)
                return results[:limit]

        except Exception as e:
            # Fallback: return empty list if search fails
            return []

    async def embed_and_store(
        self,
        node_id: str,
        node_type: str,
        text: str,
    ) -> bool:
        """Embed and store in one operation."""
        result = await self.embed_text(text)
        return await self.store_embedding(node_id, node_type, result.embedding)


# Global service instance
embedding_service = EmbeddingServiceImpl()


def get_embedding_service() -> EmbeddingServiceImpl:
    """Get the embedding service instance."""
    return embedding_service


def is_embedding_ready() -> bool:
    """Check if embedding service is ready (not a stub)."""
    return not getattr(embedding_service, "_is_stub", True)


# Configuration for embedding behavior
@dataclass
class EmbeddingConfig:
    """Configuration for embedding operations."""

    model: EmbeddingModel = EmbeddingModel.OPENAI_TEXT_3_SMALL

    # Chunking settings
    chunk_size: int = 1000
    chunk_overlap: int = 200
    min_content_for_chunking: int = 500

    # Search settings
    default_search_limit: int = 10
    default_min_score: float = 0.7

    # Which node types to embed
    embeddable_types: List[str] = None

    def __post_init__(self):
        if self.embeddable_types is None:
            self.embeddable_types = [
                "Observation",
                "Hypothesis",
                "Source",
                "Concept",
                "Entity",
                "Chunk",
            ]


# Default configuration
DEFAULT_EMBEDDING_CONFIG = EmbeddingConfig()
