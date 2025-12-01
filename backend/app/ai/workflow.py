"""AI workflow orchestrator for node processing.

This module provides the main entry point for AI-powered node processing:
1. Chunking (for long content like Sources)
2. Embedding generation
3. Similarity search to find related nodes
4. LLM-based relationship classification
5. Suggestion/relationship creation based on confidence

The workflow updates the Activity Feed with progress and results.
"""

from typing import Optional, List, Dict, Any
from dataclasses import dataclass
import uuid
import logging

from app.ai.config import AIConfig, get_ai_config
from app.ai.embeddings import EmbeddingManager, get_embedding_manager
from app.ai.similarity import SimilaritySearch, get_similarity_search
from app.ai.classifier import RelationshipClassifier, get_relationship_classifier
from app.utils.chunking import chunk_text, should_chunk, Chunk
from app.services.graph_service import graph_service
from app.services.activity_service import activity_service
from app.models.activity import ProcessingData, SuggestionData

logger = logging.getLogger(__name__)


@dataclass
class ProcessingResult:
    """Result of processing a node through the AI workflow."""
    
    node_id: str
    node_type: str
    success: bool
    chunks_created: int = 0
    embeddings_created: int = 0
    candidates_found: int = 0
    suggestions_created: int = 0
    auto_created_relationships: int = 0
    error: Optional[str] = None


class AIWorkflow:
    """Orchestrates the AI processing workflow for nodes.
    
    This is the main entry point for processing nodes through:
    1. Chunking (for long content)
    2. Embedding generation
    3. Similarity search
    4. Relationship classification
    5. Suggestion/relationship creation
    
    The workflow is designed to:
    - Update the Activity Feed with progress
    - Handle failures gracefully
    - Support future background job integration (ARQ)
    """
    
    def __init__(self, config: Optional[AIConfig] = None):
        """Initialize AI workflow.
        
        Args:
            config: AI configuration. If None, uses global config.
        """
        self.config = config or get_ai_config()
        self._embedding_manager: Optional[EmbeddingManager] = None
        self._similarity_search: Optional[SimilaritySearch] = None
        self._classifier: Optional[RelationshipClassifier] = None
    
    @property
    def embedding_manager(self) -> EmbeddingManager:
        """Lazy initialization of embedding manager."""
        if self._embedding_manager is None:
            self._embedding_manager = get_embedding_manager()
        return self._embedding_manager
    
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
    def is_ready(self) -> bool:
        """Check if AI workflow is properly configured."""
        return self.config.is_configured
    
    async def process_node(
        self,
        node_id: str,
        node_type: str,
        content: str,
        node_label: str,
    ) -> ProcessingResult:
        """Process a node through the full AI workflow.
        
        This is the main entry point. It orchestrates:
        1. Chunking (for long content)
        2. Embedding generation
        3. Similarity search
        4. Relationship classification
        5. Suggestion/relationship creation
        
        Args:
            node_id: The node to process.
            node_type: Node type (Observation, Source, etc.).
            content: Text content to process.
            node_label: Display label for activity feed.
            
        Returns:
            ProcessingResult with statistics.
        """
        result = ProcessingResult(
            node_id=node_id,
            node_type=node_type,
            success=False,
        )
        
        if not self.is_ready:
            result.error = "AI not configured (missing THOUGHTLAB_OPENAI_API_KEY)"
            logger.warning(result.error)
            return result
        
        # Create unique group ID for this processing run
        group_id = f"process-{node_id}-{uuid.uuid4().hex[:8]}"
        
        try:
            # Step 1: Start processing
            await self._update_status(
                group_id, "started",
                f"Processing: {node_label[:50]}...",
                node_id, node_type, node_label,
            )
            
            # Step 2: Chunking (for long content)
            chunks = await self._chunk_content(
                node_id, node_type, content, group_id, node_label
            )
            result.chunks_created = len(chunks) if chunks else 0
            
            # Step 3: Embedding
            embeddings_count = await self._embed_content(
                node_id, node_type, content, chunks, group_id, node_label
            )
            result.embeddings_created = embeddings_count
            
            # Step 4: Find similar nodes
            candidates = await self._find_candidates(
                node_id, content, group_id, node_label
            )
            result.candidates_found = len(candidates)
            
            # Step 5: Classify relationships and create suggestions
            suggestions, auto_created = await self._classify_and_create(
                node_id, node_type, content, node_label, candidates, group_id
            )
            result.suggestions_created = suggestions
            result.auto_created_relationships = auto_created
            
            # Step 6: Mark complete
            await self._update_status(
                group_id, "completed",
                f"Processed: {node_label[:50]}... ({suggestions} suggestions, {auto_created} auto-created)",
                node_id, node_type, node_label,
                chunks_created=result.chunks_created,
                embeddings_created=result.embeddings_created,
                suggestions_found=result.suggestions_created,
            )
            
            result.success = True
            logger.info(
                f"Processed {node_type} {node_id}: "
                f"{result.embeddings_created} embeddings, "
                f"{result.candidates_found} candidates, "
                f"{result.suggestions_created} suggestions, "
                f"{result.auto_created_relationships} auto-created"
            )
            
        except Exception as e:
            logger.exception(f"Error processing node {node_id}")
            result.error = str(e)
            
            await self._update_status(
                group_id, "failed",
                f"Failed: {node_label[:50]}... - {str(e)[:100]}",
                node_id, node_type, node_label,
                error_message=str(e),
            )
        
        return result
    
    async def _chunk_content(
        self,
        node_id: str,
        node_type: str,
        content: str,
        group_id: str,
        node_label: str,
    ) -> Optional[List[Chunk]]:
        """Chunk content if needed (for long content like Sources)."""
        # Only chunk Sources and long content
        if node_type not in ["Source"] or not should_chunk(content):
            return None
        
        await self._update_status(
            group_id, "chunking",
            f"Chunking: {node_label[:50]}...",
            node_id, node_type, node_label,
        )
        
        chunks = chunk_text(
            content,
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )
        
        # Store chunks in Neo4j
        from app.models.nodes import ChunkCreate
        
        for chunk in chunks:
            chunk_data = ChunkCreate(
                source_id=node_id,
                content=chunk.content,
                chunk_index=chunk.chunk_index,
                start_char=chunk.start_char,
                end_char=chunk.end_char,
                metadata=chunk.metadata,
            )
            await graph_service.create_chunk(chunk_data, created_by="system-llm")
        
        logger.debug(f"Created {len(chunks)} chunks for {node_id}")
        return chunks
    
    async def _embed_content(
        self,
        node_id: str,
        node_type: str,
        content: str,
        chunks: Optional[List[Chunk]],
        group_id: str,
        node_label: str,
    ) -> int:
        """Generate and store embeddings."""
        await self._update_status(
            group_id, "embedding",
            f"Embedding: {node_label[:50]}...",
            node_id, node_type, node_label,
        )
        
        count = 0
        
        if chunks:
            # Embed each chunk
            # First, get chunk IDs from Neo4j
            from app.db.neo4j import neo4j_conn
            
            for chunk in chunks:
                chunk_query = """
                MATCH (ch:Chunk {source_id: $source_id, chunk_index: $chunk_index})
                RETURN ch.id as id
                """
                async with neo4j_conn.get_session() as session:
                    result = await session.run(
                        chunk_query,
                        source_id=node_id,
                        chunk_index=chunk.chunk_index,
                    )
                    record = await result.single()
                    if record:
                        success = await self.embedding_manager.embed_and_store(
                            record["id"],
                            chunk.content,
                        )
                        if success:
                            count += 1
        else:
            # Embed full content on the node
            success = await self.embedding_manager.embed_and_store(node_id, content)
            if success:
                count = 1
        
        logger.debug(f"Created {count} embeddings for {node_id}")
        return count
    
    async def _find_candidates(
        self,
        node_id: str,
        content: str,
        group_id: str,
        node_label: str,
    ) -> List[Dict[str, Any]]:
        """Find candidate nodes for relationship discovery."""
        await self._update_status(
            group_id, "analyzing",
            f"Finding connections: {node_label[:50]}...",
            node_id, "Unknown", node_label,
        )
        
        candidates = await self.similarity_search.find_similar(
            query_text=content,
            exclude_node_id=node_id,
            limit=self.config.max_similar_nodes,
            min_score=self.config.similarity_min_score,
        )
        
        logger.debug(f"Found {len(candidates)} candidates for {node_id}")
        return candidates
    
    async def _classify_and_create(
        self,
        node_id: str,
        node_type: str,
        content: str,
        node_label: str,
        candidates: List[Dict[str, Any]],
        group_id: str,
    ) -> tuple[int, int]:
        """Classify relationships and create suggestions/relationships.
        
        For each candidate:
        1. Use LLM to classify the relationship
        2. Based on confidence:
           - >= auto_create_threshold: Create relationship automatically
           - >= suggest_threshold: Create suggestion for user review
           - < suggest_threshold: Discard silently
        
        Returns:
            Tuple of (suggestions_created, auto_created_relationships)
        """
        suggestions_created = 0
        auto_created = 0
        
        for candidate in candidates:
            # Skip if no content to analyze
            if not candidate.get("content"):
                continue
            
            # Classify the relationship
            classification = await self.classifier.classify(
                source_content=content,
                source_type=node_type,
                target_content=candidate["content"],
                target_type=candidate["node_type"],
            )
            
            # Skip if classification failed or relationship is not valid
            if classification is None or not classification.is_valid:
                continue
            
            # Determine action based on confidence
            if classification.confidence >= self.config.auto_create_threshold:
                # Auto-create relationship
                rel_id = await graph_service.create_relationship(
                    from_id=node_id,
                    to_id=candidate["node_id"],
                    rel_type=classification.relationship_type,
                    properties={
                        "confidence": classification.confidence,
                        "notes": classification.reasoning,
                    },
                    created_by="system-llm",
                )
                if rel_id:
                    auto_created += 1
                    logger.info(
                        f"Auto-created {classification.relationship_type} "
                        f"from {node_id} to {candidate['node_id']} "
                        f"(confidence: {classification.confidence:.2f})"
                    )
                    
            elif classification.confidence >= self.config.suggest_threshold:
                # Create suggestion for user review
                suggestion_data = SuggestionData(
                    from_node_id=node_id,
                    from_node_type=node_type,
                    from_node_label=node_label[:50],
                    to_node_id=candidate["node_id"],
                    to_node_type=candidate["node_type"],
                    to_node_label=(candidate["content"] or "")[:50],
                    relationship_type=classification.relationship_type,
                    confidence=classification.confidence,
                    reasoning=classification.reasoning,
                )
                await activity_service.create_suggestion(suggestion_data)
                suggestions_created += 1
                logger.debug(
                    f"Created suggestion: {classification.relationship_type} "
                    f"from {node_id} to {candidate['node_id']} "
                    f"(confidence: {classification.confidence:.2f})"
                )
            # else: confidence < suggest_threshold, discard silently
        
        return suggestions_created, auto_created
    
    async def _update_status(
        self,
        group_id: str,
        stage: str,
        message: str,
        node_id: str,
        node_type: str,
        node_label: str,
        **kwargs,
    ):
        """Update processing status in activity feed."""
        processing_data = ProcessingData(
            node_id=node_id,
            node_type=node_type,
            node_label=node_label,
            stage=stage,
            **kwargs,
        )
        
        await activity_service.update_processing_status(
            group_id=group_id,
            stage=stage,
            message=message,
            processing_data=processing_data,
        )


# Global workflow instance (lazy initialization)
_ai_workflow: Optional[AIWorkflow] = None


def get_ai_workflow() -> AIWorkflow:
    """Get the global AI workflow instance."""
    global _ai_workflow
    if _ai_workflow is None:
        _ai_workflow = AIWorkflow()
    return _ai_workflow


# Convenience function for triggering processing
async def trigger_ai_processing(
    node_id: str,
    node_type: str,
    content: str,
    node_label: str,
) -> ProcessingResult:
    """Trigger AI processing for a node.
    
    This is the main entry point for processing nodes through the AI workflow.
    Currently runs synchronously, but designed for easy ARQ integration.
    
    Args:
        node_id: The node to process.
        node_type: Node type (Observation, Source, etc.).
        content: Text content to process.
        node_label: Display label for activity feed.
        
    Returns:
        ProcessingResult with statistics.
    """
    workflow = get_ai_workflow()
    return await workflow.process_node(
        node_id=node_id,
        node_type=node_type,
        content=content,
        node_label=node_label,
    )

