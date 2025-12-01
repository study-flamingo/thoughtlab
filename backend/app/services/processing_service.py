"""Processing service for background node analysis workflow.

This service orchestrates the background processing that happens when a node
is created or updated:

1. **Chunking**: Break long content into embeddable pieces
2. **Embedding**: Generate vector embeddings for each chunk (or full content if short)
3. **Analysis**: Use LLM to find potential relationships with existing nodes
4. **Suggestions**: Create activity feed items for user review

The workflow is designed to:
- Run asynchronously (not block API responses)
- Update activity feed with progress
- Handle failures gracefully
- Support future ARQ background job integration

NOTE: This service now delegates to the AI workflow (app.ai.workflow) when AI
is configured. The legacy stub implementation is preserved for fallback.
"""

from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from datetime import datetime, UTC
import uuid
import logging

from app.services.graph_service import graph_service
from app.services.activity_service import activity_service
from app.services.embedding_service import embedding_service, is_embedding_ready, DEFAULT_EMBEDDING_CONFIG
from app.utils.chunking import chunk_text, should_chunk, Chunk
from app.models.activity import (
    ActivityType,
    ProcessingData,
    SuggestionData,
    SuggestionThresholds,
    DEFAULT_THRESHOLDS,
)

logger = logging.getLogger(__name__)


@dataclass
class ProcessingResult:
    """Result of processing a node."""
    node_id: str
    node_type: str
    success: bool
    chunks_created: int = 0
    embeddings_created: int = 0
    suggestions_found: int = 0
    auto_created_relationships: int = 0
    error: Optional[str] = None


def _is_ai_workflow_ready() -> bool:
    """Check if AI workflow is ready to use."""
    try:
        from app.ai.config import get_ai_config
        return get_ai_config().is_configured
    except Exception:
        return False


class ProcessingService:
    """Service for background node processing.
    
    This service is the orchestrator for the chunking → embedding → analysis pipeline.
    Currently designed to be called directly, but structured for easy ARQ integration.
    """
    
    def __init__(self):
        self.config = DEFAULT_EMBEDDING_CONFIG
        self.thresholds = DEFAULT_THRESHOLDS
    
    async def process_node(
        self,
        node_id: str,
        node_type: str,
        content: str,
        node_label: str,
        created_by: Optional[str] = None,
    ) -> ProcessingResult:
        """Process a node through the full pipeline.
        
        This is the main entry point for processing. It:
        1. Creates an activity to track progress
        2. Chunks content if needed
        3. Generates embeddings
        4. Analyzes for potential relationships
        5. Creates suggestions or auto-creates relationships
        
        Args:
            node_id: The node to process
            node_type: Node type (Observation, Source, etc.)
            content: Text content to process
            node_label: Display label for activity feed
            created_by: User who created the node
            
        Returns:
            ProcessingResult with statistics
        """
        # Create unique group ID for this processing run
        group_id = f"process-{node_id}-{uuid.uuid4().hex[:8]}"
        
        result = ProcessingResult(
            node_id=node_id,
            node_type=node_type,
            success=False,
        )
        
        try:
            # Step 1: Start processing activity
            await self._update_status(
                group_id=group_id,
                stage="started",
                message=f"Processing: {node_label[:50]}...",
                node_id=node_id,
                node_type=node_type,
                node_label=node_label,
            )
            
            # Step 2: Chunking
            chunks = await self._chunk_content(
                node_id=node_id,
                node_type=node_type,
                content=content,
                group_id=group_id,
                node_label=node_label,
            )
            result.chunks_created = len(chunks) if chunks else 0
            
            # Step 3: Embedding
            embeddings_count = await self._embed_content(
                node_id=node_id,
                node_type=node_type,
                content=content,
                chunks=chunks,
                group_id=group_id,
                node_label=node_label,
            )
            result.embeddings_created = embeddings_count
            
            # Step 4: Relationship Analysis
            suggestions, auto_created = await self._analyze_relationships(
                node_id=node_id,
                node_type=node_type,
                content=content,
                node_label=node_label,
                group_id=group_id,
            )
            result.suggestions_found = len(suggestions)
            result.auto_created_relationships = auto_created
            
            # Step 5: Mark complete
            await self._update_status(
                group_id=group_id,
                stage="completed",
                message=f"Processed: {node_label[:50]}... ({result.suggestions_found} suggestions)",
                node_id=node_id,
                node_type=node_type,
                node_label=node_label,
                chunks_created=result.chunks_created,
                embeddings_created=result.embeddings_created,
                suggestions_found=result.suggestions_found,
            )
            
            result.success = True
            
        except Exception as e:
            logger.exception(f"Error processing node {node_id}")
            result.error = str(e)
            
            await self._update_status(
                group_id=group_id,
                stage="failed",
                message=f"Failed: {node_label[:50]}... - {str(e)[:100]}",
                node_id=node_id,
                node_type=node_type,
                node_label=node_label,
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
        """Chunk content if it's long enough.
        
        Only Source nodes typically need chunking. Observations, hypotheses, etc.
        are usually short enough to embed directly.
        """
        # Only chunk Sources (and potentially long Observations in future)
        if node_type not in ["Source"] or not should_chunk(content):
            return None
        
        await self._update_status(
            group_id=group_id,
            stage="chunking",
            message=f"Chunking: {node_label[:50]}...",
            node_id=node_id,
            node_type=node_type,
            node_label=node_label,
        )
        
        # Chunk the content
        chunks = chunk_text(
            content,
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )
        
        # Store chunks in Neo4j
        for chunk in chunks:
            from app.models.nodes import ChunkCreate
            
            chunk_data = ChunkCreate(
                source_id=node_id,
                content=chunk.content,
                chunk_index=chunk.chunk_index,
                start_char=chunk.start_char,
                end_char=chunk.end_char,
                metadata=chunk.metadata,
            )
            await graph_service.create_chunk(chunk_data, created_by="system-llm")
        
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
        """Generate and store embeddings.
        
        If content was chunked, embed each chunk.
        Otherwise, embed the full content directly on the node.
        """
        if not is_embedding_ready():
            logger.info("Embedding service not ready (stub), skipping embeddings")
            return 0
        
        await self._update_status(
            group_id=group_id,
            stage="embedding",
            message=f"Embedding: {node_label[:50]}...",
            node_id=node_id,
            node_type=node_type,
            node_label=node_label,
        )
        
        embeddings_created = 0
        
        if chunks:
            # Embed each chunk
            # TODO: Use batch embedding for efficiency
            for chunk in chunks:
                success = await embedding_service.embed_and_store(
                    node_id=f"chunk-{node_id}-{chunk.chunk_index}",
                    node_type="Chunk",
                    text=chunk.content,
                )
                if success:
                    embeddings_created += 1
        else:
            # Embed full content on the node
            success = await embedding_service.embed_and_store(
                node_id=node_id,
                node_type=node_type,
                text=content,
            )
            if success:
                embeddings_created = 1
        
        return embeddings_created
    
    async def _analyze_relationships(
        self,
        node_id: str,
        node_type: str,
        content: str,
        node_label: str,
        group_id: str,
    ) -> tuple[List[Dict[str, Any]], int]:
        """Analyze content to find potential relationships.
        
        This is where the LLM magic happens:
        1. Search for similar content using vector similarity
        2. Use LLM to classify potential relationships
        3. Create suggestions or auto-create based on confidence
        
        Returns:
            Tuple of (suggestions created, auto-created count)
        """
        await self._update_status(
            group_id=group_id,
            stage="analyzing",
            message=f"Finding connections: {node_label[:50]}...",
            node_id=node_id,
            node_type=node_type,
            node_label=node_label,
        )
        
        suggestions = []
        auto_created = 0
        
        # TODO: Implement actual relationship analysis with LangChain
        # This will involve:
        # 1. Vector similarity search to find candidate nodes
        # 2. LLM call to classify relationship type and confidence
        # 3. Create suggestions or auto-create based on thresholds
        
        # Example of what the LLM integration will look like:
        """
        # Find similar nodes
        similar_nodes = await embedding_service.search_similar(
            query_embedding=node_embedding,
            node_types=["Observation", "Hypothesis", "Source"],
            limit=20,
            min_score=0.5,
        )
        
        # For each similar node, ask LLM to classify relationship
        for similar in similar_nodes:
            classification = await llm_service.classify_relationship(
                source_content=content,
                target_content=similar.content,
                source_type=node_type,
                target_type=similar.node_type,
            )
            
            # classification = {"type": "SUPPORTS", "confidence": 0.85, "reasoning": "..."}
            
            action = self.thresholds.get_action(classification["confidence"])
            
            if action == "auto_create":
                # Create relationship automatically
                await graph_service.create_relationship(
                    from_id=node_id,
                    to_id=similar.node_id,
                    rel_type=classification["type"],
                    properties={"confidence": classification["confidence"]},
                    created_by="system-llm",
                )
                auto_created += 1
                
            elif action == "suggest":
                # Create activity for user review
                suggestion = SuggestionData(
                    from_node_id=node_id,
                    from_node_type=node_type,
                    from_node_label=node_label,
                    to_node_id=similar.node_id,
                    to_node_type=similar.node_type,
                    to_node_label=similar.content[:50],
                    relationship_type=classification["type"],
                    confidence=classification["confidence"],
                    reasoning=classification["reasoning"],
                )
                await activity_service.create_suggestion(suggestion)
                suggestions.append(suggestion)
        """
        
        # For now, log that we would analyze
        logger.info(
            f"Would analyze relationships for {node_type} {node_id} "
            f"(LLM integration pending)"
        )
        
        return suggestions, auto_created
    
    async def _update_status(
        self,
        group_id: str,
        stage: str,
        message: str,
        node_id: str,
        node_type: str,
        node_label: str,
        chunks_created: Optional[int] = None,
        embeddings_created: Optional[int] = None,
        suggestions_found: Optional[int] = None,
        error_message: Optional[str] = None,
    ) -> str:
        """Update processing status in activity feed."""
        processing_data = ProcessingData(
            node_id=node_id,
            node_type=node_type,
            node_label=node_label,
            stage=stage,
            chunks_created=chunks_created,
            embeddings_created=embeddings_created,
            suggestions_found=suggestions_found,
            error_message=error_message,
        )
        
        return await activity_service.update_processing_status(
            group_id=group_id,
            stage=stage,
            message=message,
            processing_data=processing_data,
        )
    
    async def reprocess_node(self, node_id: str) -> ProcessingResult:
        """Re-process an existing node (e.g., after content update).
        
        This will:
        1. Delete existing chunks
        2. Clear existing embeddings
        3. Run full processing pipeline again
        """
        # Get node details
        node = await graph_service.get_node(node_id)
        if not node:
            return ProcessingResult(
                node_id=node_id,
                node_type="Unknown",
                success=False,
                error="Node not found",
            )
        
        node_type = node.get("type", "Unknown")
        
        # Get content based on node type
        content = self._get_node_content(node, node_type)
        node_label = self._get_node_label(node, node_type)
        
        # Delete existing chunks if Source
        if node_type == "Source":
            await graph_service.delete_source_chunks(node_id)
        
        # Re-run processing
        return await self.process_node(
            node_id=node_id,
            node_type=node_type,
            content=content,
            node_label=node_label,
        )
    
    @staticmethod
    def _get_node_content(node: Dict[str, Any], node_type: str) -> str:
        """Extract embeddable content from a node."""
        content_fields = {
            "Observation": "text",
            "Hypothesis": "description",
            "Source": "content",
            "Concept": "description",
            "Entity": "description",
        }
        field = content_fields.get(node_type, "text")
        return node.get(field, "") or ""
    
    @staticmethod
    def _get_node_label(node: Dict[str, Any], node_type: str) -> str:
        """Get display label for a node."""
        label_fields = {
            "Observation": "text",
            "Hypothesis": "name",
            "Source": "title",
            "Concept": "name",
            "Entity": "name",
        }
        field = label_fields.get(node_type, "id")
        label = node.get(field, node.get("id", "Unknown"))
        # Truncate if too long
        return label[:100] if label else "Unknown"


# Global service instance
processing_service = ProcessingService()


# Utility for triggering processing (can be called from routes or background jobs)
async def trigger_node_processing(
    node_id: str,
    node_type: str,
    content: str,
    node_label: str,
    created_by: Optional[str] = None,
) -> ProcessingResult:
    """Trigger background processing for a node.
    
    This is the function that should be called after node creation.
    Currently runs synchronously, but will be wrapped in ARQ job later.
    
    When AI is configured (THOUGHTLAB_OPENAI_API_KEY set), this delegates
    to the AI workflow for full LangChain-powered processing. Otherwise,
    it falls back to the legacy stub implementation.
    
    Usage:
        # After creating a node
        result = await trigger_node_processing(
            node_id=new_node_id,
            node_type="Observation",
            content=observation_text,
            node_label=observation_text[:50],
        )
    """
    # Use AI workflow if configured
    if _is_ai_workflow_ready():
        from app.ai.workflow import trigger_ai_processing, ProcessingResult as AIResult
        
        ai_result = await trigger_ai_processing(
            node_id=node_id,
            node_type=node_type,
            content=content,
            node_label=node_label,
        )
        
        # Convert AI result to legacy format
        return ProcessingResult(
            node_id=ai_result.node_id,
            node_type=ai_result.node_type,
            success=ai_result.success,
            chunks_created=ai_result.chunks_created,
            embeddings_created=ai_result.embeddings_created,
            suggestions_found=ai_result.suggestions_created,
            auto_created_relationships=ai_result.auto_created_relationships,
            error=ai_result.error,
        )
    
    # Fall back to legacy stub implementation
    logger.info(
        "AI not configured, using legacy processing stub. "
        "Set THOUGHTLAB_OPENAI_API_KEY for full AI processing."
    )
    return await processing_service.process_node(
        node_id=node_id,
        node_type=node_type,
        content=content,
        node_label=node_label,
        created_by=created_by,
    )

