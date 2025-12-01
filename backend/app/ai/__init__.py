"""AI integration module for ThoughtLab.

This module provides AI-powered features:
- Embedding generation using OpenAI
- Vector similarity search using Neo4j
- Relationship classification using LLM
- Processing workflow orchestration

Components:
- config.py: AI configuration settings
- embeddings.py: OpenAI embeddings + Neo4j storage
- similarity.py: Vector similarity search
- classifier.py: Relationship classification
- workflow.py: Main processing workflow
"""

from app.ai.config import AIConfig, get_ai_config
from app.ai.embeddings import EmbeddingManager, get_embedding_manager
from app.ai.similarity import SimilaritySearch, get_similarity_search
from app.ai.classifier import (
    RelationshipClassifier,
    RelationshipClassification,
    get_relationship_classifier,
)
from app.ai.workflow import (
    AIWorkflow,
    ProcessingResult,
    get_ai_workflow,
    trigger_ai_processing,
)

__all__ = [
    # Config
    "AIConfig",
    "get_ai_config",
    # Embeddings
    "EmbeddingManager",
    "get_embedding_manager",
    # Similarity
    "SimilaritySearch",
    "get_similarity_search",
    # Classifier
    "RelationshipClassifier",
    "RelationshipClassification",
    "get_relationship_classifier",
    # Workflow
    "AIWorkflow",
    "ProcessingResult",
    "get_ai_workflow",
    "trigger_ai_processing",
]

