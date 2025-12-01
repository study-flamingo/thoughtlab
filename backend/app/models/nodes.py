from pydantic import BaseModel, Field, field_validator
from datetime import datetime
from typing import Optional, List, Dict, Any, ClassVar
from enum import Enum
import re


class NodeType(str, Enum):
    SOURCE = "Source"
    OBSERVATION = "Observation"
    HYPOTHESIS = "Hypothesis"
    CONCEPT = "Concept"
    ENTITY = "Entity"
    CHUNK = "Chunk"


# Suggested relationship types (open string, not enum-constrained)
# LLM can create any relationship type; these are UI suggestions
SUGGESTED_RELATIONSHIP_TYPES: List[str] = [
    "SUPPORTS",
    "CONTRADICTS",
    "RELATES_TO",
    "CITES",
    "DERIVED_FROM",
    "OBSERVED_IN",
    "DISCUSSES",
    "INSPIRED_BY",
    "PRECEDES",
    "CAUSES",
    "PART_OF",
    "SIMILAR_TO",
    "HAS_CHUNK",
]


def normalize_relationship_type(value: str) -> str:
    """Normalize relationship type to UPPER_SNAKE_CASE."""
    # Replace spaces and hyphens with underscores
    normalized = re.sub(r'[\s\-]+', '_', value.strip())
    # Remove any non-alphanumeric characters except underscores
    normalized = re.sub(r'[^A-Za-z0-9_]', '', normalized)
    # Convert to uppercase
    return normalized.upper()


class LinkItem(BaseModel):
    """A clickable link with optional label"""
    url: str = Field(..., min_length=1)
    label: Optional[str] = Field(None, max_length=200)


class NodeBase(BaseModel):
    """Base properties for all nodes"""
    id: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    links: Optional[List[LinkItem]] = None


class ObservationCreate(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000)
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    subject_ids: Optional[List[str]] = None
    concept_names: Optional[List[str]] = None
    links: Optional[List[LinkItem]] = None


class ObservationUpdate(BaseModel):
    text: Optional[str] = Field(None, min_length=1, max_length=10000)
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    concept_names: Optional[List[str]] = None
    links: Optional[List[LinkItem]] = None


class ObservationResponse(NodeBase):
    text: str
    confidence: float
    concept_names: List[str] = Field(default_factory=list)
    type: str = "Observation"


class SourceCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=500)
    url: Optional[str] = None  # Primary URL for the source
    source_type: str = Field(default="paper")  # paper, forum, article, etc.
    content: Optional[str] = None
    published_date: Optional[datetime] = None
    links: Optional[List[LinkItem]] = None  # Additional related links


class SourceUpdate(BaseModel):
    title: Optional[str] = Field(None, min_length=1, max_length=500)
    url: Optional[str] = None
    source_type: Optional[str] = None
    content: Optional[str] = None
    published_date: Optional[datetime] = None
    links: Optional[List[LinkItem]] = None


class SourceResponse(NodeBase):
    title: str
    url: Optional[str] = None
    source_type: str
    content: Optional[str] = None
    published_date: Optional[datetime] = None
    type: str = "Source"


class HypothesisCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)  # Short display name for the graph
    claim: str = Field(..., min_length=1, max_length=10000)  # Full hypothesis statement
    supporting_evidence_ids: Optional[List[str]] = None
    status: str = Field(default="proposed")  # proposed, tested, confirmed, rejected
    links: Optional[List[LinkItem]] = None


class HypothesisUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=200)
    claim: Optional[str] = Field(None, min_length=1, max_length=10000)
    status: Optional[str] = None
    links: Optional[List[LinkItem]] = None


class HypothesisResponse(NodeBase):
    name: str
    claim: str
    status: str
    type: str = "Hypothesis"


class EntityCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=500)
    entity_type: str = Field(default="generic")  # person, organization, location, concept, etc.
    description: Optional[str] = None
    properties: Optional[Dict[str, Any]] = None
    links: Optional[List[LinkItem]] = None


class EntityUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=500)
    entity_type: Optional[str] = None
    description: Optional[str] = None
    properties: Optional[Dict[str, Any]] = None
    links: Optional[List[LinkItem]] = None


class EntityResponse(NodeBase):
    name: str
    entity_type: str
    description: Optional[str] = None
    properties: Optional[Dict[str, Any]] = None
    type: str = "Entity"


class ConceptCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=500)
    description: Optional[str] = None
    domain: str = Field(default="general")  # general, science, technology, etc.
    links: Optional[List[LinkItem]] = None


class ConceptUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=500)
    description: Optional[str] = None
    domain: Optional[str] = None
    links: Optional[List[LinkItem]] = None


class ConceptResponse(NodeBase):
    name: str
    description: Optional[str] = None
    domain: str
    type: str = "Concept"


# -----------------------------------------------------------------------------
# Chunk Model (for long-form content chunking)
# -----------------------------------------------------------------------------

class ChunkCreate(BaseModel):
    """Create a chunk of content from a parent Source node."""
    parent_id: str = Field(..., description="ID of the parent Source node")
    content: str = Field(..., min_length=1, max_length=10000)
    chunk_index: int = Field(..., ge=0, description="Position in parent document")
    start_char: Optional[int] = Field(None, ge=0, description="Start character offset in parent")
    end_char: Optional[int] = Field(None, ge=0, description="End character offset in parent")
    metadata: Optional[Dict[str, Any]] = None


class ChunkUpdate(BaseModel):
    """Update an existing chunk."""
    content: Optional[str] = Field(None, min_length=1, max_length=10000)
    metadata: Optional[Dict[str, Any]] = None


class ChunkResponse(NodeBase):
    parent_id: str
    content: str
    chunk_index: int
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    type: str = "Chunk"


# -----------------------------------------------------------------------------
# Relationship Models (open string types with suggestions)
# -----------------------------------------------------------------------------

class RelationshipCreate(BaseModel):
    from_id: str
    to_id: str
    relationship_type: str = Field(..., min_length=1, max_length=100)
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    notes: Optional[str] = None
    # Authorship tracking
    created_by: Optional[str] = Field(None, description="User ID or 'system-llm'")
    # Inverse relationship metadata for asymmetrical relationships when reversed
    inverse_relationship_type: Optional[str] = Field(None, max_length=100)
    inverse_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    inverse_notes: Optional[str] = None

    @field_validator('relationship_type', 'inverse_relationship_type', mode='before')
    @classmethod
    def normalize_type(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        return normalize_relationship_type(v)


class RelationshipUpdate(BaseModel):
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    notes: Optional[str] = None
    relationship_type: Optional[str] = Field(None, max_length=100)
    inverse_relationship_type: Optional[str] = Field(None, max_length=100)
    inverse_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    inverse_notes: Optional[str] = None
    # Feedback fields (for LLM-suggested relationships)
    approved: Optional[bool] = Field(None, description="User has reviewed this relationship")
    feedback_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="User rating of suggestion quality")

    @field_validator('relationship_type', 'inverse_relationship_type', mode='before')
    @classmethod
    def normalize_type(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        return normalize_relationship_type(v)


class RelationshipResponse(BaseModel):
    id: str
    from_id: str
    to_id: str
    relationship_type: str
    confidence: Optional[float] = None
    notes: Optional[str] = None
    # Authorship tracking
    created_by: Optional[str] = None
    approved: Optional[bool] = None
    feedback_score: Optional[float] = None
    # Inverse relationship
    inverse_relationship_type: Optional[str] = None
    inverse_confidence: Optional[float] = None
    inverse_notes: Optional[str] = None
    created_at: Optional[datetime] = None
