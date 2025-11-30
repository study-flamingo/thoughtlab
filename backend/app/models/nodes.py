from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum


class NodeType(str, Enum):
    SOURCE = "Source"
    OBSERVATION = "Observation"
    HYPOTHESIS = "Hypothesis"
    CONCEPT = "Concept"
    ENTITY = "Entity"


class RelationshipType(str, Enum):
    SUPPORTS = "SUPPORTS"
    CONTRADICTS = "CONTRADICTS"
    RELATES_TO = "RELATES_TO"
    OBSERVED_IN = "OBSERVED_IN"
    DISCUSSES = "DISCUSSES"
    EXTRACTED_FROM = "EXTRACTED_FROM"
    DERIVED_FROM = "DERIVED_FROM"


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


class RelationshipCreate(BaseModel):
    from_id: str
    to_id: str
    relationship_type: RelationshipType
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    notes: Optional[str] = None
    # Inverse relationship metadata for asymmetrical relationships when reversed
    inverse_relationship_type: Optional[RelationshipType] = None
    inverse_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    inverse_notes: Optional[str] = None


class RelationshipUpdate(BaseModel):
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    notes: Optional[str] = None
    relationship_type: Optional[RelationshipType] = None
    inverse_relationship_type: Optional[RelationshipType] = None
    inverse_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    inverse_notes: Optional[str] = None


class RelationshipResponse(BaseModel):
    id: str
    from_id: str
    to_id: str
    relationship_type: RelationshipType
    confidence: Optional[float] = None
    notes: Optional[str] = None
    inverse_relationship_type: Optional[RelationshipType] = None
    inverse_confidence: Optional[float] = None
    inverse_notes: Optional[str] = None
    created_at: Optional[datetime] = None
