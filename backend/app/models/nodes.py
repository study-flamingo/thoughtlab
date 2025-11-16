from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, List
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


class NodeBase(BaseModel):
    """Base properties for all nodes"""
    id: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class ObservationCreate(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000)
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    subject_ids: Optional[List[str]] = None
    concept_names: Optional[List[str]] = None


class ObservationResponse(NodeBase):
    text: str
    confidence: float
    concept_names: List[str] = []
    type: str = "Observation"


class SourceCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=500)
    url: Optional[str] = None
    source_type: str = Field(default="paper")  # paper, forum, article, etc.
    content: Optional[str] = None
    published_date: Optional[datetime] = None


class SourceResponse(NodeBase):
    title: str
    url: Optional[str] = None
    source_type: str
    content: Optional[str] = None
    published_date: Optional[datetime] = None
    type: str = "Source"


class HypothesisCreate(BaseModel):
    claim: str = Field(..., min_length=1, max_length=10000)
    supporting_evidence_ids: Optional[List[str]] = None
    status: str = Field(default="proposed")  # proposed, tested, confirmed, rejected


class HypothesisResponse(NodeBase):
    claim: str
    status: str
    type: str = "Hypothesis"


class RelationshipCreate(BaseModel):
    from_id: str
    to_id: str
    relationship_type: RelationshipType
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    notes: Optional[str] = None


class RelationshipResponse(BaseModel):
    id: str
    from_id: str
    to_id: str
    relationship_type: RelationshipType
    confidence: Optional[float] = None
    notes: Optional[str] = None
    created_at: Optional[datetime] = None
