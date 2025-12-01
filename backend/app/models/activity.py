"""Activity Feed models for tracking system events and user interactions.

Activities represent events in the system that users should be aware of:
- Node/relationship creation, updates, deletion (display-only)
- LLM-suggested relationships (interactive - approve/reject)
- Processing status (chunking, embedding, analysis)
- Errors and warnings

Activities can optionally carry a payload that enables interaction (e.g., opening
a node in the inspector, approving a suggested relationship).
"""

from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, Dict, Any, List
from enum import Enum


class ActivityType(str, Enum):
    """Types of activities that can appear in the feed."""
    
    # Node lifecycle (display-only)
    NODE_CREATED = "node_created"
    NODE_UPDATED = "node_updated"
    NODE_DELETED = "node_deleted"
    
    # Relationship lifecycle (display-only for user-created)
    RELATIONSHIP_CREATED = "relationship_created"
    RELATIONSHIP_UPDATED = "relationship_updated"
    RELATIONSHIP_DELETED = "relationship_deleted"
    
    # LLM suggestions (interactive)
    RELATIONSHIP_SUGGESTED = "relationship_suggested"
    RELATIONSHIP_AUTO_CREATED = "relationship_auto_created"  # Auto-created by LLM (confidence > 0.8)
    
    # Processing status (dynamic updates)
    PROCESSING_STARTED = "processing_started"
    PROCESSING_CHUNKING = "processing_chunking"
    PROCESSING_EMBEDDING = "processing_embedding"
    PROCESSING_ANALYZING = "processing_analyzing"
    PROCESSING_COMPLETED = "processing_completed"
    PROCESSING_FAILED = "processing_failed"
    
    # System
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class ActivityStatus(str, Enum):
    """Status for interactive activities (suggestions)."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"  # If suggestion becomes stale


class SuggestionData(BaseModel):
    """Data for relationship suggestions from LLM analysis."""
    from_node_id: str
    from_node_type: str
    from_node_label: str  # Display name (e.g., node title or first 50 chars)
    to_node_id: str
    to_node_type: str
    to_node_label: str
    relationship_type: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: Optional[str] = None  # LLM's explanation for the suggestion


class ProcessingData(BaseModel):
    """Data for processing status activities."""
    node_id: str
    node_type: str
    node_label: str
    stage: str  # "chunking", "embedding", "analyzing", "completed", "failed"
    progress: Optional[float] = Field(None, ge=0.0, le=1.0)  # 0-1 progress
    chunks_created: Optional[int] = None
    embeddings_created: Optional[int] = None
    suggestions_found: Optional[int] = None
    error_message: Optional[str] = None


class ActivityCreate(BaseModel):
    """Create a new activity."""
    type: ActivityType
    message: str
    
    # Optional references for navigation
    node_id: Optional[str] = None
    node_type: Optional[str] = None
    relationship_id: Optional[str] = None
    
    # Structured data for specific activity types
    suggestion_data: Optional[SuggestionData] = None
    processing_data: Optional[ProcessingData] = None
    
    # For suggestions
    status: Optional[ActivityStatus] = None
    
    # Who triggered this activity
    created_by: Optional[str] = None  # User ID or "system-llm"
    
    # Grouping (for updating processing status)
    group_id: Optional[str] = None  # Links related activities (e.g., all processing for one node)


class ActivityUpdate(BaseModel):
    """Update an existing activity (mainly for status changes)."""
    status: Optional[ActivityStatus] = None
    message: Optional[str] = None
    processing_data: Optional[ProcessingData] = None
    
    # For feedback loop
    user_feedback: Optional[str] = None  # Free-form feedback on rejected suggestions


class ActivityResponse(BaseModel):
    """Activity as returned by API."""
    id: str
    type: ActivityType
    message: str
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    # Navigation references
    node_id: Optional[str] = None
    node_type: Optional[str] = None
    relationship_id: Optional[str] = None
    
    # Structured data
    suggestion_data: Optional[SuggestionData] = None
    processing_data: Optional[ProcessingData] = None
    
    # Status
    status: Optional[ActivityStatus] = None
    
    # Metadata
    created_by: Optional[str] = None
    group_id: Optional[str] = None
    
    # Computed: should this activity show an action button?
    @property
    def is_interactive(self) -> bool:
        """Returns True if this activity should show action buttons."""
        return self.type in {
            ActivityType.RELATIONSHIP_SUGGESTED,
        } and self.status == ActivityStatus.PENDING
    
    @property
    def has_navigation(self) -> bool:
        """Returns True if this activity can navigate to a node/relationship."""
        return self.node_id is not None or self.relationship_id is not None


class ActivityFilter(BaseModel):
    """Filter criteria for querying activities."""
    types: Optional[List[ActivityType]] = None
    status: Optional[ActivityStatus] = None
    node_id: Optional[str] = None  # Activities related to a specific node
    group_id: Optional[str] = None
    created_by: Optional[str] = None
    since: Optional[datetime] = None
    limit: int = Field(default=50, le=200)
    include_dismissed: bool = False  # Include rejected/expired


# Confidence thresholds for relationship suggestions
class SuggestionThresholds(BaseModel):
    """Configurable thresholds for handling LLM suggestions."""
    auto_create_min: float = Field(default=0.8, ge=0.0, le=1.0)  # >= this: auto-create
    suggest_min: float = Field(default=0.6, ge=0.0, le=1.0)       # >= this: suggest to user
    # Below suggest_min: discard silently
    
    def get_action(self, confidence: float) -> str:
        """Determine action based on confidence score."""
        if confidence >= self.auto_create_min:
            return "auto_create"
        elif confidence >= self.suggest_min:
            return "suggest"
        else:
            return "discard"


# Default thresholds instance
DEFAULT_THRESHOLDS = SuggestionThresholds()

