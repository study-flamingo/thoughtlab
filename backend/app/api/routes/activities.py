"""Activity Feed API routes.

Endpoints for managing the activity feed:
- GET /activities - List activities with filtering
- GET /activities/{id} - Get single activity
- POST /activities/{id}/approve - Approve a suggestion
- POST /activities/{id}/reject - Reject a suggestion
- GET /activities/pending - Get pending suggestions
- GET /activities/processing/{node_id} - Get processing status for a node
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional, List
from datetime import datetime

from app.services.activity_service import activity_service
from app.services.graph_service import graph_service
from app.models.activity import (
    ActivityType,
    ActivityStatus,
    ActivityResponse,
    ActivityFilter,
)

router = APIRouter(prefix="/activities", tags=["activities"])


@router.get("", response_model=List[ActivityResponse])
async def list_activities(
    types: Optional[str] = Query(None, description="Comma-separated activity types"),
    status: Optional[ActivityStatus] = None,
    node_id: Optional[str] = None,
    group_id: Optional[str] = None,
    since: Optional[datetime] = None,
    limit: int = Query(50, le=200),
    include_dismissed: bool = False,
):
    """List activities with optional filtering.
    
    Use `types` to filter by activity type (comma-separated).
    Use `status` to filter by status (pending, approved, rejected).
    Use `node_id` to get activities related to a specific node.
    Use `since` to get activities after a certain time.
    """
    # Parse types if provided
    type_list = None
    if types:
        try:
            type_list = [ActivityType(t.strip()) for t in types.split(",")]
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Invalid activity type: {e}")
    
    filter = ActivityFilter(
        types=type_list,
        status=status,
        node_id=node_id,
        group_id=group_id,
        since=since,
        limit=limit,
        include_dismissed=include_dismissed,
    )
    
    return await activity_service.list(filter)


@router.get("/pending", response_model=List[ActivityResponse])
async def get_pending_suggestions(limit: int = Query(20, le=100)):
    """Get pending relationship suggestions awaiting user review."""
    return await activity_service.get_pending_suggestions(limit=limit)


@router.get("/processing/{node_id}", response_model=Optional[ActivityResponse])
async def get_processing_status(node_id: str):
    """Get the latest processing status for a node."""
    return await activity_service.get_processing_status(node_id)


@router.get("/{activity_id}", response_model=ActivityResponse)
async def get_activity(activity_id: str):
    """Get a single activity by ID."""
    activity = await activity_service.get(activity_id)
    if not activity:
        raise HTTPException(status_code=404, detail="Activity not found")
    return activity


@router.post("/{activity_id}/approve")
async def approve_suggestion(activity_id: str):
    """Approve a relationship suggestion.
    
    This will:
    1. Mark the suggestion as approved
    2. Create the suggested relationship
    3. Return the created relationship ID
    """
    # Get suggestion data
    suggestion_data = await activity_service.approve_suggestion(activity_id)
    if not suggestion_data:
        raise HTTPException(
            status_code=400,
            detail="Cannot approve: activity not found, not a suggestion, or already processed"
        )
    
    # Create the relationship
    rel_id = await graph_service.create_relationship(
        from_id=suggestion_data["from_node_id"],
        to_id=suggestion_data["to_node_id"],
        rel_type=suggestion_data["relationship_type"],
        properties={
            "confidence": suggestion_data.get("confidence"),
            "notes": suggestion_data.get("reasoning"),
        },
        created_by="system-llm",  # Mark as LLM-created but user-approved
    )
    
    if not rel_id:
        raise HTTPException(
            status_code=500,
            detail="Failed to create relationship"
        )
    
    return {
        "message": "Suggestion approved",
        "relationship_id": rel_id,
        "activity_id": activity_id,
    }


@router.post("/{activity_id}/reject")
async def reject_suggestion(
    activity_id: str,
    feedback: Optional[str] = Query(None, description="Optional feedback on why this was rejected"),
):
    """Reject a relationship suggestion.
    
    Optionally provide feedback that will be stored for training data.
    """
    success = await activity_service.reject_suggestion(activity_id, feedback)
    if not success:
        raise HTTPException(
            status_code=400,
            detail="Cannot reject: activity not found, not a suggestion, or already processed"
        )
    
    return {
        "message": "Suggestion rejected",
        "activity_id": activity_id,
        "feedback_stored": feedback is not None,
    }


@router.delete("/{activity_id}")
async def delete_activity(activity_id: str):
    """Delete an activity (admin only in future)."""
    success = await activity_service.delete(activity_id)
    if not success:
        raise HTTPException(status_code=404, detail="Activity not found")
    return {"message": "Activity deleted", "id": activity_id}

