"""Tests for ActivityService."""

import pytest
from datetime import datetime, UTC, timedelta
from app.services.activity_service import activity_service
from app.models.activity import (
    ActivityType,
    ActivityStatus,
    ActivityCreate,
    ActivityUpdate,
    ActivityFilter,
    SuggestionData,
    ProcessingData,
)


@pytest.mark.asyncio
async def test_create_activity(clean_neo4j):
    """Test creating a basic activity."""
    data = ActivityCreate(
        type=ActivityType.NODE_CREATED,
        message="Test node created",
        node_id="test-node-1",
        node_type="Observation",
    )
    activity_id = await activity_service.create(data)

    assert activity_id is not None
    assert isinstance(activity_id, str)


@pytest.mark.asyncio
async def test_get_activity(clean_neo4j):
    """Test retrieving an activity by ID."""
    # Create an activity
    data = ActivityCreate(
        type=ActivityType.NODE_CREATED,
        message="Test node created",
        node_id="test-node-1",
        node_type="Observation",
    )
    activity_id = await activity_service.create(data)

    # Retrieve it
    activity = await activity_service.get(activity_id)

    assert activity is not None
    assert activity.id == activity_id
    assert activity.type == ActivityType.NODE_CREATED
    assert activity.message == "Test node created"
    assert activity.node_id == "test-node-1"


@pytest.mark.asyncio
async def test_get_nonexistent_activity(clean_neo4j):
    """Test getting a non-existent activity returns None."""
    activity = await activity_service.get("nonexistent-id")
    assert activity is None


@pytest.mark.asyncio
async def test_list_activities(clean_neo4j):
    """Test listing activities."""
    # Create multiple activities
    for i in range(3):
        await activity_service.create(
            ActivityCreate(
                type=ActivityType.NODE_CREATED,
                message=f"Node {i} created",
                node_id=f"node-{i}",
            )
        )

    # List all
    activities = await activity_service.list()

    assert len(activities) == 3


@pytest.mark.asyncio
async def test_list_activities_with_type_filter(clean_neo4j):
    """Test filtering activities by type."""
    # Create activities of different types
    await activity_service.create(
        ActivityCreate(type=ActivityType.NODE_CREATED, message="Node created")
    )
    await activity_service.create(
        ActivityCreate(type=ActivityType.RELATIONSHIP_CREATED, message="Rel created")
    )
    await activity_service.create(
        ActivityCreate(type=ActivityType.NODE_CREATED, message="Another node")
    )

    # Filter by type
    filter = ActivityFilter(types=[ActivityType.NODE_CREATED])
    activities = await activity_service.list(filter)

    assert len(activities) == 2
    assert all(a.type == ActivityType.NODE_CREATED for a in activities)


@pytest.mark.asyncio
async def test_list_activities_with_status_filter(clean_neo4j):
    """Test filtering activities by status."""
    # Create a pending suggestion
    suggestion = SuggestionData(
        from_node_id="node-1",
        from_node_type="Observation",
        from_node_label="First",
        to_node_id="node-2",
        to_node_type="Hypothesis",
        to_node_label="Second",
        relationship_type="SUPPORTS",
        confidence=0.8,
    )
    await activity_service.create(
        ActivityCreate(
            type=ActivityType.RELATIONSHIP_SUGGESTED,
            message="Suggestion 1",
            suggestion_data=suggestion,
            status=ActivityStatus.PENDING,
        )
    )

    # Create a regular activity (no status)
    await activity_service.create(
        ActivityCreate(type=ActivityType.NODE_CREATED, message="Node created")
    )

    # Filter by pending status
    filter = ActivityFilter(status=ActivityStatus.PENDING)
    activities = await activity_service.list(filter)

    assert len(activities) == 1
    assert activities[0].status == ActivityStatus.PENDING


@pytest.mark.asyncio
async def test_list_activities_with_node_filter(clean_neo4j):
    """Test filtering activities by node_id."""
    # Create activities for different nodes
    await activity_service.create(
        ActivityCreate(
            type=ActivityType.NODE_CREATED,
            message="Node 1 created",
            node_id="node-1",
        )
    )
    await activity_service.create(
        ActivityCreate(
            type=ActivityType.NODE_UPDATED,
            message="Node 1 updated",
            node_id="node-1",
        )
    )
    await activity_service.create(
        ActivityCreate(
            type=ActivityType.NODE_CREATED,
            message="Node 2 created",
            node_id="node-2",
        )
    )

    # Filter by node_id
    filter = ActivityFilter(node_id="node-1")
    activities = await activity_service.list(filter)

    assert len(activities) == 2
    assert all(a.node_id == "node-1" for a in activities)


@pytest.mark.asyncio
async def test_list_activities_excludes_dismissed_by_default(clean_neo4j):
    """Test that rejected/expired activities are excluded by default."""
    # Create a rejected suggestion
    suggestion = SuggestionData(
        from_node_id="node-1",
        from_node_type="Observation",
        from_node_label="First",
        to_node_id="node-2",
        to_node_type="Hypothesis",
        to_node_label="Second",
        relationship_type="SUPPORTS",
        confidence=0.8,
    )
    activity_id = await activity_service.create(
        ActivityCreate(
            type=ActivityType.RELATIONSHIP_SUGGESTED,
            message="Rejected suggestion",
            suggestion_data=suggestion,
            status=ActivityStatus.PENDING,
        )
    )
    # Mark as rejected
    await activity_service.update(
        activity_id, ActivityUpdate(status=ActivityStatus.REJECTED)
    )

    # Create a normal activity
    await activity_service.create(
        ActivityCreate(type=ActivityType.NODE_CREATED, message="Node created")
    )

    # Default list should exclude rejected
    activities = await activity_service.list()
    assert len(activities) == 1

    # Include dismissed should show all
    filter = ActivityFilter(include_dismissed=True)
    all_activities = await activity_service.list(filter)
    assert len(all_activities) == 2


@pytest.mark.asyncio
async def test_update_activity_status(clean_neo4j):
    """Test updating activity status."""
    # Create a pending suggestion
    suggestion = SuggestionData(
        from_node_id="node-1",
        from_node_type="Observation",
        from_node_label="First",
        to_node_id="node-2",
        to_node_type="Hypothesis",
        to_node_label="Second",
        relationship_type="SUPPORTS",
        confidence=0.8,
    )
    activity_id = await activity_service.create(
        ActivityCreate(
            type=ActivityType.RELATIONSHIP_SUGGESTED,
            message="Suggestion",
            suggestion_data=suggestion,
            status=ActivityStatus.PENDING,
        )
    )

    # Update status
    success = await activity_service.update(
        activity_id, ActivityUpdate(status=ActivityStatus.APPROVED)
    )

    assert success is True

    # Verify update
    activity = await activity_service.get(activity_id)
    assert activity.status == ActivityStatus.APPROVED
    assert activity.updated_at is not None


@pytest.mark.asyncio
async def test_update_activity_message(clean_neo4j):
    """Test updating activity message."""
    activity_id = await activity_service.create(
        ActivityCreate(
            type=ActivityType.PROCESSING_STARTED,
            message="Processing started",
        )
    )

    success = await activity_service.update(
        activity_id, ActivityUpdate(message="Processing 50% complete")
    )

    assert success is True

    activity = await activity_service.get(activity_id)
    assert activity.message == "Processing 50% complete"


@pytest.mark.asyncio
async def test_delete_activity(clean_neo4j):
    """Test deleting an activity."""
    activity_id = await activity_service.create(
        ActivityCreate(type=ActivityType.INFO, message="Test activity")
    )

    # Verify it exists
    activity = await activity_service.get(activity_id)
    assert activity is not None

    # Delete it
    success = await activity_service.delete(activity_id)
    assert success is True

    # Verify it's gone
    activity = await activity_service.get(activity_id)
    assert activity is None


@pytest.mark.asyncio
async def test_delete_nonexistent_activity(clean_neo4j):
    """Test deleting non-existent activity returns False."""
    success = await activity_service.delete("nonexistent-id")
    assert success is False


@pytest.mark.asyncio
async def test_get_pending_suggestions(clean_neo4j):
    """Test getting pending suggestions."""
    # Create some suggestions
    suggestion = SuggestionData(
        from_node_id="node-1",
        from_node_type="Observation",
        from_node_label="First",
        to_node_id="node-2",
        to_node_type="Hypothesis",
        to_node_label="Second",
        relationship_type="SUPPORTS",
        confidence=0.8,
    )

    # Pending suggestions
    await activity_service.create(
        ActivityCreate(
            type=ActivityType.RELATIONSHIP_SUGGESTED,
            message="Suggestion 1",
            suggestion_data=suggestion,
            status=ActivityStatus.PENDING,
        )
    )
    await activity_service.create(
        ActivityCreate(
            type=ActivityType.RELATIONSHIP_SUGGESTED,
            message="Suggestion 2",
            suggestion_data=suggestion,
            status=ActivityStatus.PENDING,
        )
    )

    # Approved suggestion (shouldn't appear)
    activity_id = await activity_service.create(
        ActivityCreate(
            type=ActivityType.RELATIONSHIP_SUGGESTED,
            message="Approved suggestion",
            suggestion_data=suggestion,
            status=ActivityStatus.PENDING,
        )
    )
    await activity_service.update(
        activity_id, ActivityUpdate(status=ActivityStatus.APPROVED)
    )

    # Get pending
    pending = await activity_service.get_pending_suggestions()

    assert len(pending) == 2
    assert all(s.status == ActivityStatus.PENDING for s in pending)


@pytest.mark.asyncio
async def test_create_node_activity(clean_neo4j):
    """Test convenience method for creating node activities."""
    activity_id = await activity_service.create_node_activity(
        activity_type=ActivityType.NODE_CREATED,
        node_id="test-node",
        node_type="Observation",
        message="Observation created",
        created_by="user-123",
    )

    activity = await activity_service.get(activity_id)

    assert activity is not None
    assert activity.type == ActivityType.NODE_CREATED
    assert activity.node_id == "test-node"
    assert activity.node_type == "Observation"
    assert activity.created_by == "user-123"


@pytest.mark.asyncio
async def test_create_suggestion(clean_neo4j):
    """Test creating a relationship suggestion."""
    suggestion_data = SuggestionData(
        from_node_id="node-1",
        from_node_type="Observation",
        from_node_label="My observation about AI",
        to_node_id="node-2",
        to_node_type="Hypothesis",
        to_node_label="AI will change everything",
        relationship_type="SUPPORTS",
        confidence=0.75,
        reasoning="Both discuss AI impact",
    )

    activity_id = await activity_service.create_suggestion(suggestion_data)

    activity = await activity_service.get(activity_id)

    assert activity is not None
    assert activity.type == ActivityType.RELATIONSHIP_SUGGESTED
    assert activity.status == ActivityStatus.PENDING
    assert activity.suggestion_data is not None
    assert activity.suggestion_data.from_node_id == "node-1"
    assert activity.suggestion_data.confidence == 0.75
    assert "SUPPORTS" in activity.message
    assert "75%" in activity.message


@pytest.mark.asyncio
async def test_approve_suggestion(clean_neo4j):
    """Test approving a suggestion returns suggestion data."""
    suggestion_data = SuggestionData(
        from_node_id="node-1",
        from_node_type="Observation",
        from_node_label="First",
        to_node_id="node-2",
        to_node_type="Hypothesis",
        to_node_label="Second",
        relationship_type="SUPPORTS",
        confidence=0.8,
    )
    activity_id = await activity_service.create_suggestion(suggestion_data)

    # Approve it
    result = await activity_service.approve_suggestion(activity_id)

    assert result is not None
    assert result["from_node_id"] == "node-1"
    assert result["to_node_id"] == "node-2"
    assert result["relationship_type"] == "SUPPORTS"

    # Verify activity is now approved
    activity = await activity_service.get(activity_id)
    assert activity.status == ActivityStatus.APPROVED


@pytest.mark.asyncio
async def test_approve_nonexistent_suggestion(clean_neo4j):
    """Test approving non-existent suggestion returns None."""
    result = await activity_service.approve_suggestion("nonexistent-id")
    assert result is None


@pytest.mark.asyncio
async def test_approve_already_processed_suggestion(clean_neo4j):
    """Test approving already approved suggestion returns None."""
    suggestion_data = SuggestionData(
        from_node_id="node-1",
        from_node_type="Observation",
        from_node_label="First",
        to_node_id="node-2",
        to_node_type="Hypothesis",
        to_node_label="Second",
        relationship_type="SUPPORTS",
        confidence=0.8,
    )
    activity_id = await activity_service.create_suggestion(suggestion_data)

    # Approve once
    await activity_service.approve_suggestion(activity_id)

    # Try to approve again
    result = await activity_service.approve_suggestion(activity_id)
    assert result is None


@pytest.mark.asyncio
async def test_reject_suggestion(clean_neo4j):
    """Test rejecting a suggestion."""
    suggestion_data = SuggestionData(
        from_node_id="node-1",
        from_node_type="Observation",
        from_node_label="First",
        to_node_id="node-2",
        to_node_type="Hypothesis",
        to_node_label="Second",
        relationship_type="SUPPORTS",
        confidence=0.8,
    )
    activity_id = await activity_service.create_suggestion(suggestion_data)

    # Reject with feedback
    success = await activity_service.reject_suggestion(
        activity_id, "These nodes aren't actually related"
    )

    assert success is True

    # Verify activity is rejected
    filter = ActivityFilter(include_dismissed=True)
    activities = await activity_service.list(filter)
    activity = next((a for a in activities if a.id == activity_id), None)

    assert activity is not None
    assert activity.status == ActivityStatus.REJECTED


@pytest.mark.asyncio
async def test_reject_nonexistent_suggestion(clean_neo4j):
    """Test rejecting non-existent suggestion returns False."""
    success = await activity_service.reject_suggestion("nonexistent-id")
    assert success is False


@pytest.mark.asyncio
async def test_update_processing_status_creates_new(clean_neo4j):
    """Test update_processing_status creates new activity when no existing."""
    processing_data = ProcessingData(
        node_id="source-1",
        node_type="Source",
        node_label="Research Paper",
        stage="started",
    )

    activity_id = await activity_service.update_processing_status(
        group_id="processing-123",
        stage="started",
        message="Starting to process Research Paper",
        processing_data=processing_data,
    )

    activity = await activity_service.get(activity_id)

    assert activity is not None
    assert activity.type == ActivityType.PROCESSING_STARTED
    assert activity.group_id == "processing-123"


@pytest.mark.asyncio
async def test_update_processing_status_updates_existing(clean_neo4j):
    """Test update_processing_status updates existing activity."""
    group_id = "processing-456"

    # Create initial processing activity
    processing_data = ProcessingData(
        node_id="source-1",
        node_type="Source",
        node_label="Research Paper",
        stage="started",
    )
    activity_id = await activity_service.update_processing_status(
        group_id=group_id,
        stage="started",
        message="Starting processing",
        processing_data=processing_data,
    )

    # Update to chunking stage
    updated_data = ProcessingData(
        node_id="source-1",
        node_type="Source",
        node_label="Research Paper",
        stage="chunking",
        chunks_created=5,
    )
    updated_id = await activity_service.update_processing_status(
        group_id=group_id,
        stage="chunking",
        message="Chunking: 5 chunks created",
        processing_data=updated_data,
    )

    # Should return same activity ID
    assert updated_id == activity_id

    # Verify update
    activity = await activity_service.get(activity_id)
    assert activity.message == "Chunking: 5 chunks created"
    assert activity.type == ActivityType.PROCESSING_CHUNKING


@pytest.mark.asyncio
async def test_get_processing_status_for_node(clean_neo4j):
    """Test getting latest processing status for a node."""
    processing_data = ProcessingData(
        node_id="source-1",
        node_type="Source",
        node_label="Research Paper",
        stage="embedding",
        chunks_created=10,
        embeddings_created=5,
    )

    await activity_service.update_processing_status(
        group_id="processing-789",
        stage="embedding",
        message="Creating embeddings",
        processing_data=processing_data,
    )

    # Get processing status
    status = await activity_service.get_processing_status("source-1")

    assert status is not None
    assert status.processing_data is not None
    assert status.processing_data.node_id == "source-1"
    assert status.processing_data.stage == "embedding"


@pytest.mark.asyncio
async def test_get_processing_status_nonexistent_node(clean_neo4j):
    """Test getting processing status for node with no processing returns None."""
    status = await activity_service.get_processing_status("nonexistent-node")
    assert status is None


@pytest.mark.asyncio
async def test_activity_ordering(clean_neo4j):
    """Test activities are returned in descending order by created_at."""
    # Create activities with small delays (implicit through sequential creation)
    for i in range(3):
        await activity_service.create(
            ActivityCreate(
                type=ActivityType.NODE_CREATED,
                message=f"Node {i}",
            )
        )

    activities = await activity_service.list()

    # Most recent first
    assert activities[0].message == "Node 2"
    assert activities[1].message == "Node 1"
    assert activities[2].message == "Node 0"


@pytest.mark.asyncio
async def test_activity_limit(clean_neo4j):
    """Test limit parameter in activity list."""
    # Create 5 activities
    for i in range(5):
        await activity_service.create(
            ActivityCreate(
                type=ActivityType.NODE_CREATED,
                message=f"Node {i}",
            )
        )

    # Request only 3
    filter = ActivityFilter(limit=3)
    activities = await activity_service.list(filter)

    assert len(activities) == 3

