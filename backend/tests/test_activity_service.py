"""Tests for ActivityService.

These tests require Neo4j to be running and use async fixtures.
"""

import pytest
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
    data = ActivityCreate(
        type=ActivityType.NODE_CREATED,
        message="Test node created",
        node_id="test-node-1",
        node_type="Observation",
    )
    activity_id = await activity_service.create(data)

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
    created_ids = []
    for i in range(3):
        activity_id = await activity_service.create(
            ActivityCreate(
                type=ActivityType.NODE_CREATED,
                message=f"Node {i} created",
                node_id=f"node-{i}",
            )
        )
        created_ids.append(activity_id)

    # List all
    activities = await activity_service.list()

    # Should have at least the 3 we created
    assert len(activities) >= 3
    activity_ids = [a.id for a in activities]
    for created_id in created_ids:
        assert created_id in activity_ids


@pytest.mark.asyncio
async def test_list_activities_with_type_filter(clean_neo4j):
    """Test filtering activities by type."""
    # Create activities of different types
    node_id1 = await activity_service.create(
        ActivityCreate(type=ActivityType.NODE_CREATED, message="Node created")
    )
    await activity_service.create(
        ActivityCreate(type=ActivityType.RELATIONSHIP_CREATED, message="Rel created")
    )
    node_id2 = await activity_service.create(
        ActivityCreate(type=ActivityType.NODE_CREATED, message="Another node")
    )

    # Filter by type
    filter = ActivityFilter(types=[ActivityType.NODE_CREATED])
    activities = await activity_service.list(filter)

    activity_ids = [a.id for a in activities]
    assert node_id1 in activity_ids
    assert node_id2 in activity_ids
    assert all(a.type == ActivityType.NODE_CREATED for a in activities)


@pytest.mark.asyncio
async def test_list_activities_with_status_filter(clean_neo4j):
    """Test filtering activities by status."""
    # Create an activity with pending status
    pending_id = await activity_service.create(
        ActivityCreate(
            type=ActivityType.RELATIONSHIP_SUGGESTED,
            message="Suggested relationship",
            status=ActivityStatus.PENDING,
        )
    )
    # Create one with approved status
    await activity_service.create(
        ActivityCreate(
            type=ActivityType.RELATIONSHIP_SUGGESTED,
            message="Approved suggestion",
            status=ActivityStatus.APPROVED,
        )
    )

    # Filter by pending status
    filter = ActivityFilter(status=ActivityStatus.PENDING)
    activities = await activity_service.list(filter)

    activity_ids = [a.id for a in activities]
    assert pending_id in activity_ids
    assert all(a.status == ActivityStatus.PENDING for a in activities)


@pytest.mark.asyncio
async def test_list_activities_with_node_filter(clean_neo4j):
    """Test filtering activities by node_id."""
    target_node_id = "specific-node-123"
    
    # Create activity for specific node
    activity_id = await activity_service.create(
        ActivityCreate(
            type=ActivityType.NODE_CREATED,
            message="Node created",
            node_id=target_node_id,
        )
    )
    # Create activity for different node
    await activity_service.create(
        ActivityCreate(
            type=ActivityType.NODE_CREATED,
            message="Other node",
            node_id="other-node",
        )
    )

    # Filter by node_id
    filter = ActivityFilter(node_id=target_node_id)
    activities = await activity_service.list(filter)

    activity_ids = [a.id for a in activities]
    assert activity_id in activity_ids
    assert all(a.node_id == target_node_id for a in activities)


@pytest.mark.asyncio
async def test_update_activity_status(clean_neo4j):
    """Test updating activity status."""
    activity_id = await activity_service.create(
        ActivityCreate(
            type=ActivityType.RELATIONSHIP_SUGGESTED,
            message="Suggestion",
            status=ActivityStatus.PENDING,
        )
    )

    # Update status
    success = await activity_service.update(
        activity_id,
        ActivityUpdate(status=ActivityStatus.APPROVED),
    )

    assert success is True

    # Verify the update
    activity = await activity_service.get(activity_id)
    assert activity.status == ActivityStatus.APPROVED


@pytest.mark.asyncio
async def test_update_activity_message(clean_neo4j):
    """Test updating activity message."""
    activity_id = await activity_service.create(
        ActivityCreate(
            type=ActivityType.INFO,
            message="Original message",
        )
    )

    success = await activity_service.update(
        activity_id,
        ActivityUpdate(message="Updated message"),
    )

    assert success is True

    # Verify the update
    activity = await activity_service.get(activity_id)
    assert activity.message == "Updated message"


@pytest.mark.asyncio
async def test_delete_activity(clean_neo4j):
    """Test deleting an activity."""
    activity_id = await activity_service.create(
        ActivityCreate(
            type=ActivityType.INFO,
            message="To be deleted",
        )
    )

    result = await activity_service.delete(activity_id)
    assert result is True

    # Verify it's gone
    activity = await activity_service.get(activity_id)
    assert activity is None


@pytest.mark.asyncio
async def test_delete_nonexistent_activity(clean_neo4j):
    """Test deleting non-existent activity returns False."""
    result = await activity_service.delete("nonexistent-id")
    assert result is False


@pytest.mark.asyncio
async def test_get_pending_suggestions(clean_neo4j):
    """Test getting pending suggestions."""
    suggestion = SuggestionData(
        from_node_id="node-1",
        from_node_type="Observation",
        from_node_label="First",
        to_node_id="node-2",
        to_node_type="Hypothesis",
        to_node_label="Second",
        relationship_type="SUPPORTS",
        confidence=0.75,
    )

    suggestion_id = await activity_service.create(
        ActivityCreate(
            type=ActivityType.RELATIONSHIP_SUGGESTED,
            message="Suggested relationship",
            suggestion_data=suggestion,
            status=ActivityStatus.PENDING,
        )
    )

    # Get pending suggestions
    suggestions = await activity_service.get_pending_suggestions()

    suggestion_ids = [s.id for s in suggestions]
    assert suggestion_id in suggestion_ids


@pytest.mark.asyncio
async def test_create_node_activity(clean_neo4j):
    """Test the helper method for creating node activities."""
    activity_id = await activity_service.create_node_activity(
        activity_type=ActivityType.NODE_CREATED,
        node_id="test-node",
        node_type="Observation",
        message="Created new observation",
    )

    activity = await activity_service.get(activity_id)
    assert activity is not None
    assert activity.type == ActivityType.NODE_CREATED
    assert activity.node_id == "test-node"
    assert activity.message == "Created new observation"


@pytest.mark.asyncio
async def test_create_suggestion(clean_neo4j):
    """Test the helper method for creating suggestions."""
    suggestion_data = SuggestionData(
        from_node_id="node-1",
        from_node_type="Observation",
        from_node_label="First observation",
        to_node_id="node-2",
        to_node_type="Hypothesis",
        to_node_label="Test hypothesis",
        relationship_type="SUPPORTS",
        confidence=0.75,
        reasoning="These are related",
    )

    activity_id = await activity_service.create_suggestion(
        suggestion_data=suggestion_data,
        created_by="test",
    )

    activity = await activity_service.get(activity_id)
    assert activity is not None
    assert activity.type == ActivityType.RELATIONSHIP_SUGGESTED
    assert activity.status == ActivityStatus.PENDING
    assert activity.suggestion_data is not None
    assert activity.suggestion_data.confidence == 0.75


@pytest.mark.asyncio
async def test_approve_suggestion(clean_neo4j):
    """Test approving a suggestion."""
    suggestion = SuggestionData(
        from_node_id="node-1",
        from_node_type="Observation",
        from_node_label="First",
        to_node_id="node-2",
        to_node_type="Hypothesis",
        to_node_label="Second",
        relationship_type="SUPPORTS",
        confidence=0.75,
    )

    activity_id = await activity_service.create(
        ActivityCreate(
            type=ActivityType.RELATIONSHIP_SUGGESTED,
            message="Suggested relationship",
            suggestion_data=suggestion,
            status=ActivityStatus.PENDING,
        )
    )

    # approve_suggestion returns the suggestion data dict or None
    result = await activity_service.approve_suggestion(activity_id)
    assert result is not None
    assert result["from_node_id"] == "node-1"
    assert result["relationship_type"] == "SUPPORTS"

    # Check status was updated
    activity = await activity_service.get(activity_id)
    assert activity.status == ActivityStatus.APPROVED


@pytest.mark.asyncio
async def test_approve_nonexistent_suggestion(clean_neo4j):
    """Test approving non-existent suggestion returns None."""
    result = await activity_service.approve_suggestion("nonexistent-id")
    assert result is None


@pytest.mark.asyncio
async def test_approve_already_processed_suggestion(clean_neo4j):
    """Test approving already processed suggestion returns None."""
    suggestion = SuggestionData(
        from_node_id="node-1",
        from_node_type="Observation",
        from_node_label="First",
        to_node_id="node-2",
        to_node_type="Hypothesis",
        to_node_label="Second",
        relationship_type="SUPPORTS",
        confidence=0.75,
    )

    activity_id = await activity_service.create(
        ActivityCreate(
            type=ActivityType.RELATIONSHIP_SUGGESTED,
            message="Suggested relationship",
            suggestion_data=suggestion,
            status=ActivityStatus.APPROVED,  # Already approved
        )
    )

    result = await activity_service.approve_suggestion(activity_id)
    assert result is None


@pytest.mark.asyncio
async def test_reject_suggestion(clean_neo4j):
    """Test rejecting a suggestion."""
    suggestion = SuggestionData(
        from_node_id="node-1",
        from_node_type="Observation",
        from_node_label="First",
        to_node_id="node-2",
        to_node_type="Hypothesis",
        to_node_label="Second",
        relationship_type="SUPPORTS",
        confidence=0.75,
    )

    activity_id = await activity_service.create(
        ActivityCreate(
            type=ActivityType.RELATIONSHIP_SUGGESTED,
            message="Suggested relationship",
            suggestion_data=suggestion,
            status=ActivityStatus.PENDING,
        )
    )

    result = await activity_service.reject_suggestion(
        activity_id, feedback="Not related"
    )
    assert result is True

    # Check status was updated
    activity = await activity_service.get(activity_id)
    assert activity.status == ActivityStatus.REJECTED


@pytest.mark.asyncio
async def test_reject_nonexistent_suggestion(clean_neo4j):
    """Test rejecting non-existent suggestion returns False."""
    result = await activity_service.reject_suggestion("nonexistent-id")
    assert result is False


@pytest.mark.asyncio
async def test_get_processing_status_nonexistent_node(clean_neo4j):
    """Test getting processing status for node without processing."""
    status = await activity_service.get_processing_status("nonexistent-node")
    assert status is None
