"""Tests for Activity Feed API routes."""

import pytest
from fastapi.testclient import TestClient
from app.services.activity_service import activity_service
from app.models.activity import (
    ActivityType,
    ActivityStatus,
    ActivityCreate,
    SuggestionData,
    ProcessingData,
)


def test_list_activities_empty(client: TestClient, clean_neo4j):
    """Test listing activities when empty."""
    response = client.get("/api/v1/activities")
    assert response.status_code == 200
    data = response.json()
    assert data == []


def test_list_activities(client: TestClient, clean_neo4j):
    """Test listing activities."""
    # Create some activities first via the test client
    # Since we can't easily call async service from sync test, 
    # we'll create them via the graph endpoints that generate activities
    
    # Create an observation (should generate node_created activity)
    client.post(
        "/api/v1/nodes/observations",
        json={"text": "Test observation", "confidence": 0.8},
    )

    response = client.get("/api/v1/activities")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


def test_list_activities_with_type_filter(client: TestClient, clean_neo4j):
    """Test filtering activities by type."""
    response = client.get(
        "/api/v1/activities",
        params={"types": "node_created,relationship_created"},
    )
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


def test_list_activities_with_invalid_type(client: TestClient, clean_neo4j):
    """Test filtering with invalid type returns 400."""
    response = client.get(
        "/api/v1/activities",
        params={"types": "invalid_type"},
    )
    assert response.status_code == 400
    assert "Invalid activity type" in response.json()["detail"]


def test_list_activities_with_status_filter(client: TestClient, clean_neo4j):
    """Test filtering activities by status."""
    response = client.get(
        "/api/v1/activities",
        params={"status": "pending"},
    )
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


def test_list_activities_with_node_filter(client: TestClient, clean_neo4j):
    """Test filtering activities by node_id."""
    response = client.get(
        "/api/v1/activities",
        params={"node_id": "some-node-id"},
    )
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


def test_list_activities_with_limit(client: TestClient, clean_neo4j):
    """Test limiting activity results."""
    response = client.get(
        "/api/v1/activities",
        params={"limit": 10},
    )
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) <= 10


def test_list_activities_with_include_dismissed(client: TestClient, clean_neo4j):
    """Test including dismissed activities."""
    response = client.get(
        "/api/v1/activities",
        params={"include_dismissed": True},
    )
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


def test_get_pending_suggestions_empty(client: TestClient, clean_neo4j):
    """Test getting pending suggestions when empty."""
    response = client.get("/api/v1/activities/pending")
    assert response.status_code == 200
    data = response.json()
    assert data == []


def test_get_pending_suggestions_with_limit(client: TestClient, clean_neo4j):
    """Test getting pending suggestions with limit."""
    response = client.get(
        "/api/v1/activities/pending",
        params={"limit": 5},
    )
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) <= 5


def test_get_processing_status_not_found(client: TestClient, clean_neo4j):
    """Test getting processing status for node without processing."""
    response = client.get("/api/v1/activities/processing/nonexistent-node")
    assert response.status_code == 200
    # Returns null when no processing found
    assert response.json() is None


def test_get_activity_not_found(client: TestClient, clean_neo4j):
    """Test getting non-existent activity returns 404."""
    response = client.get("/api/v1/activities/nonexistent-id")
    assert response.status_code == 404
    assert "Activity not found" in response.json()["detail"]


def test_approve_suggestion_not_found(client: TestClient, clean_neo4j):
    """Test approving non-existent suggestion returns 400."""
    response = client.post("/api/v1/activities/nonexistent-id/approve")
    assert response.status_code == 400
    assert "Cannot approve" in response.json()["detail"]


def test_reject_suggestion_not_found(client: TestClient, clean_neo4j):
    """Test rejecting non-existent suggestion returns 400."""
    response = client.post("/api/v1/activities/nonexistent-id/reject")
    assert response.status_code == 400
    assert "Cannot reject" in response.json()["detail"]


def test_delete_activity_not_found(client: TestClient, clean_neo4j):
    """Test deleting non-existent activity returns 404."""
    response = client.delete("/api/v1/activities/nonexistent-id")
    assert response.status_code == 404


# Integration tests using async fixtures


@pytest.mark.asyncio
async def test_list_activities_integration(client: TestClient, clean_neo4j):
    """Integration test for listing activities."""
    # Create activities directly via service
    await activity_service.create(
        ActivityCreate(
            type=ActivityType.NODE_CREATED,
            message="Test node created",
            node_id="test-node-1",
        )
    )
    await activity_service.create(
        ActivityCreate(
            type=ActivityType.RELATIONSHIP_CREATED,
            message="Test relationship created",
        )
    )

    response = client.get("/api/v1/activities")
    assert response.status_code == 200
    data = response.json()
    assert len(data) >= 2


@pytest.mark.asyncio
async def test_get_activity_integration(client: TestClient, clean_neo4j):
    """Integration test for getting a single activity."""
    activity_id = await activity_service.create(
        ActivityCreate(
            type=ActivityType.NODE_CREATED,
            message="Test node created",
            node_id="test-node-1",
            node_type="Observation",
        )
    )

    response = client.get(f"/api/v1/activities/{activity_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == activity_id
    assert data["type"] == "node_created"
    assert data["message"] == "Test node created"
    assert data["node_id"] == "test-node-1"


@pytest.mark.asyncio
async def test_get_pending_suggestions_integration(client: TestClient, clean_neo4j):
    """Integration test for getting pending suggestions."""
    suggestion = SuggestionData(
        from_node_id="node-1",
        from_node_type="Observation",
        from_node_label="First observation",
        to_node_id="node-2",
        to_node_type="Hypothesis",
        to_node_label="Test hypothesis",
        relationship_type="SUPPORTS",
        confidence=0.75,
    )

    await activity_service.create(
        ActivityCreate(
            type=ActivityType.RELATIONSHIP_SUGGESTED,
            message="Suggested: First observation SUPPORTS Test hypothesis",
            suggestion_data=suggestion,
            status=ActivityStatus.PENDING,
        )
    )

    response = client.get("/api/v1/activities/pending")
    assert response.status_code == 200
    data = response.json()
    assert len(data) >= 1
    assert data[0]["type"] == "relationship_suggested"
    assert data[0]["status"] == "pending"


@pytest.mark.asyncio
async def test_approve_suggestion_integration(client: TestClient, clean_neo4j):
    """Integration test for approving a suggestion."""
    # First create the nodes that the suggestion references
    obs1 = client.post(
        "/api/v1/nodes/observations",
        json={"text": "Observation 1", "confidence": 0.8},
    ).json()["id"]
    
    obs2 = client.post(
        "/api/v1/nodes/observations",
        json={"text": "Observation 2", "confidence": 0.8},
    ).json()["id"]

    # Create a suggestion
    suggestion = SuggestionData(
        from_node_id=obs1,
        from_node_type="Observation",
        from_node_label="Observation 1",
        to_node_id=obs2,
        to_node_type="Observation",
        to_node_label="Observation 2",
        relationship_type="RELATES_TO",
        confidence=0.75,
        reasoning="These observations are related",
    )

    activity_id = await activity_service.create(
        ActivityCreate(
            type=ActivityType.RELATIONSHIP_SUGGESTED,
            message="Suggested relationship",
            suggestion_data=suggestion,
            status=ActivityStatus.PENDING,
        )
    )

    # Approve the suggestion
    response = client.post(f"/api/v1/activities/{activity_id}/approve")
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "Suggestion approved"
    assert "relationship_id" in data
    assert data["activity_id"] == activity_id


@pytest.mark.asyncio
async def test_reject_suggestion_integration(client: TestClient, clean_neo4j):
    """Integration test for rejecting a suggestion."""
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

    # Reject with feedback
    response = client.post(
        f"/api/v1/activities/{activity_id}/reject",
        params={"feedback": "These nodes are not related"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "Suggestion rejected"
    assert data["feedback_stored"] is True


@pytest.mark.asyncio
async def test_reject_suggestion_without_feedback(client: TestClient, clean_neo4j):
    """Integration test for rejecting a suggestion without feedback."""
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

    # Reject without feedback
    response = client.post(f"/api/v1/activities/{activity_id}/reject")
    assert response.status_code == 200
    data = response.json()
    assert data["feedback_stored"] is False


@pytest.mark.asyncio
async def test_delete_activity_integration(client: TestClient, clean_neo4j):
    """Integration test for deleting an activity."""
    activity_id = await activity_service.create(
        ActivityCreate(
            type=ActivityType.INFO,
            message="Test activity to delete",
        )
    )

    # Delete it
    response = client.delete(f"/api/v1/activities/{activity_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "Activity deleted"
    assert data["id"] == activity_id

    # Verify it's gone
    response = client.get(f"/api/v1/activities/{activity_id}")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_get_processing_status_integration(client: TestClient, clean_neo4j):
    """Integration test for getting processing status."""
    processing_data = ProcessingData(
        node_id="source-1",
        node_type="Source",
        node_label="Research Paper",
        stage="embedding",
        chunks_created=10,
        embeddings_created=5,
    )

    await activity_service.update_processing_status(
        group_id="processing-test",
        stage="embedding",
        message="Creating embeddings: 5/10",
        processing_data=processing_data,
    )

    response = client.get("/api/v1/activities/processing/source-1")
    assert response.status_code == 200
    data = response.json()
    assert data is not None
    assert data["type"] == "processing_embedding"
    assert data["processing_data"]["node_id"] == "source-1"
    assert data["processing_data"]["stage"] == "embedding"
    assert data["processing_data"]["chunks_created"] == 10


@pytest.mark.asyncio
async def test_activity_type_filter_integration(client: TestClient, clean_neo4j):
    """Integration test for filtering by activity type."""
    # Create activities of different types
    await activity_service.create(
        ActivityCreate(type=ActivityType.NODE_CREATED, message="Node 1")
    )
    await activity_service.create(
        ActivityCreate(type=ActivityType.NODE_UPDATED, message="Node updated")
    )
    await activity_service.create(
        ActivityCreate(type=ActivityType.RELATIONSHIP_CREATED, message="Rel 1")
    )

    # Filter by node_created only
    response = client.get(
        "/api/v1/activities",
        params={"types": "node_created"},
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["type"] == "node_created"

    # Filter by multiple types
    response = client.get(
        "/api/v1/activities",
        params={"types": "node_created,node_updated"},
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    types = {a["type"] for a in data}
    assert types == {"node_created", "node_updated"}

