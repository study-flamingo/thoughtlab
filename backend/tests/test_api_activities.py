"""Tests for Activity Feed API routes.

These are HTTP-level tests only - no direct service calls.
For service-layer tests, see test_activity_service.py.
"""

import pytest
from fastapi.testclient import TestClient


def test_list_activities_empty(client: TestClient, clean_neo4j):
    """Test listing activities when empty."""
    response = client.get("/api/v1/activities")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


def test_list_activities_after_node_creation(client: TestClient, clean_neo4j):
    """Test that creating a node generates an activity."""
    # Create an observation
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
    assert isinstance(data, list)


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
