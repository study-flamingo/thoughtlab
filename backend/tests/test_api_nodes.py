import pytest
from fastapi.testclient import TestClient


def test_root_endpoint(client: TestClient):
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "running"
    assert "version" in data


def test_health_endpoint(client: TestClient):
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "services" in data
    assert "neo4j" in data["services"]
    assert "redis" in data["services"]


def test_create_observation(client: TestClient, clean_neo4j):
    """Test creating an observation"""
    response = client.post(
        "/api/v1/nodes/observations",
        json={
            "text": "Test observation",
            "confidence": 0.8,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "id" in data
    assert data["message"] == "Observation created"


def test_create_observation_validation(client: TestClient):
    """Test observation validation"""
    # Empty text should fail
    response = client.post(
        "/api/v1/nodes/observations",
        json={"text": "", "confidence": 0.8},
    )
    assert response.status_code == 422

    # Invalid confidence should fail
    response = client.post(
        "/api/v1/nodes/observations",
        json={"text": "Test", "confidence": 1.5},
    )
    assert response.status_code == 422


def test_get_observation(client: TestClient, clean_neo4j):
    """Test getting an observation"""
    # Create first
    create_response = client.post(
        "/api/v1/nodes/observations",
        json={"text": "Test observation", "confidence": 0.8},
    )
    node_id = create_response.json()["id"]

    # Get it
    response = client.get(f"/api/v1/nodes/observations/{node_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == node_id
    assert data["text"] == "Test observation"
    assert data["confidence"] == 0.8


def test_get_nonexistent_observation(client: TestClient):
    """Test getting non-existent observation"""
    response = client.get("/api/v1/nodes/observations/nonexistent-id")
    assert response.status_code == 404


def test_get_all_observations(client: TestClient, clean_neo4j):
    """Test getting all observations"""
    # Create a few observations
    for i in range(3):
        client.post(
            "/api/v1/nodes/observations",
            json={"text": f"Observation {i}", "confidence": 0.8},
        )

    # Get all
    response = client.get("/api/v1/nodes/observations")
    assert response.status_code == 200
    data = response.json()
    assert "nodes" in data
    assert len(data["nodes"]) == 3


def test_create_relationship(client: TestClient, clean_neo4j):
    """Test creating a relationship"""
    # Create two nodes
    obs1 = client.post(
        "/api/v1/nodes/observations",
        json={"text": "Observation 1", "confidence": 0.8},
    ).json()["id"]

    obs2 = client.post(
        "/api/v1/nodes/observations",
        json={"text": "Observation 2", "confidence": 0.8},
    ).json()["id"]

    # Create relationship
    response = client.post(
        "/api/v1/nodes/relationships",
        json={
            "from_id": obs1,
            "to_id": obs2,
            "relationship_type": "RELATES_TO",
            "confidence": 0.9,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "Relationship created"


def test_get_connections(client: TestClient, clean_neo4j):
    """Test getting node connections"""
    # Create nodes and relationship
    obs1 = client.post(
        "/api/v1/nodes/observations",
        json={"text": "Observation 1", "confidence": 0.8},
    ).json()["id"]

    obs2 = client.post(
        "/api/v1/nodes/observations",
        json={"text": "Observation 2", "confidence": 0.8},
    ).json()["id"]

    client.post(
        "/api/v1/nodes/relationships",
        json={
            "from_id": obs1,
            "to_id": obs2,
            "relationship_type": "RELATES_TO",
        },
    )

    # Get connections
    response = client.get(f"/api/v1/nodes/{obs1}/connections")
    assert response.status_code == 200
    data = response.json()
    assert data["node_id"] == obs1
    assert "connections" in data
    assert len(data["connections"]) > 0


def test_get_full_graph(client: TestClient, clean_neo4j):
    """Test getting full graph"""
    # Create some nodes
    for i in range(3):
        client.post(
            "/api/v1/nodes/observations",
            json={"text": f"Observation {i}", "confidence": 0.8},
        )

    response = client.get("/api/v1/graph/full")
    assert response.status_code == 200
    data = response.json()
    assert "nodes" in data
    assert "edges" in data
    assert len(data["nodes"]) == 3
