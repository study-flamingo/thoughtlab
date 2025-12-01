import pytest
from app.services.graph_service import graph_service
from app.models.nodes import ObservationCreate


@pytest.mark.asyncio
async def test_create_observation(clean_neo4j):
    """Test creating an observation via service"""
    data = ObservationCreate(
        text="Test observation",
        confidence=0.8,
        concept_names=["test", "example"],
    )
    node_id = await graph_service.create_observation(data)
    assert node_id is not None
    assert isinstance(node_id, str)


@pytest.mark.asyncio
async def test_get_observation(clean_neo4j):
    """Test getting an observation via service"""
    # Create first
    data = ObservationCreate(text="Test observation", confidence=0.8)
    node_id = await graph_service.create_observation(data)

    # Get it
    observation = await graph_service.get_observation(node_id)
    assert observation is not None
    assert observation["id"] == node_id
    assert observation["text"] == "Test observation"
    assert observation["confidence"] == 0.8


@pytest.mark.asyncio
async def test_get_nonexistent_observation(clean_neo4j):
    """Test getting non-existent observation"""
    observation = await graph_service.get_observation("nonexistent-id")
    assert observation is None


@pytest.mark.asyncio
async def test_create_relationship(clean_neo4j):
    """Test creating a relationship via service"""
    # Create two nodes
    obs1_data = ObservationCreate(text="Observation 1", confidence=0.8)
    obs2_data = ObservationCreate(text="Observation 2", confidence=0.8)

    obs1_id = await graph_service.create_observation(obs1_data)
    obs2_id = await graph_service.create_observation(obs2_data)

    # Create relationship
    success = await graph_service.create_relationship(
        obs1_id,
        obs2_id,
        "RELATES_TO",
        {"confidence": 0.9, "notes": "Test relationship"},
    )
    assert success is True


@pytest.mark.asyncio
async def test_get_node_connections(clean_neo4j):
    """Test getting node connections"""
    # Create nodes
    obs1_data = ObservationCreate(text="Observation 1", confidence=0.8)
    obs2_data = ObservationCreate(text="Observation 2", confidence=0.8)
    obs3_data = ObservationCreate(text="Observation 3", confidence=0.8)

    obs1_id = await graph_service.create_observation(obs1_data)
    obs2_id = await graph_service.create_observation(obs2_data)
    obs3_id = await graph_service.create_observation(obs3_data)

    # Create relationships
    await graph_service.create_relationship(
        obs1_id, obs2_id, "RELATES_TO"
    )
    await graph_service.create_relationship(
        obs2_id, obs3_id, "SUPPORTS"
    )

    # Get connections
    connections = await graph_service.get_node_connections(obs1_id, max_depth=2)
    assert len(connections) > 0


@pytest.mark.asyncio
async def test_get_full_graph(clean_neo4j):
    """Test getting full graph"""
    # Create some nodes
    for i in range(5):
        data = ObservationCreate(text=f"Observation {i}", confidence=0.8)
        await graph_service.create_observation(data)

    graph_data = await graph_service.get_full_graph(limit=10)
    assert "nodes" in graph_data
    assert "edges" in graph_data
    assert len(graph_data["nodes"]) == 5
