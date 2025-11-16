from fastapi import APIRouter, HTTPException
from app.models.nodes import (
    ObservationCreate,
    ObservationResponse,
    SourceCreate,
    SourceResponse,
    HypothesisCreate,
    HypothesisResponse,
    RelationshipCreate,
    RelationshipResponse,
)
from app.services.graph_service import graph_service
from typing import List

router = APIRouter(prefix="/nodes", tags=["nodes"])


@router.post("/observations", response_model=dict)
async def create_observation(data: ObservationCreate):
    """Create a new observation node"""
    node_id = await graph_service.create_observation(data)
    return {"id": node_id, "message": "Observation created"}


@router.get("/observations/{node_id}", response_model=dict)
async def get_observation(node_id: str):
    """Get an observation by ID"""
    observation = await graph_service.get_observation(node_id)
    if not observation:
        raise HTTPException(status_code=404, detail="Observation not found")
    return observation


@router.get("/observations", response_model=dict)
async def get_all_observations(limit: int = 100):
    """Get all observation nodes"""
    observations = await graph_service.get_all_observations(limit=limit)
    return {"nodes": observations}


@router.post("/sources", response_model=dict)
async def create_source(data: SourceCreate):
    """Create a new source node"""
    node_id = await graph_service.create_source(data)
    return {"id": node_id, "message": "Source created"}


@router.post("/hypotheses", response_model=dict)
async def create_hypothesis(data: HypothesisCreate):
    """Create a new hypothesis node"""
    node_id = await graph_service.create_hypothesis(data)
    return {"id": node_id, "message": "Hypothesis created"}


@router.get("/{node_id}/connections", response_model=dict)
async def get_connections(node_id: str, max_depth: int = 2):
    """Get all connections for a node"""
    connections = await graph_service.get_node_connections(node_id, max_depth)
    return {"node_id": node_id, "connections": connections}


@router.post("/relationships", response_model=dict)
async def create_relationship(data: RelationshipCreate):
    """Create a relationship between two nodes"""
    success = await graph_service.create_relationship(
        data.from_id,
        data.to_id,
        data.relationship_type,
        {
            "confidence": data.confidence,
            "notes": data.notes,
        }
    )
    if not success:
        raise HTTPException(
            status_code=400,
            detail="Could not create relationship. Check that both nodes exist."
        )
    return {"message": "Relationship created"}
