from fastapi import APIRouter, HTTPException, Query
from app.models.nodes import (
    ObservationCreate,
    ObservationUpdate,
    ObservationResponse,
    SourceCreate,
    SourceUpdate,
    SourceResponse,
    HypothesisCreate,
    HypothesisUpdate,
    HypothesisResponse,
    EntityCreate,
    EntityUpdate,
    EntityResponse,
    RelationshipCreate,
    RelationshipResponse,
)
from app.services.graph_service import graph_service
from typing import List
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

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
async def get_all_observations(limit: int = Query(100, ge=1, le=1000)):
    """Get all observation nodes"""
    observations = await graph_service.get_all_observations(limit=limit)
    return {"nodes": observations}


@router.put("/observations/{node_id}", response_model=dict)
async def update_observation(node_id: str, data: ObservationUpdate):
    """Update an observation node"""
    success = await graph_service.update_observation(node_id, data)
    if not success:
        raise HTTPException(status_code=404, detail="Observation not found or no changes provided")
    return {"id": node_id, "message": "Observation updated"}


@router.post("/sources", response_model=dict)
async def create_source(data: SourceCreate):
    """Create a new source node"""
    node_id = await graph_service.create_source(data)
    return {"id": node_id, "message": "Source created"}


@router.put("/hypotheses/{node_id}", response_model=dict)
async def update_hypothesis(node_id: str, data: HypothesisUpdate):
    """Update a hypothesis node"""
    success = await graph_service.update_hypothesis(node_id, data)
    if not success:
        raise HTTPException(status_code=404, detail="Hypothesis not found or no changes provided")
    return {"id": node_id, "message": "Hypothesis updated"}


@router.post("/hypotheses", response_model=dict)
async def create_hypothesis(data: HypothesisCreate):
    """Create a new hypothesis node"""
    node_id = await graph_service.create_hypothesis(data)
    return {"id": node_id, "message": "Hypothesis created"}


@router.put("/entities/{node_id}", response_model=dict)
async def update_entity(node_id: str, data: EntityUpdate):
    """Update an entity node"""
    success = await graph_service.update_entity(node_id, data)
    if not success:
        raise HTTPException(status_code=404, detail="Entity not found or no changes provided")
    return {"id": node_id, "message": "Entity updated"}


@router.post("/entities", response_model=dict)
async def create_entity(data: EntityCreate):
    """Create a new entity node"""
    node_id = await graph_service.create_entity(data)
    return {"id": node_id, "message": "Entity created"}


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
            # Inverse metadata stored on the relationship for reference
            "inverse_relationship_type": data.inverse_relationship_type.value if data.inverse_relationship_type else None,
            "inverse_confidence": data.inverse_confidence,
            "inverse_notes": data.inverse_notes,
        }
    )
    if not success:
        raise HTTPException(
            status_code=400,
            detail="Could not create relationship. Check that both nodes exist."
        )
    return {"message": "Relationship created"}


@router.get("/{node_id}/connections", response_model=dict)
async def get_connections(node_id: str, max_depth: int = Query(2, ge=1, le=5)):
    """Get all connections for a node"""
    connections = await graph_service.get_node_connections(node_id, max_depth)
    return {"node_id": node_id, "connections": connections}


@router.get("/{node_id}", response_model=dict)
async def get_node(node_id: str):
    """Get any node by ID"""
    try:
        node = await graph_service.get_node(node_id)
        if not node:
            raise HTTPException(status_code=404, detail="Node not found")
        # Use jsonable_encoder to safely handle special types
        return JSONResponse(content=jsonable_encoder(node))
    except HTTPException:
        # Re-raise explicit HTTP errors
        raise
    except Exception as e:
        # Surface unexpected errors with a clear message
        raise HTTPException(status_code=500, detail=f"Failed to fetch node: {str(e)}")
