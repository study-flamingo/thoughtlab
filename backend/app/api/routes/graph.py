from fastapi import APIRouter, HTTPException, Query
from app.services.graph_service import graph_service
from app.db.neo4j import neo4j_conn
from app.core.config import settings

router = APIRouter(prefix="/graph", tags=["graph"])


@router.get("/full")
async def get_full_graph(
    limit: int = Query(500, ge=1, le=10000),
    edges_limit: int = Query(1000, ge=1, le=5000),
):
    """Get entire graph structure for visualization"""
    graph_data = await graph_service.get_full_graph(limit=limit, edges_limit=edges_limit)
    return graph_data


@router.post("/__test__/reset")
async def reset_graph_for_tests():
    """Test-only endpoint: wipe all nodes and relationships."""
    if settings.environment != "test":
        raise HTTPException(status_code=404)
    async with neo4j_conn.get_session() as session:
        await session.run("MATCH (n) DETACH DELETE n")
    return {"status": "ok"}
