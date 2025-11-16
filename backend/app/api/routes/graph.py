from fastapi import APIRouter
from app.services.graph_service import graph_service

router = APIRouter(prefix="/graph", tags=["graph"])


@router.get("/full")
async def get_full_graph(limit: int = 500):
    """Get entire graph structure for visualization"""
    graph_data = await graph_service.get_full_graph(limit=limit)
    return graph_data
