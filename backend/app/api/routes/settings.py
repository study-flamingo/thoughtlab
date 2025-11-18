from fastapi import APIRouter, HTTPException
from app.models.settings import AppSettingsUpdate
from app.services.graph_service import graph_service
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/settings", tags=["settings"])


@router.get("", response_model=dict)
async def get_settings():
    """Fetch application settings (singleton)."""
    settings = await graph_service.get_settings()
    return JSONResponse(content=jsonable_encoder(settings))


@router.put("", response_model=dict)
async def update_settings(update: AppSettingsUpdate):
    """Update application settings."""
    try:
        settings = await graph_service.update_settings(update)
        return JSONResponse(content=jsonable_encoder(settings))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update settings: {str(e)}")


