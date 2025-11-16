from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.db.neo4j import neo4j_conn
from app.db.redis import redis_conn
from app.api.routes import nodes, graph


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Starting up...")
    await neo4j_conn.connect()
    await redis_conn.connect()
    yield
    # Shutdown
    await neo4j_conn.disconnect()
    await redis_conn.disconnect()
    print("Shutting down...")


app = FastAPI(
    title="Research Connection Graph API",
    description="API for managing research knowledge graphs with AI-powered connection discovery",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite dev server
        "http://localhost:3000",  # Alternative React port
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {
        "message": "Research Connection Graph API",
        "version": "0.1.0",
        "status": "running"
    }


# Register routers
app.include_router(nodes.router, prefix="/api/v1")
app.include_router(graph.router, prefix="/api/v1")


@app.get("/health")
async def health_check():
    health_status = {
        "api": "healthy",
        "neo4j": "unknown",
        "redis": "unknown",
        "environment": settings.environment
    }
    
    # Check Neo4j
    try:
        async with neo4j_conn.get_session() as session:
            await session.run("RETURN 1")
        health_status["neo4j"] = "healthy"
    except Exception as e:
        health_status["neo4j"] = f"unhealthy: {str(e)}"
    
    # Check Redis
    try:
        await redis_conn.get_client().ping()
        health_status["redis"] = "healthy"
    except Exception as e:
        health_status["redis"] = f"unhealthy: {str(e)}"
    
    overall = "healthy" if all(
        v == "healthy" or v == settings.environment
        for k, v in health_status.items()
        if k != "environment"
    ) else "degraded"
    
    return {"status": overall, "services": health_status}
