from contextlib import asynccontextmanager
import logging
import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.db.neo4j import neo4j_conn
from app.db.redis import redis_conn
from app.api.routes import nodes, graph, settings as settings_routes, activities, tools
from app.mcp import create_mcp_server
from typing import Callable, Awaitable


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logging.info("Starting up...")
    # Retry/backoff for external services
    async def _retry(op: Callable[[], Awaitable[None]], name: str, attempts: int = 3, delay_s: float = 0.5):
        last_exc = None
        for i in range(attempts):
            try:
                await op()
                return
            except Exception as e:
                last_exc = e
                logging.warning(f"{name} connect attempt {i+1}/{attempts} failed: {e}")
                await asyncio.sleep(delay_s)
        if last_exc:
            raise last_exc

    # Connect to Neo4j (required)
    await _retry(neo4j_conn.connect, "neo4j")

    # Connect to Redis (optional - for caching)
    try:
        await _retry(redis_conn.connect, "redis", attempts=1)
    except Exception as e:
        logging.warning(f"Redis not available: {e}. Continuing without cache.")

    yield

    # Shutdown
    await neo4j_conn.disconnect()
    try:
        await redis_conn.disconnect()
    except Exception:
        pass  # Redis wasn't connected
    logging.info("Shutting down...")


app = FastAPI(
    title="Research Connection Graph API",
    description="API for managing research knowledge graphs with AI-powered connection discovery",
    version="0.2.0-alpha",
    lifespan=lifespan,
)

# Basic logging config
logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    # Keep explicit origins for clarity during development
    allow_origins=settings.cors_allow_origins,
    # Also allow localhost/127.0.0.1 on any port (useful when Vite picks another port)
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
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
app.include_router(settings_routes.router, prefix="/api/v1")
app.include_router(activities.router, prefix="/api/v1")
app.include_router(tools.router, prefix="/api/v1")

# Mount MCP server at /mcp for Streamable HTTP access
try:
    mcp_server = create_mcp_server()
    app.mount("/mcp", mcp_server.http_app())
    logging.info("MCP server mounted at /mcp")
except Exception as e:
    logging.warning(f"Failed to mount MCP server: {e}")


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
        # Prefer driver-level connectivity check if available
        if neo4j_conn.driver is not None:
            await neo4j_conn.driver.verify_connectivity()
        else:
            async with neo4j_conn.get_session() as session:
                await session.run("RETURN 1")
        health_status["neo4j"] = "healthy"
    except Exception as e:
        health_status["neo4j"] = f"unhealthy: {str(e)}"
    
    # Check Redis (optional)
    try:
        if redis_conn.client is not None:
            await redis_conn.get_client().ping()
            health_status["redis"] = "healthy"
        else:
            health_status["redis"] = "not configured (optional)"
    except Exception as e:
        health_status["redis"] = f"not available: {str(e)}"
    
    # Overall health check - Redis is optional, so ignore it for overall status
    overall = "healthy" if all(
        v == "healthy" or v == settings.environment
        for k, v in health_status.items()
        if k not in ("environment", "redis")
    ) and health_status["neo4j"] == "healthy" else "degraded"
    
    return {"status": overall, "services": health_status}
