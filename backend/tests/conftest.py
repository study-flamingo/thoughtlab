import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from app.main import app
from app.db.neo4j import neo4j_conn
from app.core.config import settings

# Ensure app runs in test mode for gated test endpoints
settings.environment = "test"


@pytest.fixture
def client(clean_neo4j):
    """Create test client; app lifespan manages DB connections."""
    with TestClient(app) as c:
        yield c


@pytest_asyncio.fixture(loop_scope="function")
async def clean_neo4j(request: pytest.FixtureRequest):
    """Ensure Neo4j is connected and clean database before/after each test."""
    uses_test_client = "client" in request.fixturenames
    if uses_test_client:
        # For API tests, reset via app endpoint to ensure same event loop
        with TestClient(app) as c:
            c.post("/api/v1/graph/__test__/reset")
        yield
        with TestClient(app) as c:
            c.post("/api/v1/graph/__test__/reset")
    else:
        # For service-layer async tests, manage connection directly
        if neo4j_conn.driver is None:
            await neo4j_conn.connect()
        try:
            async with neo4j_conn.get_session() as session:
                await session.run("MATCH (n) DETACH DELETE n")
        except Exception:
            # Best-effort cleanup; ignore event loop/transport edge cases
            pass
        yield
        try:
            async with neo4j_conn.get_session() as session:
                await session.run("MATCH (n) DETACH DELETE n")
        except Exception:
            pass
