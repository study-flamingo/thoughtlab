import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.db.neo4j import neo4j_conn
from app.db.redis import redis_conn


@pytest.fixture(scope="session")
async def setup_databases():
    """Setup database connections for tests"""
    await neo4j_conn.connect()
    await redis_conn.connect()
    yield
    await neo4j_conn.disconnect()
    await redis_conn.disconnect()


@pytest.fixture
def client(setup_databases):
    """Create test client"""
    return TestClient(app)


@pytest.fixture
async def clean_neo4j():
    """Clean Neo4j database before each test"""
    async with neo4j_conn.get_session() as session:
        await session.run("MATCH (n) DETACH DELETE n")
    yield
    # Cleanup after test
    async with neo4j_conn.get_session() as session:
        await session.run("MATCH (n) DETACH DELETE n")
