# Backend Tests

## Running Tests

```bash
cd backend
source venv/bin/activate
pytest
```

## Test Coverage

```bash
pytest --cov=app --cov-report=html
```

## Test Structure

- `tests/test_api_nodes.py` - API endpoint tests
- `tests/test_graph_service.py` - Service layer tests
- `tests/test_models.py` - Pydantic model validation tests
- `tests/conftest.py` - Shared fixtures and setup

## Test Fixtures

- `client` - FastAPI test client
- `clean_neo4j` - Cleans Neo4j database before/after each test
- `setup_databases` - Sets up database connections for test session

## Writing New Tests

1. Create test file: `tests/test_*.py`
2. Import fixtures from `conftest.py`
3. Use `@pytest.mark.asyncio` for async tests
4. Use `clean_neo4j` fixture to ensure clean state

Example:
```python
def test_my_endpoint(client: TestClient, clean_neo4j):
    response = client.get("/api/v1/my-endpoint")
    assert response.status_code == 200
```
