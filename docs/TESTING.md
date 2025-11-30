# Testing Guide

## Quick Reference

| Stack | Command | Coverage |
|-------|---------|----------|
| Backend | `pytest` | `pytest --cov=app --cov-report=html` |
| Frontend | `npm test` | `npm run test:coverage` |

---

## Backend Tests (pytest)

### Setup

```bash
cd backend
source .venv/bin/activate
# Dependencies already include pytest
```

### Run Tests

```bash
pytest                              # All tests
pytest tests/test_api_nodes.py      # Specific file
pytest tests/test_api_nodes.py::test_create_observation  # Specific test
pytest --cov=app --cov-report=html  # With coverage
```

### Test Files

| File | Purpose |
|------|---------|
| `tests/test_api_nodes.py` | API endpoint integration tests |
| `tests/test_graph_service.py` | Service layer unit tests |
| `tests/test_models.py` | Pydantic model validation |
| `tests/conftest.py` | Shared fixtures |

### Fixtures

- `client` — FastAPI test client
- `clean_neo4j` — Cleans database before/after each test
- `setup_databases` — Sets up connections for test session

### Writing Tests

```python
def test_my_endpoint(client: TestClient, clean_neo4j):
    response = client.get("/api/v1/my-endpoint")
    assert response.status_code == 200
```

For async tests:
```python
@pytest.mark.asyncio
async def test_async_function():
    result = await my_async_function()
    assert result is not None
```

---

## Frontend Tests (vitest)

### Setup

```bash
cd frontend
npm install  # Includes vitest and testing-library
```

### Run Tests

```bash
npm test                # Watch mode
npm test -- --run       # Run once
npm run test:ui         # Visual UI
npm run test:coverage   # With coverage
```

### Test Files

| File | Purpose |
|------|---------|
| `src/App.test.tsx` | Main app component |
| `src/components/__tests__/*.test.tsx` | Component tests |
| `src/services/__tests__/api.test.ts` | API client tests |
| `src/test/setup.ts` | Test configuration |
| `src/test/utils.tsx` | Test utilities |

### Writing Tests

```typescript
import { describe, it, expect } from 'vitest';
import { render, screen } from '../test/utils';
import MyComponent from '../MyComponent';

describe('MyComponent', () => {
  it('renders correctly', () => {
    render(<MyComponent />);
    expect(screen.getByText('Hello')).toBeInTheDocument();
  });
});
```

Use `renderWithProviders` for components needing React Query:
```typescript
import { renderWithProviders } from '../test/utils';
renderWithProviders(<MyComponent />);
```

---

## Coverage

### Current Test Coverage

**Backend:**
- ✅ API endpoints (create, read, list, update, delete)
- ✅ Service layer CRUD operations
- ✅ Model validation
- ✅ Error handling
- ✅ Relationship operations

**Frontend:**
- ✅ Component rendering
- ✅ User interactions
- ✅ API integration
- ✅ Loading/error states
- ✅ Form validation

---

## Best Practices

1. **Test behavior, not implementation** — Focus on what users see/do
2. **Keep tests isolated** — Each test should be independent
3. **Use descriptive names** — Test names explain what they verify
4. **Mock external dependencies** — Don't hit real APIs in tests
5. **Clean up** — Use fixtures to reset state between tests

---

## CI/CD Example

```yaml
name: Tests
on: [push, pull_request]

jobs:
  backend:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
      - run: |
          cd backend
          pip install -r requirements.txt
          pytest --cov=app

  frontend:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
      - run: |
          cd frontend
          npm ci
          npm test -- --run
```

---

## Troubleshooting

**Backend: Database connection errors**
- Ensure Docker services are running: `docker-compose up -d`
- Check `.env` credentials

**Backend: Import errors**
- Activate virtual environment
- Run `pip install -r requirements.txt`

**Frontend: Module not found**
- Run `npm install`
- Check `vitest.config.ts`

**Frontend: React Query errors**
- Use `renderWithProviders` from test utils
