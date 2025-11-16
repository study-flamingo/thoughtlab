# Testing Guide

This project includes comprehensive tests for both backend and frontend.

## Backend Tests

### Setup

Backend tests use `pytest` with async support.

```bash
cd backend
source venv/bin/activate
pip install -r requirements.txt  # Includes pytest
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html

# Run specific test file
pytest tests/test_api_nodes.py

# Run specific test
pytest tests/test_api_nodes.py::test_create_observation
```

### Test Files

- **test_api_nodes.py** - API endpoint integration tests
- **test_graph_service.py** - Service layer unit tests
- **test_models.py** - Pydantic model validation tests

### Test Coverage

Current coverage includes:
- ✅ API endpoints (create, read, list)
- ✅ Service layer CRUD operations
- ✅ Model validation
- ✅ Error handling
- ✅ Relationship creation and queries

## Frontend Tests

### Setup

Frontend tests use `vitest` with React Testing Library.

```bash
cd frontend
npm install  # Installs vitest and testing libraries
```

### Running Tests

```bash
# Run tests in watch mode
npm test

# Run tests once
npm test -- --run

# Run with UI
npm run test:ui

# Run with coverage
npm run test:coverage
```

### Test Files

- **App.test.tsx** - Main app component tests
- **CreateNodeModal.test.tsx** - Modal component tests
- **GraphVisualizer.test.tsx** - Graph display component tests
- **ActivityFeed.test.tsx** - Activity feed component tests
- **api.test.ts** - API client tests

### Test Coverage

Current coverage includes:
- ✅ Component rendering
- ✅ User interactions
- ✅ API integration
- ✅ Loading and error states
- ✅ Form validation

## Continuous Integration

To set up CI/CD:

1. **Backend**: Add pytest to CI pipeline
2. **Frontend**: Add vitest to CI pipeline
3. **Coverage**: Set coverage thresholds

Example GitHub Actions workflow:

```yaml
name: Tests

on: [push, pull_request]

jobs:
  backend-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
      - run: |
          cd backend
          pip install -r requirements.txt
          pytest --cov=app

  frontend-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3
      - run: |
          cd frontend
          npm install
          npm test -- --run
```

## Best Practices

1. **Write tests first** - TDD helps design better APIs
2. **Test behavior, not implementation** - Focus on what users see/do
3. **Keep tests isolated** - Each test should be independent
4. **Use descriptive names** - Test names should explain what they test
5. **Mock external dependencies** - Don't hit real APIs in tests
6. **Clean up** - Use fixtures to reset state between tests

## Troubleshooting

### Backend Tests

**Database connection errors:**
- Ensure Docker services are running: `docker-compose up -d`
- Check `.env` file has correct credentials

**Import errors:**
- Make sure virtual environment is activated
- Run `pip install -r requirements.txt`

### Frontend Tests

**Module not found:**
- Run `npm install` to install dependencies
- Check `vitest.config.ts` is correct

**React Query errors:**
- Use `renderWithProviders` from test utils
- Ensure QueryClientProvider wraps components
