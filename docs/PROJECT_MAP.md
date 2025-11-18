## Project Map

This document maps the repository’s structure, major components, and the responsibilities of key files, with links so you can jump straight into the code.

## High-level architecture
- **Backend**: FastAPI (Python) with Neo4j (primary store), Redis (cache/RT), optional Postgres (scaffolded)
- **Frontend**: React + TypeScript + Vite + Tailwind + React Query + Cytoscape.js
- **Infra**: Docker Compose for Neo4j and Redis
- **Tests**: pytest (backend), vitest/testing-library (frontend)

## Backend (FastAPI)
- App entrypoint and wiring
  - [app/main.py](../backend/app/main.py): FastAPI app, CORS, lifespan startup/shutdown (connects to Neo4j/Redis), mounts routers, `/` and `/health`.
  - [app/core/config.py](../backend/app/core/config.py): Centralized environment config via pydantic-settings (`NEO4J_*`, `DATABASE_URL`, `REDIS_URL`, `SECRET_KEY`, etc.).
- API routes
  - [app/api/routes/nodes.py](../backend/app/api/routes/nodes.py): CRUD for `Observation`, `Source`, `Hypothesis`, `Entity`; relationship creation; generic `GET /nodes/{id}`; `GET /{id}/connections`.
  - [app/api/routes/graph.py](../backend/app/api/routes/graph.py): `GET /graph/full` for full graph visualization payload.
- Services (domain logic)
  - [app/services/graph_service.py](../backend/app/services/graph_service.py): All Neo4j Cypher logic; create/update/read nodes; create relationships; `get_node_connections`; `get_full_graph`; JSON-safe conversions.
- Models (request/response DTOs, enums)
  - [app/models/nodes.py](../backend/app/models/nodes.py): Pydantic models for node types (`Observation`, `Hypothesis`, `Source`, `Concept`, `Entity`) and `RelationshipType`; `*Create`/`*Update`/`*Response` DTOs.
- Database connectors
  - [app/db/neo4j.py](../backend/app/db/neo4j.py): Async driver manager, `neo4j_conn.get_session()`.
  - [app/db/redis.py](../backend/app/db/redis.py): Async Redis client manager, `redis_conn.get_client()`.
  - [app/db/postgres.py](../backend/app/db/postgres.py): Async SQLAlchemy engine/session (currently scaffolded, not used by routes).
- Dependencies
  - [backend/requirements.txt](../backend/requirements.txt): Core backend dependencies and test deps.
- Tests
  - [backend/tests/](../backend/tests/): Backend test suite and fixtures.
    - [test_api_nodes.py](../backend/tests/test_api_nodes.py)
    - [test_graph_service.py](../backend/tests/test_graph_service.py)
    - [test_models.py](../backend/tests/test_models.py)
    - [conftest.py](../backend/tests/conftest.py)

### Backend endpoints (quick index)
- Graph
  - `GET /api/v1/graph/full` → [graph.py](../backend/app/api/routes/graph.py) → `graph_service.get_full_graph`
- Nodes
  - `POST /api/v1/nodes/observations` → create Observation
  - `GET /api/v1/nodes/observations/{id}` → get Observation
  - `GET /api/v1/nodes/observations` → list Observations
  - `PUT /api/v1/nodes/observations/{id}` → update Observation
  - `POST /api/v1/nodes/sources` → create Source
  - `POST /api/v1/nodes/hypotheses` / `PUT /api/v1/nodes/hypotheses/{id}` → create/update Hypothesis
  - `POST /api/v1/nodes/entities` / `PUT /api/v1/nodes/entities/{id}` → create/update Entity
  - `POST /api/v1/nodes/relationships` → create relationship
  - `GET /api/v1/nodes/{id}/connections?max_depth=2` → connections around a node
  - `GET /api/v1/nodes/{id}` → generic node by id

## Frontend (React + Vite)
- App bootstrap
  - [src/main.tsx](../frontend/src/main.tsx): React root; React Query provider with defaults.
  - [src/App.tsx](../frontend/src/App.tsx): Layout shell; header (“Add Node”), main content (graph), sidebar (inspector/feed), modal wiring.
- Components
  - [components/GraphVisualizer.tsx](../frontend/src/components/GraphVisualizer.tsx): Fetches full graph; renders Cytoscape graph; node selection and highlight; fit/reset; legend.
  - [components/NodeInspector.tsx](../frontend/src/components/NodeInspector.tsx): Loads a node by id; type-specific edit forms for Observation/Entity/Hypothesis; saves then invalidates caches.
  - [components/CreateNodeModal.tsx](../frontend/src/components/CreateNodeModal.tsx): Create Observation/Entity (placeholders for Source/Hypothesis UI); invalidates graph and closes.
  - [components/ActivityFeed.tsx](../frontend/src/components/ActivityFeed.tsx): Placeholder UI for future real-time updates.
- API client and types
  - [services/api.ts](../frontend/src/services/api.ts): Axios client; endpoints for full graph, node CRUD/updates, relationships (suggestion endpoints are placeholders).
  - [types/graph.ts](../frontend/src/types/graph.ts): TS types for nodes, edges, graph payloads, suggestions.
- Config and tooling
  - [package.json](../frontend/package.json): Scripts, dependencies.
  - [index.html](../frontend/index.html), [src/index.css](../frontend/src/index.css), [tailwind.config.js](../frontend/tailwind.config.js)
  - [vite.config.ts](../frontend/vite.config.ts), [vitest.config.ts](../frontend/vitest.config.ts), [tsconfig.json](../frontend/tsconfig.json)
- Tests
  - [src/App.test.tsx](../frontend/src/App.test.tsx)
  - [components/__tests__/ActivityFeed.test.tsx](../frontend/src/components/__tests__/ActivityFeed.test.tsx)
  - [components/__tests__/CreateNodeModal.test.tsx](../frontend/src/components/__tests__/CreateNodeModal.test.tsx)
  - [components/__tests__/GraphVisualizer.test.tsx](../frontend/src/components/__tests__/GraphVisualizer.test.tsx)
  - [services/__tests__/api.test.ts](../frontend/src/services/__tests__/api.test.ts)
  - Test setup: [src/test/setup.ts](../frontend/src/test/setup.ts)

## Orchestration & scripts
- Compose services
  - [docker-compose.yml](../docker-compose.yml): Neo4j (+APOC) and Redis with health checks and volumes.
  - Neo4j init: [docker/neo4j/init.cypher](../docker/neo4j/init.cypher)
- Local scripts (root)
  - [scripts/setup.sh](../scripts/setup.sh), [scripts/setup.ps1](../scripts/setup.ps1): End-to-end setup.
  - [scripts/start.sh](../scripts/start.sh), [scripts/stop.sh](../scripts/stop.sh), [scripts/restart.sh](../scripts/restart.sh)
  - [scripts/init_neo4j.sh](../scripts/init_neo4j.sh), [backend/scripts/init_neo4j.py](../backend/scripts/init_neo4j.py)
  - [scripts/check_versions.py](../scripts/check_versions.py)

## Documentation
- Overview & decisions
  - [01-PROJECT_OVERVIEW.md](./01-PROJECT_OVERVIEW.md)
  - [02-ARCHITECTURE_DECISIONS.md](./02-ARCHITECTURE_DECISIONS.md)
  - [03-DEVELOPMENT_ROADMAP.md](./03-DEVELOPMENT_ROADMAP.md)
  - [04-TECHNICAL_SETUP_GUIDE.md](./04-TECHNICAL_SETUP_GUIDE.md)
- Top-level guides
  - [README.md](../README.md)
  - [BACKEND_SETUP.md](./BACKEND_SETUP.md)
  - [FRONTEND_SETUP.md](./FRONTEND_SETUP.md)
  - [UV_SETUP.md](./UV_SETUP.md)
  - [TESTING.md](./TESTING.md)

## How pieces connect (data flow)
- Frontend
  - Graph view: `GraphVisualizer` → `graphApi.getFullGraph()` → `GET /api/v1/graph/full` → Neo4j via `graph_service.get_full_graph`.
  - Node details: `NodeInspector` → `graphApi.getNode(id)` → `GET /api/v1/nodes/{id}` → `graph_service.get_node`.
  - Create node: `CreateNodeModal` → `graphApi.createObservation|createEntity` → `POST /api/v1/nodes/...` → service `create_*` → invalidate `['graph']`.
  - Update node: `NodeInspector` → `graphApi.update*` → `PUT /api/v1/nodes/...` → service `update_*` → invalidate `['node', id]` and `['graph']`.
- Backend
  - Routes validate DTOs (Pydantic) → delegate to `graph_service` → Cypher via `neo4j_conn.get_session()` → JSON-safe responses (temporal types normalized).

## Where to start coding
- Add/modify API endpoints: [app/api/routes/](../backend/app/api/routes/)
- Extend graph logic: [app/services/graph_service.py](../backend/app/services/graph_service.py)
- Adjust validations/models: [app/models/nodes.py](../backend/app/models/nodes.py)
- Infra/config: [app/core/config.py](../backend/app/core/config.py), [app/db/*](../backend/app/db/), [docker-compose.yml](../docker-compose.yml)
- UI features: [src/components/](../frontend/src/components/)
- API/types: [src/services/api.ts](../frontend/src/services/api.ts), [src/types/graph.ts](../frontend/src/types/graph.ts)

## Quick commands
- Backend
  - Dev server: `uvicorn app.main:app --reload` (from [backend](../backend/))
  - Tests: `pytest` (from [backend](../backend/))
- Frontend
  - Dev server: `npm run dev` (from [frontend](../frontend/))
  - Tests: `npm test` (from [frontend](../frontend/))


