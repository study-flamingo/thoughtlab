## Project Map

Quick reference for finding code in the ThoughtLab repository.

---

## High-Level Architecture

- **Backend**: FastAPI (Python) with Neo4j (graph), Redis (cache), PostgreSQL (scaffolded)
- **Frontend**: React + TypeScript + Vite + Tailwind + React Query + Cytoscape.js
- **AI**: LangChain/LangGraph + OpenAI for embeddings, agents, and MCP server
- **Infra**: Docker Compose for Neo4j and Redis

---

## Backend (FastAPI)

### App Entrypoint & Configuration

- [app/main.py](../backend/app/main.py) - FastAPI app, CORS, lifespan, mounts routers
- [app/core/config.py](../backend/app/core/config.py) - Environment configuration (Pydantic settings)

### API Routes

- [app/api/routes/nodes.py](../backend/app/api/routes/nodes.py) - Node CRUD (Observation, Source, Hypothesis, Entity, Concept)
- [app/api/routes/graph.py](../backend/app/api/routes/graph.py) - Full graph visualization endpoint
- [app/api/routes/settings.py](../backend/app/api/routes/settings.py) - App settings (theme, layout, colors)
- [app/api/routes/activities.py](../backend/app/api/routes/activities.py) - Activity feed, suggestions, processing status
- [app/api/routes/tools.py](../backend/app/api/routes/tools.py) - LLM-powered tools (find related, summarize, etc.)

### Services (Business Logic)

- [app/services/graph_service.py](../backend/app/services/graph_service.py) - Neo4j operations (CRUD, relationships, queries)
- [app/services/activity_service.py](../backend/app/services/activity_service.py) - Activity feed CRUD, suggestions workflow
- [app/services/processing_service.py](../backend/app/services/processing_service.py) - Background processing orchestrator
- [app/services/job_service.py](../backend/app/services/job_service.py) - Redis-based job queue for async operations
- [app/services/report_service.py](../backend/app/services/report_service.py) - Report storage for LangGraph results
- [app/services/embedding_service.py](../backend/app/services/embedding_service.py) - Legacy embedding interface (stub)

### Tool Service (Modular LLM Operations)

- [app/services/tools/service.py](../backend/app/services/tools/service.py) - ToolService facade class
- [app/services/tools/base.py](../backend/app/services/tools/base.py) - Shared config, LLM init, Neo4j helpers
- [app/services/tools/operations/node_analysis.py](../backend/app/services/tools/operations/node_analysis.py) - find_related, summarize, confidence
- [app/services/tools/operations/node_modification.py](../backend/app/services/tools/operations/node_modification.py) - reclassify, merge, web_evidence
- [app/services/tools/operations/relationship_analysis.py](../backend/app/services/tools/operations/relationship_analysis.py) - edge summarize, confidence, reclassify

### Shared Tool Definitions

- [app/tools/tool_definitions.py](../backend/app/tools/tool_definitions.py) - Tool metadata, execution modes, parameters
- [app/tools/tool_registry.py](../backend/app/tools/tool_registry.py) - Registry for MCP + LangGraph consumption

### AI Module (LangChain/LangGraph)

- [app/ai/config.py](../backend/app/ai/config.py) - AI configuration (models, thresholds, API keys)
- [app/ai/embeddings.py](../backend/app/ai/embeddings.py) - OpenAI embeddings via LangChain
- [app/ai/similarity.py](../backend/app/ai/similarity.py) - Vector similarity search (Neo4j indexes)
- [app/ai/classifier.py](../backend/app/ai/classifier.py) - LLM relationship classification
- [app/ai/workflow.py](../backend/app/ai/workflow.py) - Main AI processing workflow

### Agent Layer (LangGraph)

- [app/agents/agent.py](../backend/app/agents/agent.py) - ReAct agent with intelligent tool selection
- [app/agents/agent_tools.py](../backend/app/agents/agent_tools.py) - LangGraph tools (call ToolService directly)
- [app/agents/config.py](../backend/app/agents/config.py) - Agent configuration
- [app/agents/state.py](../backend/app/agents/state.py) - Agent state management

### MCP Server (Mounted at /mcp)

- [app/mcp/server.py](../backend/app/mcp/server.py) - FastMCP server creation
- [app/mcp/mcp_tools.py](../backend/app/mcp/mcp_tools.py) - MCP tool wrappers (call ToolService directly)
- [mcp_server.py](../backend/mcp_server.py) - Standalone MCP server entry point

### Models (Request/Response DTOs)

- [app/models/nodes.py](../backend/app/models/nodes.py) - Node types (Observation, Hypothesis, Source, etc.)
- [app/models/activity.py](../backend/app/models/activity.py) - Activity types, suggestions, processing data
- [app/models/settings.py](../backend/app/models/settings.py) - App settings models
- [app/models/tool_models.py](../backend/app/models/tool_models.py) - AI tool request/response models
- [app/models/job_models.py](../backend/app/models/job_models.py) - Job queue and report models

### Database Connectors

- [app/db/neo4j.py](../backend/app/db/neo4j.py) - Async Neo4j driver manager
- [app/db/redis.py](../backend/app/db/redis.py) - Async Redis client manager
- [app/db/postgres.py](../backend/app/db/postgres.py) - Async SQLAlchemy (scaffolded)

### Utilities

- [app/utils/chunking.py](../backend/app/utils/chunking.py) - Text chunking for embeddings

### Examples & Validation

- [examples/agent_demo.py](../backend/examples/agent_demo.py) - LangGraph agent demo
- [validate_tools.py](../backend/validate_tools.py) - Validate backend API
- [validate_agent.py](../backend/validate_agent.py) - Validate LangGraph layer
- [validate_mcp.py](../backend/validate_mcp.py) - Validate MCP server

### Tests

- [tests/conftest.py](../backend/tests/conftest.py) - Shared fixtures
- [tests/test_api_nodes.py](../backend/tests/test_api_nodes.py) - API endpoint tests
- [tests/test_graph_service.py](../backend/tests/test_graph_service.py) - Service layer tests
- [tests/test_models.py](../backend/tests/test_models.py) - Model validation tests
- [tests/test_api_activities.py](../backend/tests/test_api_activities.py) - Activity feed tests
- [tests/test_activity_service.py](../backend/tests/test_activity_service.py) - Activity service tests
- [tests/test_api_tools.py](../backend/tests/test_api_tools.py) - AI tool API endpoint tests
- [tests/test_tool_service.py](../backend/tests/test_tool_service.py) - Tool service unit tests

---

## Backend API Endpoints

### Graph

- `GET /api/v1/graph/full` - Get full graph for visualization

### Nodes

- `POST /api/v1/nodes/observations` - Create Observation
- `GET /api/v1/nodes/observations/{id}` - Get Observation
- `GET /api/v1/nodes/observations` - List Observations
- `PUT /api/v1/nodes/observations/{id}` - Update Observation
- `POST /api/v1/nodes/sources` - Create Source
- `POST /api/v1/nodes/hypotheses` - Create Hypothesis
- `PUT /api/v1/nodes/hypotheses/{id}` - Update Hypothesis
- `POST /api/v1/nodes/entities` - Create Entity
- `PUT /api/v1/nodes/entities/{id}` - Update Entity
- `GET /api/v1/nodes/{id}` - Get any node by ID
- `GET /api/v1/nodes/{id}/connections` - Get node connections
- `DELETE /api/v1/nodes/{id}` - Delete node

### Relationships

- `POST /api/v1/nodes/relationships` - Create relationship
- `GET /api/v1/nodes/relationships/{id}` - Get relationship
- `PUT /api/v1/nodes/relationships/{id}` - Update relationship
- `DELETE /api/v1/nodes/relationships/{id}` - Delete relationship

### Settings

- `GET /api/v1/settings` - Get app settings
- `PUT /api/v1/settings` - Update app settings

### Activities

- `GET /api/v1/activities` - List activities
- `GET /api/v1/activities/pending` - Get pending suggestions
- `GET /api/v1/activities/processing/{node_id}` - Get processing status
- `GET /api/v1/activities/{id}` - Get activity
- `POST /api/v1/activities/{id}/approve` - Approve suggestion
- `POST /api/v1/activities/{id}/reject` - Reject suggestion

### AI Tools

- `GET /api/v1/tools/health` - Backend health check
- `GET /api/v1/tools/capabilities` - List available tools
- `POST /api/v1/tools/nodes/{id}/find-related` - Find semantically similar nodes
- `POST /api/v1/tools/nodes/{id}/summarize` - AI-powered summary
- `POST /api/v1/tools/nodes/{id}/summarize-with-context` - Summary with relationships
- `POST /api/v1/tools/nodes/{id}/recalculate-confidence` - Re-assess node confidence
- `POST /api/v1/tools/nodes/{id}/reclassify` - Change node type
- `POST /api/v1/tools/nodes/{id}/search-web-evidence` - Web search (placeholder)
- `POST /api/v1/tools/nodes/merge` - Merge two nodes
- `POST /api/v1/tools/relationships/{id}/summarize` - Explain relationship
- `POST /api/v1/tools/relationships/{id}/recalculate-confidence` - Re-assess edge confidence
- `POST /api/v1/tools/relationships/{id}/reclassify` - Change relationship type

---

## Frontend (React + Vite)

### App Bootstrap

- [src/main.tsx](../frontend/src/main.tsx) - React root, React Query provider
- [src/App.tsx](../frontend/src/App.tsx) - Layout, header, modals, selection state

### Components

- [components/GraphVisualizer.tsx](../frontend/src/components/GraphVisualizer.tsx) - Cytoscape graph visualization
- [components/NodeInspector.tsx](../frontend/src/components/NodeInspector.tsx) - Node details, editing, and AI tools
- [components/RelationInspector.tsx](../frontend/src/components/RelationInspector.tsx) - Relationship details and AI tools
- [components/CreateNodeModal.tsx](../frontend/src/components/CreateNodeModal.tsx) - Create nodes modal
- [components/CreateRelationModal.tsx](../frontend/src/components/CreateRelationModal.tsx) - Create relationships modal
- [components/SettingsModal.tsx](../frontend/src/components/SettingsModal.tsx) - App settings
- [components/ActivityFeed.tsx](../frontend/src/components/ActivityFeed.tsx) - Activity feed with polling
- [components/Toast.tsx](../frontend/src/components/Toast.tsx) - Toast notification system
- [components/AIToolsSection.tsx](../frontend/src/components/AIToolsSection.tsx) - Collapsible AI tools section

### API Client & Types

- [services/api.ts](../frontend/src/services/api.ts) - Axios client, API endpoints
- [types/graph.ts](../frontend/src/types/graph.ts) - Node, edge, graph types
- [types/settings.ts](../frontend/src/types/settings.ts) - Settings types
- [types/activity.ts](../frontend/src/types/activity.ts) - Activity types
- [types/tools.ts](../frontend/src/types/tools.ts) - AI tool request/response types

### Configuration

- [package.json](../frontend/package.json) - Scripts, dependencies
- [vite.config.ts](../frontend/vite.config.ts) - Vite build config
- [vitest.config.ts](../frontend/vitest.config.ts) - Test config
- [tailwind.config.js](../frontend/tailwind.config.js) - Tailwind config

### Tests

- [src/App.test.tsx](../frontend/src/App.test.tsx) - App component tests
- [components/__tests__/ActivityFeed.test.tsx](../frontend/src/components/__tests__/ActivityFeed.test.tsx)
- [components/__tests__/CreateNodeModal.test.tsx](../frontend/src/components/__tests__/CreateNodeModal.test.tsx)
- [components/__tests__/GraphVisualizer.test.tsx](../frontend/src/components/__tests__/GraphVisualizer.test.tsx)
- [components/__tests__/SettingsModal.test.tsx](../frontend/src/components/__tests__/SettingsModal.test.tsx)
- [services/__tests__/api.test.ts](../frontend/src/services/__tests__/api.test.ts)
- [test/setup.ts](../frontend/src/test/setup.ts) - Test configuration

---

## Infrastructure & Scripts

### Docker

- [docker-compose.yml](../docker-compose.yml) - Neo4j and Redis services
- [docker/neo4j/init.cypher](../docker/neo4j/init.cypher) - Neo4j initialization

### Scripts

- [setup.sh](../scripts/setup.sh) - End-to-end setup (Linux/Mac/WSL)
- [setup.ps1](../scripts/setup.ps1) - End-to-end setup (Windows)
- [start.sh](../scripts/start.sh) - Start all services
- [stop.sh](../scripts/stop.sh) - Stop all services
- [restart.sh](../scripts/restart.sh) - Restart all services
- [init_neo4j.sh](../scripts/init_neo4j.sh) - Initialize Neo4j constraints/indexes
- [backend/scripts/init_neo4j.py](../backend/scripts/init_neo4j.py) - Python Neo4j init
- [check_versions.py](../scripts/check_versions.py) - Check Python/Node versions

---

## Documentation

| Document | Purpose |
|----------|---------|
| [README.md](../README.md) | Project introduction, quickstart |
| [DEVELOPMENT_GUIDE.md](../DEVELOPMENT_GUIDE.md) | Comprehensive dev guide (architecture, setup, workflows) |
| [PROJECT_MAP.md](./PROJECT_MAP.md) | This file - code location reference |
| [SETUP.md](./SETUP.md) | Detailed installation guide |
| [TESTING.md](./TESTING.md) | Testing guide |
| [DEPENDENCIES.md](./DEPENDENCIES.md) | Dependency management with uv |
| [MCP_SERVER_GUIDE.md](./MCP_SERVER_GUIDE.md) | MCP server setup and usage |
| [SECURITY.md](../SECURITY.md) | Security policy |

---

## Feature Locations

### Node Types
- **Models**: `backend/app/models/nodes.py`
- **Service**: `backend/app/services/graph_service.py`
- **API**: `backend/app/api/routes/nodes.py`
- **Frontend**: `frontend/src/components/CreateNodeModal.tsx`, `NodeInspector.tsx`

### Relationships
- **Models**: `backend/app/models/nodes.py` (relationship types)
- **Service**: `backend/app/services/graph_service.py` (create/update/delete)
- **API**: `backend/app/api/routes/nodes.py`
- **Frontend**: `frontend/src/components/CreateRelationModal.tsx`, `RelationInspector.tsx`

### Graph Visualization
- **Frontend**: `frontend/src/components/GraphVisualizer.tsx` (Cytoscape.js)
- **API**: `backend/app/api/routes/graph.py` (`GET /graph/full`)
- **Service**: `backend/app/services/graph_service.py` (`get_full_graph`)

### Activity Feed
- **Models**: `backend/app/models/activity.py`
- **Service**: `backend/app/services/activity_service.py`
- **API**: `backend/app/api/routes/activities.py`
- **Frontend**: `frontend/src/components/ActivityFeed.tsx`

### AI Embedding & Similarity
- **Config**: `backend/app/ai/config.py`
- **Embeddings**: `backend/app/ai/embeddings.py` (OpenAI integration)
- **Similarity**: `backend/app/ai/similarity.py` (Neo4j vector search)
- **Classifier**: `backend/app/ai/classifier.py` (LLM relationship classification)
- **Workflow**: `backend/app/ai/workflow.py` (orchestration)

### LLM Tools (3-Layer Architecture)
- **API**: `backend/app/api/routes/tools.py`
- **Core Service**: `backend/app/services/tools/` (modular operations)
- **Tool Definitions**: `backend/app/tools/tool_definitions.py`
- **Tool Registry**: `backend/app/tools/tool_registry.py`
- **Models**: `backend/app/models/tool_models.py`
- **Job Queue**: `backend/app/services/job_service.py`
- **Reports**: `backend/app/services/report_service.py`

### LangGraph Agents
- **Agent**: `backend/app/agents/agent.py`
- **Tools**: `backend/app/agents/agent_tools.py` (call ToolService directly)
- **Config**: `backend/app/agents/config.py`
- **Demo**: `backend/examples/agent_demo.py`

### MCP Server (Mounted at /mcp)
- **Server**: `backend/app/mcp/server.py`
- **Tools**: `backend/app/mcp/mcp_tools.py` (call ToolService directly)
- **Entry**: `backend/mcp_server.py`
- **Config**: `claude_desktop_config.example.json`
- **Admin Mode**: Set `THOUGHTLAB_MCP_ADMIN_MODE=true` for dangerous tools

### Settings
- **Models**: `backend/app/models/settings.py`
- **Service**: `backend/app/services/graph_service.py` (settings CRUD)
- **API**: `backend/app/api/routes/settings.py`
- **Frontend**: `frontend/src/components/SettingsModal.tsx`

---

## Data Flow Examples

### Create Node with AI Analysis

```
Frontend → POST /api/v1/nodes/observations
          ↓
Route validates with Pydantic
          ↓
graph_service.create_observation()
          ↓
Neo4j creates node
          ↓
processing_service delegates to ai_workflow
          ↓
- Generate embeddings (app/ai/embeddings.py)
- Store in Neo4j vector index
- Find similar nodes (app/ai/similarity.py)
- Classify relationships (app/ai/classifier.py)
- Create activities for suggestions
          ↓
Return node ID to frontend
```

### Graph Visualization

```
GraphVisualizer → graphApi.getFullGraph()
                ↓
GET /api/v1/graph/full
                ↓
graph_service.get_full_graph()
                ↓
Cypher query to Neo4j
                ↓
Return {nodes: [...], edges: [...]}
                ↓
GraphVisualizer renders with Cytoscape.js
```

### LangGraph Agent Query

```
User: "Find nodes related to obs-123"
                ↓
Agent (app/agents/agent.py) - ReAct reasoning
                ↓
Selects find_related_nodes tool
                ↓
Tool (agent_tools.py) calls ToolService directly (in-process)
                ↓
ToolService.find_related_nodes()
                ↓
- Get node from graph_service
- Find similar via similarity search
- Format results
- Save report to ReportService
                ↓
Return to agent
                ↓
Agent synthesizes response to user
```

### MCP Server Query

```
Claude Desktop: "Find related nodes for obs-123"
                ↓
Streamable HTTP to /mcp
                ↓
MCP tool (mcp_tools.py) calls ToolService directly
                ↓
ToolService.find_related_nodes()
                ↓
- Get node from graph_service
- Find similar via similarity search
- Format results
                ↓
Return formatted text to Claude Desktop
```

---

## Quick Commands

### Backend
```bash
cd backend
uvicorn app.main:app --reload  # Dev server
pytest                          # Run tests
pytest --cov=app                # With coverage
python validate_tools.py        # Validate tools API
python validate_agent.py        # Validate agent
python validate_mcp.py          # Validate MCP server
python examples/agent_demo.py   # Agent demo
```

### Frontend
```bash
cd frontend
npm run dev                     # Dev server
npm test                        # Run tests
npm run test:coverage           # With coverage
npm run build                   # Production build
```

### Infrastructure
```bash
docker-compose up -d            # Start services
docker-compose ps               # Check status
docker-compose logs neo4j       # View logs
./start.sh                      # Start all
./stop.sh                       # Stop all
```

---

**Last Updated**: 2026-01-03
