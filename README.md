# ThoughtLab 💭
- v0.2.0-alpha

A web-based research application that helps you discover meaningful connections between sources, observations, hypotheses, and concepts using graph database technology.

## Features

✅ **Node Management** - Create and manage observations, sources, and hypotheses
✅ **Relationship Tracking** - Connect nodes with typed relationships (SUPPORTS, CONTRADICTS, RELATES_TO, etc.)
✅ **Interactive Graph Visualization** - Visualize your knowledge graph with Cytoscape.js (zoom, pan, click nodes)
✅ **RESTful API** - Complete API for programmatic access
✅ **Node Details Panel** - Click nodes to view details
✅ **Real-time Updates** - Activity feed ready for live updates (WebSocket integration coming soon)
✅ **AI-Powered Analysis** - LLM-powered graph operations via REST API, LangGraph agents, and MCP server
✅ **Claude Desktop Integration** - Use Claude to interact with your knowledge graph via MCP
✅ **Intelligent Agents** - LangGraph agents with automatic tool selection for complex reasoning
✅ **Vector Search** - Semantic similarity search using OpenAI embeddings  

## Quickstart

### Prerequisites

- **Python 3.11+** (`python3 --version`)
- **Node.js 18+** (`node --version`)
- **Docker & Docker Compose** (`docker --version`)

### One-Command Setup

```bash
# Linux/Mac/WSL/Git Bash
./setup.sh

# Windows (PowerShell)
.\scripts\setup.ps1
```

This will:
- ✅ Install Python dependencies (using uv)
- ✅ Install Node.js dependencies
- ✅ Create and configure `.env` files
- ✅ Start Docker services (Neo4j, Redis)
- ✅ Initialize Neo4j database

### Start the App

```bash
./start.sh
```

This starts both backend and frontend servers. Open **<http://localhost:5173>** in your browser.

### Other Commands

```bash
./stop.sh      # Stop all servers
./restart.sh   # Restart all servers
```

## Manual Setup

If you prefer to set things up manually:

### 1. Start Database Services

```bash
docker-compose up -d
docker-compose ps  # Verify services are healthy
```

### 2. Set Up Backend

```bash
cd backend
uv venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
cp .env.example .env
# Edit .env and set SECRET_KEY (or setup.sh does this automatically)
uvicorn app.main:app --reload
```

### 3. Set Up Frontend

```bash
cd frontend
npm install
cp .env.example .env
npm run dev
```

### 4. Initialize Neo4j (First Time Only)

```bash
docker exec research-graph-neo4j cypher-shell -u neo4j -p research_graph_password < docker/neo4j/init.cypher
```

## Usage

### Creating Nodes

#### Via Web Interface

1. Open <http://localhost:5173>
2. Click **"+ Add Node"** button
3. Select node type (Observation, Source, Hypothesis, Entity)
4. Fill in the form and click **"Create"**

#### Via API

```bash
# Create an observation
curl -X POST "http://localhost:8000/api/v1/nodes/observations" \
  -H "Content-Type: application/json" \
  -d '{"text": "I observed that X leads to Y", "confidence": 0.8}'

# Create a source
curl -X POST "http://localhost:8000/api/v1/nodes/sources" \
  -H "Content-Type: application/json" \
  -d '{"title": "Research Paper", "url": "https://example.com", "source_type": "paper"}'
```

### Creating Relationships

```bash
curl -X POST "http://localhost:8000/api/v1/nodes/relationships" \
  -H "Content-Type: application/json" \
  -d '{
    "from_id": "node-id-1",
    "to_id": "node-id-2",
    "relationship_type": "SUPPORTS",
    "confidence": 0.9
  }'
```

**Relationship types:** `SUPPORTS`, `CONTRADICTS`, `RELATES_TO`, `OBSERVED_IN`, `DISCUSSES`, `EXTRACTED_FROM`, `DERIVED_FROM`

### Graph Visualization

- **Zoom**: Mouse wheel or pinch gesture
- **Pan**: Click and drag background
- **Select Node**: Click to view details
- **Node Colors**:
  - 🔵 Blue (circle) = Observation
  - 🟢 Green (diamond) = Hypothesis
  - 🟡 Yellow (rectangle) = Source
  - 🟣 Purple (hexagon) = Concept
  - 🔴 Red (rounded rectangle) = Entity

## API Documentation

- **Swagger UI**: <http://localhost:8000/docs>
- **ReDoc**: <http://localhost:8000/redoc>
- **Health Check**: <http://localhost:8000/health>

## Project Structure

```
thoughtlab/
├── setup.sh              # One-command setup
├── start.sh              # Start all servers
├── stop.sh               # Stop all servers
├── restart.sh            # Restart servers
├── backend/              # Python FastAPI backend
│   ├── app/
│   │   ├── api/routes/   # API endpoints
│   │   ├── core/         # Configuration
│   │   ├── db/           # Database connections
│   │   ├── models/       # Pydantic models
│   │   └── services/     # Business logic
│   └── tests/            # Backend tests
├── frontend/             # React TypeScript frontend
│   ├── src/
│   │   ├── components/   # React components
│   │   ├── services/     # API client
│   │   └── types/        # TypeScript types
│   └── package.json
├── docker-compose.yml    # Database services
├── docs/                 # Documentation
└── scripts/              # Utility scripts
    ├── setup.ps1         # Windows PowerShell setup
    └── install-hooks.sh  # Install git hooks
```

## Testing

### Backend

```bash
cd backend && source .venv/bin/activate
pytest                              # Run tests
pytest --cov=app --cov-report=html  # With coverage
```

### Frontend

```bash
cd frontend
npm test                 # Run tests
npm run test:coverage    # With coverage
```

See [TESTING.md](./docs/TESTING.md) for details.

## Configuration

### Backend (`backend/.env`)

```bash
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=research_graph_password
DATABASE_URL=sqlite:///./research_graph.db
REDIS_URL=redis://localhost:6379
SECRET_KEY=your-secret-key-here

# Optional LLM keys
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
```

### Frontend (`frontend/.env`)

```bash
VITE_API_URL=http://localhost:8000/api/v1
```

## Security

This project includes security scanning:

- **GitHub Action**: Scans for secrets on every push/PR using Gitleaks
- **Pre-commit hook**: Local secret detection (install with `./scripts/install-hooks.sh`)
- **Dependency audits**: npm and pip dependencies are scanned

See [SECURITY.md](./SECURITY.md) for details.

## Troubleshooting

### Backend won't start

```bash
# Check if port 8000 is in use
lsof -i :8000  # or: netstat -an | grep 8000

# Make sure venv is activated
source backend/.venv/bin/activate
```

### Frontend connection errors

- Verify backend is running: <http://localhost:8000/health>
- Check `frontend/.env` has correct `VITE_API_URL`

### Docker issues

```bash
# Check service logs
docker-compose logs neo4j

# Reset (WARNING: deletes data)
docker-compose down -v && docker-compose up -d
```

### Neo4j not ready

Neo4j takes ~30 seconds to start. Wait and retry, or check:

```bash
docker-compose ps  # Should show "healthy"
```

## Documentation

### Core Documentation

- [DEVELOPMENT_GUIDE.md](./DEVELOPMENT_GUIDE.md) - Comprehensive dev guide (architecture, setup, workflows, extending)
- [PROJECT_MAP.md](./docs/PROJECT_MAP.md) - Code structure and file locations
- [SECURITY.md](./SECURITY.md) - Security policy

### Detailed References

- [SETUP.md](./docs/SETUP.md) - Detailed setup instructions
- [TESTING.md](./docs/TESTING.md) - Testing guide
- [DEPENDENCIES.md](./docs/DEPENDENCIES.md) - Dependency management
- [MCP_SERVER_GUIDE.md](./docs/MCP_SERVER_GUIDE.md) - MCP server setup and usage
- [PHASE_7_API_SPEC.md](./docs/PHASE_7_API_SPEC.md) - Backend API reference
- [PHASE_8_LANGGRAPH_INTEGRATION.md](./docs/PHASE_8_LANGGRAPH_INTEGRATION.md) - LangGraph integration guide

## Current Status

### ✅ Implemented

- Backend API with CRUD operations
- Frontend UI with node creation
- Graph database (Neo4j) integration
- Relationship management
- Interactive graph visualization
- Activity feed with processing status
- AI-powered relationship discovery (LangChain + LangGraph)
- Vector embeddings and similarity search
- Automatic connection suggestions based on semantic similarity
- Comprehensive test suite
- Security scanning (Gitleaks)

### 🚧 Coming Soon

- LLM-powered graph operations (find related, summarize, web search, merge nodes)
- Natural language tool interface for graph manipulation
- User confirmation system for destructive operations
- Real-time WebSocket updates
- Advanced search and filtering
- User authentication
- Graph export/import
- MCP server for Claude Desktop integration
- Chrome extension for web capture

## License

MIT License - See [LICENSE](./LICENSE) for details.

## Contributing

1. Install git hooks: `./scripts/install-hooks.sh`
2. Create a feature branch
3. Make changes and run tests
4. Submit a pull request

See [SECURITY.md](./SECURITY.md) for security guidelines.
