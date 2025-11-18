# Research Connection Graph

A web-based research application that helps you discover meaningful connections between sources, observations, hypotheses, and concepts using graph database technology.

## Features

✅ **Node Management** - Create and manage observations, sources, and hypotheses  
✅ **Relationship Tracking** - Connect nodes with typed relationships (SUPPORTS, CONTRADICTS, RELATES_TO, etc.)  
✅ **Interactive Graph Visualization** - Visualize your knowledge graph with Cytoscape.js (zoom, pan, click nodes)  
✅ **RESTful API** - Complete API for programmatic access  
✅ **Node Details Panel** - Click nodes to view details  
✅ **Real-time Updates** - Activity feed ready for live updates (WebSocket integration coming soon)  

## Quickstart

### Option 1: Automated Setup (Recommended)

**One command setup:**

```bash
# Linux/Mac/WSL
./scripts/setup.sh

# Windows (PowerShell)
.\scripts\setup.ps1
```

This will:
- ✅ Start Docker services
- ✅ Set up Python virtual environment
- ✅ Install all dependencies
- ✅ Create and configure `.env` files
- ✅ Initialize Neo4j database

**Then start the app:**

```bash
# Linux/Mac/WSL
./scripts/start.sh

# Or manually in two terminals:
# Terminal 1: cd backend && source .venv/bin/activate && uvicorn app.main:app --reload
# Terminal 2: cd frontend && npm run dev
```

### Option 2: Manual Setup

1. **Start databases:**
   ```bash
   docker-compose up -d
   ```

2. **Start backend** (in `backend/` directory):
   ```bash
   uv venv  # Creates .venv (uv's default)
   source .venv/bin/activate
   uv pip install -r requirements.txt
   cp .env.example .env  # Edit .env with your SECRET_KEY
   uvicorn app.main:app --reload
   ```

3. **Start frontend** (in `frontend/` directory):
   ```bash
   npm install
   npm run dev
   ```

**Done!** Open http://localhost:5173 to see your graph.

> **First time?** See [Quick Start](#quick-start) below for detailed setup including Neo4j initialization.

## Quick Start

### Prerequisites

- **Python 3.11+** (check with `python3 --version`)
- **Node.js 18+** (check with `node --version`)
- **Docker & Docker Compose** (check with `docker --version`)

### 1. Start Database Services

```bash
# Start Neo4j and Redis
docker-compose up -d

# Verify services are running
docker-compose ps
```

All services should show "healthy" status. This may take 30-60 seconds on first run.

### 2. Set Up Backend

```bash
cd backend

# Create virtual environment (uv defaults to .venv)
uv venv

# Activate virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv pip install -r requirements.txt

# Create .env file (if not exists)
cp .env.example .env

# Generate a secure SECRET_KEY
python3 -c "import secrets; print(secrets.token_urlsafe(32))"
# Copy the output and update SECRET_KEY in .env

# Start the backend server
uvicorn app.main:app --reload
```

The API will be available at:
- **API**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

### 3. Set Up Frontend

```bash
cd frontend

# Install dependencies
npm install

# Create .env file (if not exists)
cp .env.example .env

# Start development server
npm run dev
```

The frontend will be available at http://localhost:5173

### 4. Initialize Neo4j (First Time Only)

After Neo4j is running, initialize indexes and constraints:

```bash
# Option 1: Using Docker (recommended)
docker exec research-graph-neo4j cypher-shell -u neo4j -p research_graph_password < docker/neo4j/init.cypher

# Option 2: Using Neo4j Browser
# Open http://localhost:7474
# Login: neo4j / research_graph_password
# Copy and paste contents of docker/neo4j/init.cypher
```

## Usage

### Creating Nodes

#### Via Web Interface

1. Open http://localhost:5173
2. Click **"Add Node"** button in the header
3. Select node type (Observation, Source, Hypothesis, etc.)
4. Fill in the form:
   - **Observation**: Enter text and set confidence level
   - **Source**: Enter title, optional URL, and source type
   - **Hypothesis**: Enter claim and status
5. Click **"Create Node"**

#### Via API

```bash
# Create an observation
curl -X POST "http://localhost:8000/api/v1/nodes/observations" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "I observed that X leads to Y",
    "confidence": 0.8,
    "concept_names": ["X", "Y"]
  }'

# Create a source
curl -X POST "http://localhost:8000/api/v1/nodes/sources" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Research Paper Title",
    "url": "https://example.com/paper",
    "source_type": "paper"
  }'

# Create a hypothesis
curl -X POST "http://localhost:8000/api/v1/nodes/hypotheses" \
  -H "Content-Type: application/json" \
  -d '{
    "claim": "Hypothesis about the relationship",
    "status": "proposed"
  }'
```

### Viewing Nodes

#### Via Web Interface

- **Graph Visualization**: Nodes and relationships are displayed in an interactive graph
  - **Zoom**: Use mouse wheel or pinch gesture
  - **Pan**: Click and drag the background
  - **Select Node**: Click on any node to view its details in the side panel
  - **Node Colors**: 
    - 🔵 Blue (circle) = Observation
    - 🟢 Green (diamond) = Hypothesis
    - 🟡 Yellow (rectangle) = Source
    - 🟣 Purple (hexagon) = Concept
    - 🔴 Red (rounded rectangle) = Entity
  - **Relationship Colors**:
    - Green = SUPPORTS
    - Red (dashed) = CONTRADICTS
    - Gray = RELATES_TO
  - **Controls**: Use "Fit" button to zoom to all nodes, "Reset" to reset view

#### Via API

```bash
# Get all observations
curl "http://localhost:8000/api/v1/nodes/observations"

# Get a specific observation
curl "http://localhost:8000/api/v1/nodes/observations/{node_id}"

# Get full graph
curl "http://localhost:8000/api/v1/graph/full"
```

### Creating Relationships

#### Via API

```bash
# Create a relationship between two nodes
curl -X POST "http://localhost:8000/api/v1/nodes/relationships" \
  -H "Content-Type: application/json" \
  -d '{
    "from_id": "node-id-1",
    "to_id": "node-id-2",
    "relationship_type": "SUPPORTS",
    "confidence": 0.9,
    "notes": "Optional notes about the relationship"
  }'
```

Available relationship types:
- `SUPPORTS` - One node supports another
- `CONTRADICTS` - One node contradicts another
- `RELATES_TO` - General relationship
- `OBSERVED_IN` - Observation made in a source
- `DISCUSSES` - Source discusses a concept
- `EXTRACTED_FROM` - Extracted from a source
- `DERIVED_FROM` - Derived from another node

### Querying Connections

```bash
# Get all connections for a node (within 2 hops)
curl "http://localhost:8000/api/v1/nodes/{node_id}/connections?max_depth=2"
```

## API Documentation

Full interactive API documentation is available at:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Project Structure

```
thoughtlab/
├── backend/                 # Python FastAPI backend
│   ├── app/
│   │   ├── api/routes/     # API endpoints
│   │   ├── core/           # Configuration
│   │   ├── db/             # Database connections
│   │   ├── models/         # Pydantic models
│   │   ├── services/       # Business logic
│   │   └── main.py        # FastAPI app
│   ├── tests/              # Backend tests
│   └── requirements.txt    # Python dependencies
│
├── frontend/               # React TypeScript frontend
│   ├── src/
│   │   ├── components/    # React components
│   │   ├── services/      # API client
│   │   ├── types/        # TypeScript types
│   │   └── App.tsx       # Main component
│   └── package.json       # Node dependencies
│
├── docker-compose.yml      # Database services
└── docs/                  # Project documentation
```

## Testing

### Backend Tests

```bash
cd backend
source .venv/bin/activate
pytest

# With coverage
pytest --cov=app --cov-report=html
```

### Frontend Tests

```bash
cd frontend
npm test

# With UI
npm run test:ui

# With coverage
npm run test:coverage
```

See [TESTING.md](./docs/TESTING.md) for detailed testing information.

## Configuration

### Backend Environment Variables

Edit `backend/.env`:

```bash
# Database connections (match docker-compose.yml)
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=research_graph_password

# Optional: PostgreSQL (scaffolded; not required by default)
DATABASE_URL=postgresql+asyncpg://research_user:research_db_password@localhost/research_graph
REDIS_URL=redis://localhost:6379

# Security
SECRET_KEY=your-secret-key-here

# LLM (optional, for future features)
OPENAI_API_KEY=sk-your-key-here
```

### Frontend Environment Variables

Edit `frontend/.env`:

```bash
VITE_API_URL=http://localhost:8000/api/v1
VITE_WS_URL=ws://localhost:8000/ws
```

## Troubleshooting

### Backend Issues

**Port already in use:**
```bash
# Find what's using port 8000
lsof -i :8000
# Kill it or change port in uvicorn command
```

**Database connection errors:**
- Ensure Docker services are running: `docker-compose ps`
- Check `.env` file has correct credentials
- Wait 30-60 seconds after starting Docker services

**Import errors:**
- Make sure virtual environment is activated: `source .venv/bin/activate`
- Reinstall dependencies: `uv pip install -r requirements.txt`

### Frontend Issues

**Module not found:**
- Run `npm install` to install dependencies
- Check Node.js version: `node --version` (should be 18+)

**API connection errors:**
- Verify backend is running: http://localhost:8000/health
- Check `frontend/.env` has correct `VITE_API_URL`
- Check browser console for CORS errors

**Build errors:**
```bash
# Clear cache and reinstall
rm -rf node_modules package-lock.json
npm install
```

### Docker Issues

**Services not starting:**
```bash
# Check logs
docker-compose logs neo4j
docker-compose logs redis

# Reset if needed (WARNING: deletes data)
docker-compose down -v
docker-compose up -d
```

**Neo4j connection refused:**
- Wait longer (Neo4j takes ~30 seconds to start)
- Verify credentials match docker-compose.yml
- Check port 7687 is not blocked

## Development Workflow

1. **Start databases** (one time per session):
   ```bash
   docker-compose up -d
   ```

2. **Start backend** (Terminal 1):
   ```bash
   cd backend
   source .venv/bin/activate
   uvicorn app.main:app --reload
   ```

3. **Start frontend** (Terminal 2):
   ```bash
   cd frontend
   npm run dev
   ```

4. **Make changes:**
   - Backend: Auto-reloads with `--reload` flag
   - Frontend: Hot module replacement via Vite

## Current Status

### ✅ Implemented

- Backend API with CRUD operations
- Frontend UI with node creation
- Graph database (Neo4j) integration
- Relationship management
- Health checks
- Comprehensive test suite

### 🚧 Coming Soon

- Real-time WebSocket updates
- LLM-powered connection suggestions
- Advanced search and filtering
- User authentication
- Graph export/import
- Custom node styling

## Documentation

- [Project Overview](./docs/01-PROJECT_OVERVIEW.md)
- [Architecture Decisions](./docs/02-ARCHITECTURE_DECISIONS.md)
- [Development Roadmap](./docs/03-DEVELOPMENT_ROADMAP.md)
- [Technical Setup Guide](./docs/04-TECHNICAL_SETUP_GUIDE.md)
- [Testing Guide](./docs/TESTING.md)
- [Backend Setup](./docs/BACKEND_SETUP.md)
- [Frontend Setup](./docs/FRONTEND_SETUP.md)
- [Python uv Setup](./docs/UV_SETUP.md)

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]

## Support

For issues and questions, please check:
1. This README
2. Documentation in `docs/` directory
3. API documentation at http://localhost:8000/docs


