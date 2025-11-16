# Development Roadmap

This document outlines the step-by-step development plan for the Research Connection Graph application, organized into phases with clear milestones and checkpoints.

---

## Phase 0: Environment Setup (Days 1-2)

**Objective:** Get all development tools and services running locally.

### Step 0.1: Install Prerequisites

**Instructions:**
1. Install Python 3.11+ (pyenv recommended for version management)
2. Install Node.js 18+ (nvm recommended)
3. Install Docker and Docker Compose
4. Install Git
5. Install a code editor (VS Code recommended with Python and TypeScript extensions)

**Verification:**
```bash
python --version  # Should be 3.11+
node --version    # Should be 18+
docker --version
docker-compose --version
```

### Step 0.2: Create Project Structure

**Instructions:**
```bash
mkdir research-graph
cd research-graph
git init

# Create directory structure
mkdir -p backend/app/{api/routes,core,db,models,services,workers}
mkdir -p frontend/src/{components,hooks,services,stores,types}
mkdir -p docker/neo4j
mkdir docs

# Create initial files
touch backend/requirements.txt
touch backend/app/main.py
touch frontend/package.json
touch docker-compose.yml
touch .gitignore
touch README.md
```

**Rationale:** Separating backend and frontend from the start makes the codebase easier to navigate. The directory structure mirrors our architecture decisions.

### Step 0.3: Set Up Docker Compose for Databases

**Instructions:**
Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  neo4j:
    image: neo4j:5.15-community
    container_name: research-graph-neo4j
    ports:
      - "7474:7474"  # Browser UI
      - "7687:7687"  # Bolt protocol
    environment:
      - NEO4J_AUTH=neo4j/your_password_here
      - NEO4J_PLUGINS=["apoc", "graph-data-science"]
    volumes:
      - neo4j_data:/data
      - neo4j_logs:/logs

  postgres:
    image: postgres:15
    container_name: research-graph-postgres
    ports:
      - "5432:5432"
    environment:
      - POSTGRES_USER=research_user
      - POSTGRES_PASSWORD=your_password_here
      - POSTGRES_DB=research_graph
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    container_name: research-graph-redis
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

volumes:
  neo4j_data:
  postgres_data:
  redis_data:
```

**Run:**
```bash
docker-compose up -d
```

**Verification:**
- Neo4j Browser: http://localhost:7474 (login with neo4j/your_password_here)
- PostgreSQL: `psql -h localhost -U research_user -d research_graph`
- Redis: `redis-cli ping` should return `PONG`

**Rationale:** Using Docker ensures consistent environments across development machines and simplifies deployment later. All three services are needed from the start.

### Step 0.4: Set Up Python Virtual Environment

**Instructions:**
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install initial dependencies
pip install fastapi uvicorn[standard] python-dotenv

# Create requirements.txt
pip freeze > requirements.txt
```

**Create `backend/.env`:**
```
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password_here

DATABASE_URL=postgresql+asyncpg://research_user:your_password_here@localhost/research_graph

REDIS_URL=redis://localhost:6379

SECRET_KEY=generate_a_random_secret_key_here
```

**Rationale:** Virtual environment isolates dependencies. Environment variables keep secrets out of code.

### Step 0.5: Set Up Frontend Project

**Instructions:**
```bash
cd frontend
npm create vite@latest . -- --template react-ts

# Install dependencies
npm install

# Install additional packages we'll need
npm install @tanstack/react-query zustand axios
npm install -D tailwindcss postcss autoprefixer
npx tailwindcss init -p
```

**Configure Tailwind in `tailwind.config.js`:**
```javascript
/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {},
  },
  plugins: [],
}
```

**Verification:**
```bash
npm run dev  # Should start dev server at http://localhost:5173
```

**Rationale:** Vite provides fast development experience. Installing core packages now avoids context-switching later.

### Phase 0 Checkpoint

✅ All three databases running in Docker  
✅ Can access Neo4j Browser UI  
✅ Python virtual environment active  
✅ Frontend dev server running  
✅ Basic project structure in place  

---

## Phase 1: Backend Foundation (Days 3-7)

**Objective:** Create basic FastAPI application with database connections and health checks.

### Step 1.1: FastAPI Application Skeleton

**Instructions:**

Create `backend/app/main.py`:
```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Research Connection Graph API",
    description="API for managing research knowledge graphs",
    version="0.1.0"
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Research Graph API", "status": "running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
```

**Run:**
```bash
cd backend
uvicorn app.main:app --reload
```

**Verification:**
- http://localhost:8000 returns JSON
- http://localhost:8000/docs shows Swagger UI
- http://localhost:8000/health returns healthy status

**Rationale:** Start simple, verify FastAPI is working before adding complexity.

### Step 1.2: Database Connection Modules

**Install additional dependencies:**
```bash
pip install neo4j asyncpg sqlalchemy[asyncio] redis
```

**Create `backend/app/core/config.py`:**
```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    neo4j_uri: str
    neo4j_user: str
    neo4j_password: str
    database_url: str
    redis_url: str
    secret_key: str
    
    class Config:
        env_file = ".env"

settings = Settings()
```

**Create `backend/app/db/neo4j.py`:**
```python
from neo4j import AsyncGraphDatabase
from app.core.config import settings

class Neo4jConnection:
    def __init__(self):
        self.driver = None
    
    async def connect(self):
        self.driver = AsyncGraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password)
        )
        # Verify connection
        async with self.driver.session() as session:
            await session.run("RETURN 1")
    
    async def disconnect(self):
        if self.driver:
            await self.driver.close()
    
    def get_session(self):
        return self.driver.session()

neo4j_conn = Neo4jConnection()
```

**Create `backend/app/db/postgres.py`:**
```python
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from app.core.config import settings

engine = create_async_engine(settings.database_url, echo=True)
async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
Base = declarative_base()

async def get_db():
    async with async_session() as session:
        yield session
```

**Update `backend/app/main.py` to include startup/shutdown:**
```python
from contextlib import asynccontextmanager
from app.db.neo4j import neo4j_conn

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await neo4j_conn.connect()
    yield
    # Shutdown
    await neo4j_conn.disconnect()

app = FastAPI(
    title="Research Connection Graph API",
    lifespan=lifespan,
    # ... rest of config
)
```

**Verification:**
- Server starts without errors
- No connection errors in logs
- Health check still works

**Rationale:** Connection pooling and proper lifecycle management prevent resource leaks and connection issues.

### Step 1.3: Health Check with Database Status

**Update health check to verify all connections:**
```python
@app.get("/health")
async def health_check():
    health_status = {
        "api": "healthy",
        "neo4j": "unknown",
        "postgres": "unknown",
        "redis": "unknown"
    }
    
    # Check Neo4j
    try:
        async with neo4j_conn.get_session() as session:
            await session.run("RETURN 1")
        health_status["neo4j"] = "healthy"
    except Exception as e:
        health_status["neo4j"] = f"unhealthy: {str(e)}"
    
    # Check Postgres (implement similar)
    # Check Redis (implement similar)
    
    overall = "healthy" if all(
        v == "healthy" for v in health_status.values()
    ) else "degraded"
    
    return {"status": overall, "services": health_status}
```

**Rationale:** Comprehensive health checks catch configuration issues early and help with deployment monitoring.

### Step 1.4: Define Pydantic Models

**Create `backend/app/models/nodes.py`:**
```python
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, List
from enum import Enum

class NodeType(str, Enum):
    SOURCE = "source"
    OBSERVATION = "observation"
    HYPOTHESIS = "hypothesis"
    CONCEPT = "concept"
    ENTITY = "entity"

class RelationshipType(str, Enum):
    SUPPORTS = "SUPPORTS"
    CONTRADICTS = "CONTRADICTS"
    RELATES_TO = "RELATES_TO"
    OBSERVED_IN = "OBSERVED_IN"
    DISCUSSES = "DISCUSSES"
    EXTRACTED_FROM = "EXTRACTED_FROM"
    DERIVED_FROM = "DERIVED_FROM"

class NodeBase(BaseModel):
    """Base properties for all nodes"""
    id: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

class ObservationCreate(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000)
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    subject_ids: Optional[List[str]] = None
    concept_names: Optional[List[str]] = None

class ObservationResponse(NodeBase):
    text: str
    confidence: float
    concept_names: List[str] = []
    
class SourceCreate(BaseModel):
    title: str
    url: Optional[str] = None
    source_type: str  # paper, forum, article, etc.
    content: Optional[str] = None
    published_date: Optional[datetime] = None

class HypothesisCreate(BaseModel):
    claim: str
    supporting_evidence_ids: Optional[List[str]] = None
    status: str = "proposed"

class RelationshipCreate(BaseModel):
    from_id: str
    to_id: str
    relationship_type: RelationshipType
    confidence: Optional[float] = None
    notes: Optional[str] = None
```

**Rationale:** Pydantic models provide validation, serialization, and documentation automatically. Define these early to establish the data contracts.

### Step 1.5: Basic Neo4j CRUD Operations

**Create `backend/app/services/graph_service.py`:**
```python
from app.db.neo4j import neo4j_conn
from app.models.nodes import ObservationCreate, ObservationResponse
from datetime import datetime
import uuid

class GraphService:
    
    async def create_observation(self, data: ObservationCreate) -> str:
        """Create an observation node, return its ID"""
        node_id = str(uuid.uuid4())
        
        query = """
        CREATE (o:Observation {
            id: $id,
            text: $text,
            confidence: $confidence,
            created_at: datetime(),
            concept_names: $concepts
        })
        RETURN o.id as id
        """
        
        async with neo4j_conn.get_session() as session:
            result = await session.run(
                query,
                id=node_id,
                text=data.text,
                confidence=data.confidence,
                concepts=data.concept_names or []
            )
            record = await result.single()
            return record["id"]
    
    async def get_observation(self, node_id: str) -> dict:
        """Fetch a single observation by ID"""
        query = """
        MATCH (o:Observation {id: $id})
        RETURN o
        """
        
        async with neo4j_conn.get_session() as session:
            result = await session.run(query, id=node_id)
            record = await result.single()
            if record:
                return dict(record["o"])
            return None
    
    async def create_relationship(
        self, 
        from_id: str, 
        to_id: str, 
        rel_type: str,
        properties: dict = None
    ) -> bool:
        """Create a relationship between two nodes"""
        query = f"""
        MATCH (a {{id: $from_id}})
        MATCH (b {{id: $to_id}})
        CREATE (a)-[r:{rel_type} $props]->(b)
        RETURN r
        """
        
        async with neo4j_conn.get_session() as session:
            result = await session.run(
                query,
                from_id=from_id,
                to_id=to_id,
                props=properties or {}
            )
            return await result.single() is not None
    
    async def get_node_connections(self, node_id: str, max_depth: int = 2):
        """Get all nodes connected to a given node within max_depth hops"""
        query = """
        MATCH path = (n {id: $id})-[*1..$depth]-(connected)
        RETURN connected, relationships(path) as rels
        LIMIT 100
        """
        
        async with neo4j_conn.get_session() as session:
            result = await session.run(query, id=node_id, depth=max_depth)
            connections = []
            async for record in result:
                connections.append({
                    "node": dict(record["connected"]),
                    "relationships": [dict(r) for r in record["rels"]]
                })
            return connections

graph_service = GraphService()
```

**Rationale:** Encapsulating database operations in a service layer keeps routes clean and makes testing easier.

### Step 1.6: Create API Routes

**Create `backend/app/api/routes/nodes.py`:**
```python
from fastapi import APIRouter, HTTPException
from app.models.nodes import (
    ObservationCreate, 
    ObservationResponse,
    RelationshipCreate
)
from app.services.graph_service import graph_service

router = APIRouter(prefix="/nodes", tags=["nodes"])

@router.post("/observations", response_model=dict)
async def create_observation(data: ObservationCreate):
    """Create a new observation node"""
    node_id = await graph_service.create_observation(data)
    return {"id": node_id, "message": "Observation created"}

@router.get("/observations/{node_id}")
async def get_observation(node_id: str):
    """Get an observation by ID"""
    observation = await graph_service.get_observation(node_id)
    if not observation:
        raise HTTPException(status_code=404, detail="Observation not found")
    return observation

@router.get("/{node_id}/connections")
async def get_connections(node_id: str, max_depth: int = 2):
    """Get all connections for a node"""
    connections = await graph_service.get_node_connections(node_id, max_depth)
    return {"node_id": node_id, "connections": connections}

@router.post("/relationships")
async def create_relationship(data: RelationshipCreate):
    """Create a relationship between two nodes"""
    success = await graph_service.create_relationship(
        data.from_id,
        data.to_id,
        data.relationship_type.value,
        {"confidence": data.confidence, "notes": data.notes}
    )
    if not success:
        raise HTTPException(status_code=400, detail="Could not create relationship")
    return {"message": "Relationship created"}
```

**Register routes in `main.py`:**
```python
from app.api.routes import nodes

app.include_router(nodes.router, prefix="/api/v1")
```

**Verification:**
- Test via Swagger UI at /docs
- Create an observation, retrieve it
- Create another node, create relationship between them
- Query connections

**Rationale:** RESTful routes with clear naming. API versioning (v1) allows future changes without breaking clients.

### Phase 1 Checkpoint

✅ FastAPI server running with auto-reload  
✅ Connected to Neo4j, Postgres, Redis  
✅ Health check reports all services  
✅ Can create observations via API  
✅ Can retrieve observations via API  
✅ Can create relationships via API  
✅ API documentation auto-generated  

---

## Phase 2: Frontend Foundation (Days 8-12)

**Objective:** Build basic React app that can communicate with the backend and display data.

### Step 2.1: Configure API Client

**Create `frontend/src/services/api.ts`:**
```typescript
import axios from 'axios';

const api = axios.create({
  baseURL: 'http://localhost:8000/api/v1',
  headers: {
    'Content-Type': 'application/json',
  },
});

export interface Observation {
  id: string;
  text: string;
  confidence: number;
  concept_names: string[];
  created_at: string;
}

export interface CreateObservationData {
  text: string;
  confidence?: number;
  subject_ids?: string[];
  concept_names?: string[];
}

export const nodesApi = {
  createObservation: (data: CreateObservationData) =>
    api.post('/nodes/observations', data),
  
  getObservation: (id: string) =>
    api.get<Observation>(`/nodes/observations/${id}`),
  
  getConnections: (id: string, maxDepth: number = 2) =>
    api.get(`/nodes/${id}/connections`, { params: { max_depth: maxDepth } }),
};

export default api;
```

**Rationale:** Centralized API client with TypeScript interfaces ensures type safety and makes refactoring easier.

### Step 2.2: Set Up React Query

**Create `frontend/src/main.tsx`:**
```typescript
import React from 'react';
import ReactDOM from 'react-dom/client';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import App from './App';
import './index.css';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 1000 * 60, // 1 minute
      retry: 1,
    },
  },
});

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      <App />
    </QueryClientProvider>
  </React.StrictMode>
);
```

**Rationale:** React Query handles caching, refetching, and loading states automatically. Less boilerplate than manual fetch + useEffect.

### Step 2.3: Basic Layout Structure

**Create `frontend/src/App.tsx`:**
```typescript
import { useState } from 'react';
import GraphVisualizer from './components/GraphVisualizer';
import ActivityFeed from './components/ActivityFeed';
import CreateNodeModal from './components/CreateNodeModal';

function App() {
  const [isCreateModalOpen, setIsCreateModalOpen] = useState(false);

  return (
    <div className="h-screen flex flex-col bg-gray-100">
      {/* Header */}
      <header className="bg-white shadow-sm border-b px-6 py-4 flex justify-between items-center">
        <h1 className="text-xl font-semibold text-gray-800">
          Research Connection Graph
        </h1>
        <button
          onClick={() => setIsCreateModalOpen(true)}
          className="bg-blue-600 text-white px-4 py-2 rounded-md hover:bg-blue-700 flex items-center gap-2"
        >
          <span className="text-lg">+</span>
          Add Node
        </button>
      </header>

      {/* Main Content */}
      <div className="flex-1 flex overflow-hidden">
        {/* Graph Visualizer (main area) */}
        <main className="flex-1 p-4">
          <GraphVisualizer />
        </main>

        {/* Activity Feed (sidebar) */}
        <aside className="w-80 bg-white border-l shadow-sm overflow-y-auto">
          <ActivityFeed />
        </aside>
      </div>

      {/* Create Node Modal */}
      {isCreateModalOpen && (
        <CreateNodeModal onClose={() => setIsCreateModalOpen(false)} />
      )}
    </div>
  );
}

export default App;
```

**Rationale:** Layout matches our planned UI. Flexbox provides responsive sizing. Component separation keeps code organized.

### Step 2.4: Placeholder Components

**Create `frontend/src/components/GraphVisualizer.tsx`:**
```typescript
export default function GraphVisualizer() {
  return (
    <div className="h-full bg-white rounded-lg shadow-sm border flex items-center justify-center">
      <p className="text-gray-500">Graph visualization will appear here</p>
    </div>
  );
}
```

**Create `frontend/src/components/ActivityFeed.tsx`:**
```typescript
export default function ActivityFeed() {
  return (
    <div className="p-4">
      <h2 className="font-semibold text-gray-800 mb-4">Activity Feed</h2>
      <div className="space-y-3">
        <div className="text-sm text-gray-500">No activity yet</div>
      </div>
    </div>
  );
}
```

**Create `frontend/src/components/CreateNodeModal.tsx`:**
```typescript
import { useState } from 'react';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { nodesApi } from '../services/api';

interface Props {
  onClose: () => void;
}

export default function CreateNodeModal({ onClose }: Props) {
  const [nodeType, setNodeType] = useState('observation');
  const [text, setText] = useState('');
  const [confidence, setConfidence] = useState(0.8);
  
  const queryClient = useQueryClient();
  
  const createMutation = useMutation({
    mutationFn: nodesApi.createObservation,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['graph'] });
      onClose();
    },
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (nodeType === 'observation') {
      createMutation.mutate({ text, confidence });
    }
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg shadow-xl w-full max-w-lg">
        <div className="px-6 py-4 border-b flex justify-between items-center">
          <h2 className="text-lg font-semibold">Create New Node</h2>
          <button onClick={onClose} className="text-gray-500 hover:text-gray-700">
            ✕
          </button>
        </div>
        
        <form onSubmit={handleSubmit} className="p-6 space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Node Type
            </label>
            <select
              value={nodeType}
              onChange={(e) => setNodeType(e.target.value)}
              className="w-full border rounded-md px-3 py-2"
            >
              <option value="observation">Observation</option>
              <option value="hypothesis">Hypothesis</option>
              <option value="source">Source</option>
              <option value="entity">Entity</option>
              <option value="concept">Concept</option>
            </select>
          </div>
          
          {nodeType === 'observation' && (
            <>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Observation Text
                </label>
                <textarea
                  value={text}
                  onChange={(e) => setText(e.target.value)}
                  className="w-full border rounded-md px-3 py-2 h-32"
                  placeholder="Describe what you observed..."
                  required
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Confidence: {(confidence * 100).toFixed(0)}%
                </label>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.05"
                  value={confidence}
                  onChange={(e) => setConfidence(parseFloat(e.target.value))}
                  className="w-full"
                />
              </div>
            </>
          )}
          
          <div className="flex justify-end gap-3 pt-4">
            <button
              type="button"
              onClick={onClose}
              className="px-4 py-2 text-gray-700 hover:bg-gray-100 rounded-md"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={createMutation.isPending}
              className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50"
            >
              {createMutation.isPending ? 'Creating...' : 'Create Node'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
```

**Verification:**
- Frontend displays layout correctly
- Modal opens and closes
- Form submission hits backend API
- Check backend logs for incoming requests

**Rationale:** Get end-to-end data flow working before adding complexity. React Query handles mutation state (loading, error) automatically.

### Step 2.5: Display Graph Data (Simple List First)

Before integrating Cytoscape.js, verify data fetching works:

**Update `GraphVisualizer.tsx`:**
```typescript
import { useQuery } from '@tanstack/react-query';
import api from '../services/api';

export default function GraphVisualizer() {
  const { data, isLoading, error } = useQuery({
    queryKey: ['graph', 'all-nodes'],
    queryFn: async () => {
      // Temporary: fetch all observations
      const response = await api.get('/nodes/observations/all');
      return response.data;
    },
  });

  if (isLoading) return <div>Loading graph...</div>;
  if (error) return <div>Error loading graph</div>;

  return (
    <div className="h-full bg-white rounded-lg shadow-sm border p-4 overflow-auto">
      <h3 className="font-semibold mb-4">Graph Nodes (Placeholder)</h3>
      {data?.nodes?.length === 0 ? (
        <p className="text-gray-500">No nodes yet. Create one to get started.</p>
      ) : (
        <ul className="space-y-2">
          {data?.nodes?.map((node: any) => (
            <li key={node.id} className="p-2 border rounded">
              <div className="font-medium">{node.id}</div>
              <div className="text-sm text-gray-600">{node.text}</div>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
```

**Add backend endpoint to fetch all nodes:**
```python
@router.get("/observations/all")
async def get_all_observations():
    """Get all observation nodes"""
    query = "MATCH (o:Observation) RETURN o ORDER BY o.created_at DESC LIMIT 100"
    async with neo4j_conn.get_session() as session:
        result = await session.run(query)
        nodes = [dict(record["o"]) async for record in result]
    return {"nodes": nodes}
```

**Rationale:** Verify data round-trip before investing in visualization library integration.

### Phase 2 Checkpoint

✅ React app structure in place  
✅ API client configured with TypeScript types  
✅ React Query managing server state  
✅ Create node modal functional  
✅ Backend receives and processes requests  
✅ Frontend displays data from backend  
✅ Basic styling with Tailwind  

---

## Phase 3: Graph Visualization (Days 13-18)

**Objective:** Integrate Cytoscape.js for interactive graph display.

### Step 3.1: Install Cytoscape.js

```bash
cd frontend
npm install cytoscape react-cytoscapejs
npm install -D @types/cytoscape
```

### Step 3.2: Create Graph Data Fetching Hook

**Create `frontend/src/hooks/useGraphData.ts`:**
```typescript
import { useQuery } from '@tanstack/react-query';
import api from '../services/api';

interface GraphNode {
  id: string;
  label: string;
  type: string;
  properties: Record<string, any>;
}

interface GraphEdge {
  id: string;
  source: string;
  target: string;
  type: string;
}

interface GraphData {
  nodes: GraphNode[];
  edges: GraphEdge[];
}

export function useGraphData() {
  return useQuery({
    queryKey: ['graph', 'full'],
    queryFn: async (): Promise<GraphData> => {
      const response = await api.get('/graph/full');
      return response.data;
    },
  });
}
```

**Add backend endpoint:**
```python
@router.get("/full")
async def get_full_graph():
    """Get entire graph structure for visualization"""
    nodes_query = """
    MATCH (n)
    WHERE n:Observation OR n:Hypothesis OR n:Source OR n:Entity OR n:Concept
    RETURN n, labels(n) as labels
    LIMIT 500
    """
    
    edges_query = """
    MATCH (a)-[r]->(b)
    WHERE (a:Observation OR a:Hypothesis OR a:Source OR a:Entity OR a:Concept)
      AND (b:Observation OR b:Hypothesis OR b:Source OR b:Entity OR b:Concept)
    RETURN a.id as source, b.id as target, type(r) as type, id(r) as edge_id
    LIMIT 1000
    """
    
    async with neo4j_conn.get_session() as session:
        nodes_result = await session.run(nodes_query)
        nodes = []
        async for record in nodes_result:
            node_data = dict(record["n"])
            node_data["type"] = record["labels"][0]  # Primary label
            nodes.append(node_data)
        
        edges_result = await session.run(edges_query)
        edges = []
        async for record in edges_result:
            edges.append({
                "id": str(record["edge_id"]),
                "source": record["source"],
                "target": record["target"],
                "type": record["type"]
            })
    
    return {"nodes": nodes, "edges": edges}
```

### Step 3.3: Implement Cytoscape Visualization

**Update `frontend/src/components/GraphVisualizer.tsx`:**
```typescript
import { useRef, useEffect } from 'react';
import CytoscapeComponent from 'react-cytoscapejs';
import cytoscape from 'cytoscape';
import { useGraphData } from '../hooks/useGraphData';

const cytoscapeStylesheet = [
  {
    selector: 'node',
    style: {
      'label': 'data(label)',
      'text-valign': 'center',
      'text-halign': 'center',
      'background-color': '#666',
      'color': '#fff',
      'font-size': '10px',
      'width': '40px',
      'height': '40px',
    },
  },
  {
    selector: 'node[type="Observation"]',
    style: {
      'background-color': '#3B82F6', // blue
    },
  },
  {
    selector: 'node[type="Hypothesis"]',
    style: {
      'background-color': '#10B981', // green
    },
  },
  {
    selector: 'node[type="Source"]',
    style: {
      'background-color': '#F59E0B', // yellow
    },
  },
  {
    selector: 'node[type="Concept"]',
    style: {
      'background-color': '#8B5CF6', // purple
    },
  },
  {
    selector: 'node[type="Entity"]',
    style: {
      'background-color': '#EF4444', // red
    },
  },
  {
    selector: 'edge',
    style: {
      'width': 2,
      'line-color': '#ccc',
      'target-arrow-color': '#ccc',
      'target-arrow-shape': 'triangle',
      'curve-style': 'bezier',
      'label': 'data(type)',
      'font-size': '8px',
      'text-rotation': 'autorotate',
    },
  },
  {
    selector: 'edge[type="SUPPORTS"]',
    style: {
      'line-color': '#10B981',
      'target-arrow-color': '#10B981',
    },
  },
  {
    selector: 'edge[type="CONTRADICTS"]',
    style: {
      'line-color': '#EF4444',
      'target-arrow-color': '#EF4444',
    },
  },
];

export default function GraphVisualizer() {
  const { data, isLoading, error } = useGraphData();
  const cyRef = useRef<cytoscape.Core | null>(null);

  const elements = data
    ? [
        ...data.nodes.map((node) => ({
          data: {
            id: node.id,
            label: node.text?.substring(0, 20) || node.id.substring(0, 8),
            type: node.type,
            ...node,
          },
        })),
        ...data.edges.map((edge) => ({
          data: {
            id: edge.id,
            source: edge.source,
            target: edge.target,
            type: edge.type,
          },
        })),
      ]
    : [];

  if (isLoading) {
    return (
      <div className="h-full bg-white rounded-lg shadow-sm border flex items-center justify-center">
        <p className="text-gray-500">Loading graph...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="h-full bg-white rounded-lg shadow-sm border flex items-center justify-center">
        <p className="text-red-500">Error loading graph</p>
      </div>
    );
  }

  return (
    <div className="h-full bg-white rounded-lg shadow-sm border">
      <CytoscapeComponent
        elements={elements}
        stylesheet={cytoscapeStylesheet}
        layout={{ name: 'cose', animate: true }}
        style={{ width: '100%', height: '100%' }}
        cy={(cy) => {
          cyRef.current = cy;
          
          // Add click handler
          cy.on('tap', 'node', (event) => {
            const node = event.target;
            console.log('Clicked node:', node.data());
            // TODO: Open node detail panel
          });
        }}
      />
    </div>
  );
}
```

**Rationale:** Cytoscape handles layout algorithms, zooming, panning automatically. Color-coding by node type provides visual clarity.

### Step 3.4: Node Selection and Details Panel

**Create `frontend/src/components/NodeDetailPanel.tsx`:**
```typescript
interface Props {
  nodeId: string;
  onClose: () => void;
}

export default function NodeDetailPanel({ nodeId, onClose }: Props) {
  const { data, isLoading } = useQuery({
    queryKey: ['node', nodeId],
    queryFn: () => nodesApi.getObservation(nodeId),
  });

  const { data: connections } = useQuery({
    queryKey: ['connections', nodeId],
    queryFn: () => nodesApi.getConnections(nodeId),
  });

  return (
    <div className="fixed right-0 top-0 h-full w-96 bg-white shadow-xl border-l z-40">
      <div className="p-4 border-b flex justify-between items-center">
        <h3 className="font-semibold">Node Details</h3>
        <button onClick={onClose}>✕</button>
      </div>
      
      <div className="p-4 overflow-y-auto">
        {isLoading ? (
          <p>Loading...</p>
        ) : (
          <>
            <div className="mb-4">
              <div className="text-sm text-gray-500">ID</div>
              <div className="font-mono text-sm">{nodeId}</div>
            </div>
            
            <div className="mb-4">
              <div className="text-sm text-gray-500">Content</div>
              <div>{data?.text}</div>
            </div>
            
            <div className="mb-4">
              <div className="text-sm text-gray-500">Confidence</div>
              <div>{((data?.confidence || 0) * 100).toFixed(0)}%</div>
            </div>
            
            <div className="mb-4">
              <div className="text-sm text-gray-500">Connections</div>
              <ul className="space-y-1">
                {connections?.connections?.map((conn: any) => (
                  <li key={conn.node.id} className="text-sm">
                    → {conn.node.id.substring(0, 8)}...
                  </li>
                ))}
              </ul>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
```

**Wire up in App.tsx:**
```typescript
const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);

// Pass setter to GraphVisualizer, show panel when selectedNodeId is set
```

### Phase 3 Checkpoint

✅ Cytoscape.js rendering graph  
✅ Nodes colored by type  
✅ Edges showing relationships  
✅ Click on node shows details  
✅ Graph auto-layouts (force-directed)  
✅ Can zoom and pan  

---

## Phase 4: Embedding and Similarity (Days 19-24)

**Objective:** Add embedding generation and vector similarity search.

### Step 4.1: Install Sentence Transformers

```bash
cd backend
pip install sentence-transformers numpy
```

### Step 4.2: Create Embedding Service

**Create `backend/app/services/embedding_service.py`:**
```python
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List

class EmbeddingService:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = None
        self.model_name = model_name
    
    def load_model(self):
        """Load model - call on startup"""
        if self.model is None:
            self.model = SentenceTransformer(self.model_name)
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        if self.model is None:
            self.load_model()
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        if self.model is None:
            self.load_model()
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()
    
    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        a = np.array(vec1)
        b = np.array(vec2)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

embedding_service = EmbeddingService()
```

**Load model on startup (in main.py lifespan):**
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    await neo4j_conn.connect()
    embedding_service.load_model()  # Add this
    yield
    await neo4j_conn.disconnect()
```

**Rationale:** Loading model once on startup avoids repeated loading overhead. Sentence-transformers handles all the complexity.

### Step 4.3: Store Embeddings in Neo4j

Neo4j 5.11+ supports vector indexes. First, create the index:

```cypher
// Run this in Neo4j Browser
CREATE VECTOR INDEX observation_embeddings IF NOT EXISTS
FOR (o:Observation)
ON o.embedding
OPTIONS {
  indexConfig: {
    `vector.dimensions`: 384,
    `vector.similarity_function`: 'cosine'
  }
}
```

**Update observation creation to include embedding:**
```python
async def create_observation(self, data: ObservationCreate) -> str:
    node_id = str(uuid.uuid4())
    
    # Generate embedding
    embedding = embedding_service.generate_embedding(data.text)
    
    query = """
    CREATE (o:Observation {
        id: $id,
        text: $text,
        confidence: $confidence,
        created_at: datetime(),
        embedding: $embedding,
        concept_names: $concepts
    })
    RETURN o.id as id
    """
    
    async with neo4j_conn.get_session() as session:
        result = await session.run(
            query,
            id=node_id,
            text=data.text,
            confidence=data.confidence,
            embedding=embedding,
            concepts=data.concept_names or []
        )
        record = await result.single()
        return record["id"]
```

### Step 4.4: Implement Vector Similarity Search

**Add to graph_service.py:**
```python
async def find_similar_nodes(
    self, 
    text: str, 
    node_type: str = "Observation",
    limit: int = 10
) -> List[dict]:
    """Find nodes with similar embeddings"""
    query_embedding = embedding_service.generate_embedding(text)
    
    query = f"""
    MATCH (n:{node_type})
    WHERE n.embedding IS NOT NULL
    WITH n, vector.similarity.cosine(n.embedding, $embedding) AS score
    WHERE score > 0.5
    RETURN n, score
    ORDER BY score DESC
    LIMIT $limit
    """
    
    async with neo4j_conn.get_session() as session:
        result = await session.run(
            query,
            embedding=query_embedding,
            limit=limit
        )
        similar = []
        async for record in result:
            node_data = dict(record["n"])
            node_data["similarity_score"] = record["score"]
            similar.append(node_data)
        return similar
```

**Add API endpoint:**
```python
@router.post("/search/similar")
async def find_similar(text: str, node_type: str = "Observation", limit: int = 10):
    """Find nodes semantically similar to given text"""
    similar = await graph_service.find_similar_nodes(text, node_type, limit)
    return {"results": similar}
```

### Phase 4 Checkpoint

✅ Embedding model loads on startup  
✅ New nodes get embeddings automatically  
✅ Vector index created in Neo4j  
✅ Can search for similar nodes by text  
✅ Similarity scores returned with results  

---

## Phase 5: LLM Integration (Days 25-32)

**Objective:** Add LLM-powered connection analysis.

### Step 5.1: Install LiteLLM

```bash
pip install litellm
```

### Step 5.2: Create LLM Service

**Create `backend/app/services/llm_service.py`:**
```python
import litellm
from typing import Optional
import json

class LLMService:
    def __init__(self, model: str = "gpt-3.5-turbo"):
        self.model = model
    
    async def analyze_connection(
        self,
        node_a_text: str,
        node_a_type: str,
        node_b_text: str,
        node_b_type: str,
        context: dict = None
    ) -> dict:
        """Analyze potential connection between two nodes"""
        
        prompt = f"""
        You are analyzing potential connections in a research knowledge graph.
        
        Node A ({node_a_type}):
        "{node_a_text}"
        
        Node B ({node_b_type}):
        "{node_b_text}"
        
        Additional context:
        {json.dumps(context or {})}
        
        Analyze if these nodes should be connected. Respond with JSON:
        {{
            "should_connect": true/false,
            "relationship_type": "SUPPORTS" | "CONTRADICTS" | "RELATES_TO" | null,
            "confidence": 0.0-1.0,
            "reasoning": "Brief explanation",
            "decision": "auto_approve" | "needs_review" | "auto_reject"
        }}
        
        Guidelines:
        - auto_approve: Very clear relationship, confidence > 0.85
        - needs_review: Plausible but uncertain, or nuanced
        - auto_reject: No meaningful connection
        
        Respond ONLY with valid JSON.
        """
        
        response = await litellm.acompletion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        return result

llm_service = LLMService()
```

**Configure in .env:**
```
OPENAI_API_KEY=your_key_here
# Or for Anthropic:
# ANTHROPIC_API_KEY=your_key_here
```

### Step 5.3: Create Analysis Worker

**Create `backend/app/workers/connection_analyzer.py`:**
```python
from app.services.graph_service import graph_service
from app.services.llm_service import llm_service
from app.services.embedding_service import embedding_service

async def analyze_new_node(node_id: str, node_type: str, node_text: str):
    """
    Analyze a newly added node for potential connections
    """
    results = {
        "auto_approved": [],
        "needs_review": [],
        "auto_rejected": []
    }
    
    # 1. Find candidate connections via embedding similarity
    candidates = await graph_service.find_similar_nodes(
        node_text, 
        limit=20
    )
    
    # 2. Evaluate each candidate with LLM
    for candidate in candidates:
        if candidate["id"] == node_id:
            continue  # Skip self
        
        analysis = await llm_service.analyze_connection(
            node_a_text=node_text,
            node_a_type=node_type,
            node_b_text=candidate["text"],
            node_b_type=candidate.get("type", "Unknown"),
            context={
                "embedding_similarity": candidate["similarity_score"]
            }
        )
        
        suggestion = {
            "from_id": node_id,
            "to_id": candidate["id"],
            "relationship_type": analysis["relationship_type"],
            "confidence": analysis["confidence"],
            "reasoning": analysis["reasoning"]
        }
        
        if analysis["decision"] == "auto_approve":
            # Create relationship automatically
            await graph_service.create_relationship(
                node_id,
                candidate["id"],
                analysis["relationship_type"],
                {
                    "confidence": analysis["confidence"],
                    "reasoning": analysis["reasoning"],
                    "created_by": "system"
                }
            )
            results["auto_approved"].append(suggestion)
        
        elif analysis["decision"] == "needs_review":
            # Queue for user review
            await queue_for_review(suggestion)
            results["needs_review"].append(suggestion)
        
        else:
            results["auto_rejected"].append(suggestion)
    
    return results
```

### Step 5.4: Background Task Integration

**Using ARQ:**
```bash
pip install arq
```

**Create `backend/app/workers/tasks.py`:**
```python
from arq import create_pool
from arq.connections import RedisSettings
from app.workers.connection_analyzer import analyze_new_node

async def analyze_node_task(ctx, node_id: str, node_type: str, node_text: str):
    """ARQ task for analyzing new nodes"""
    results = await analyze_new_node(node_id, node_type, node_text)
    
    # Broadcast results via WebSocket
    # (implementation depends on your pubsub setup)
    
    return results

class WorkerSettings:
    functions = [analyze_node_task]
    redis_settings = RedisSettings()
```

**Enqueue task when node is created:**
```python
@router.post("/nodes/observations")
async def create_observation(data: ObservationCreate):
    node_id = await graph_service.create_observation(data)
    
    # Enqueue background analysis
    redis = await create_pool(RedisSettings())
    await redis.enqueue_job(
        "analyze_node_task",
        node_id,
        "Observation",
        data.text
    )
    
    return {"id": node_id, "message": "Observation created, analyzing connections..."}
```

**Run worker:**
```bash
arq app.workers.tasks.WorkerSettings
```

### Phase 5 Checkpoint

✅ LLM service configured and working  
✅ Can analyze connection between two texts  
✅ Background worker processes new nodes  
✅ High-confidence connections made automatically  
✅ Medium-confidence queued for review  
✅ Results include reasoning  

---

## Phase 6: Real-Time Updates & Review Queue (Days 33-38)

**Objective:** WebSocket connection for live updates and user review interface.

### Step 6.1: WebSocket Setup

**Add to `backend/app/api/websocket.py`:**
```python
from fastapi import WebSocket, WebSocketDisconnect
from typing import List
import json

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                pass

manager = ConnectionManager()
```

**Add WebSocket route:**
```python
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Handle incoming messages if needed
    except WebSocketDisconnect:
        manager.disconnect(websocket)
```

### Step 6.2: Frontend WebSocket Hook

**Create `frontend/src/hooks/useWebSocket.ts`:**
```typescript
import { useEffect, useRef, useState } from 'react';

interface ActivityEvent {
  type: 'analysis_started' | 'connection_approved' | 'needs_review' | 'analysis_complete';
  data: any;
  timestamp: string;
}

export function useWebSocket() {
  const [events, setEvents] = useState<ActivityEvent[]>([]);
  const [connected, setConnected] = useState(false);
  const ws = useRef<WebSocket | null>(null);

  useEffect(() => {
    ws.current = new WebSocket('ws://localhost:8000/ws');

    ws.current.onopen = () => setConnected(true);
    ws.current.onclose = () => setConnected(false);
    
    ws.current.onmessage = (event) => {
      const data = JSON.parse(event.data);
      setEvents((prev) => [data, ...prev].slice(0, 100)); // Keep last 100
    };

    return () => ws.current?.close();
  }, []);

  return { events, connected };
}
```

### Step 6.3: Update Activity Feed Component

**Update `frontend/src/components/ActivityFeed.tsx`:**
```typescript
import { useWebSocket } from '../hooks/useWebSocket';
import { useState } from 'react';

export default function ActivityFeed() {
  const { events, connected } = useWebSocket();
  const [activeTab, setActiveTab] = useState<'pending' | 'recent'>('pending');
  
  const pendingReviews = events.filter(e => e.type === 'needs_review');
  const recentActivity = events.filter(e => e.type !== 'needs_review');

  return (
    <div className="h-full flex flex-col">
      <div className="p-4 border-b">
        <div className="flex items-center justify-between mb-3">
          <h2 className="font-semibold">Activity</h2>
          <span className={`text-xs ${connected ? 'text-green-500' : 'text-red-500'}`}>
            {connected ? '● Live' : '○ Disconnected'}
          </span>
        </div>
        
        <div className="flex gap-2">
          <button
            onClick={() => setActiveTab('pending')}
            className={`px-3 py-1 text-sm rounded ${
              activeTab === 'pending' ? 'bg-blue-100 text-blue-700' : 'bg-gray-100'
            }`}
          >
            Pending ({pendingReviews.length})
          </button>
          <button
            onClick={() => setActiveTab('recent')}
            className={`px-3 py-1 text-sm rounded ${
              activeTab === 'recent' ? 'bg-blue-100 text-blue-700' : 'bg-gray-100'
            }`}
          >
            Recent
          </button>
        </div>
      </div>
      
      <div className="flex-1 overflow-y-auto p-4 space-y-3">
        {activeTab === 'pending' ? (
          pendingReviews.length === 0 ? (
            <p className="text-sm text-gray-500">No pending reviews</p>
          ) : (
            pendingReviews.map((event, i) => (
              <PendingReviewCard key={i} event={event} />
            ))
          )
        ) : (
          recentActivity.map((event, i) => (
            <ActivityCard key={i} event={event} />
          ))
        )}
      </div>
    </div>
  );
}

function PendingReviewCard({ event }: { event: any }) {
  const handleApprove = async () => {
    // API call to approve connection
  };
  
  const handleReject = async () => {
    // API call to reject connection
  };

  return (
    <div className="border rounded p-3 bg-yellow-50">
      <div className="text-sm font-medium mb-2">Suggested Connection</div>
      <div className="text-xs text-gray-600 mb-2">
        {event.data.reasoning}
      </div>
      <div className="text-xs mb-3">
        Confidence: {(event.data.confidence * 100).toFixed(0)}%
      </div>
      <div className="flex gap-2">
        <button
          onClick={handleApprove}
          className="px-3 py-1 text-xs bg-green-600 text-white rounded hover:bg-green-700"
        >
          Approve
        </button>
        <button
          onClick={handleReject}
          className="px-3 py-1 text-xs bg-red-600 text-white rounded hover:bg-red-700"
        >
          Reject
        </button>
      </div>
    </div>
  );
}

function ActivityCard({ event }: { event: any }) {
  return (
    <div className="border rounded p-3 text-sm">
      <div className="font-medium">{event.type}</div>
      <div className="text-xs text-gray-500">{event.timestamp}</div>
    </div>
  );
}
```

### Phase 6 Checkpoint

✅ WebSocket connection established  
✅ Events broadcast in real-time  
✅ Activity feed shows live updates  
✅ Pending reviews displayed  
✅ Approve/reject actions work  
✅ Connection status indicator  

---

## Phase 7: Feedback Loop & Learning (Days 39-45)

**Objective:** Store feedback data and use it to improve suggestions.

### Step 7.1: Feedback Data Model

**Create PostgreSQL table:**
```sql
CREATE TABLE feedback (
    id SERIAL PRIMARY KEY,
    suggestion_id VARCHAR(255),
    from_node_id VARCHAR(255),
    to_node_id VARCHAR(255),
    relationship_type VARCHAR(50),
    original_confidence FLOAT,
    llm_reasoning TEXT,
    user_action VARCHAR(20),  -- approved, rejected, edited, undone
    user_id INTEGER,
    created_at TIMESTAMP DEFAULT NOW()
);
```

### Step 7.2: Record All Feedback

**Backend endpoints for feedback:**
```python
@router.post("/feedback/approve")
async def approve_suggestion(suggestion_id: str, user_id: int):
    # Create the relationship
    # Record the feedback
    # Return confirmation

@router.post("/feedback/reject")
async def reject_suggestion(suggestion_id: str, user_id: int):
    # Don't create relationship
    # Record the feedback
    # Return confirmation
```

### Step 7.3: Adjust Confidence Based on Feedback

As feedback accumulates, adjust the LLM prompts and thresholds:
- Track approval rate by relationship type
- Track approval rate by confidence range
- Identify patterns in rejections

### Phase 7 Checkpoint

✅ Feedback stored in PostgreSQL  
✅ All user actions recorded  
✅ Can analyze feedback patterns  
✅ Thresholds adjustable based on data  

---

## Future Phases (Outline)

### Phase 8: Natural Language Querying
- LLM interprets questions about the graph
- Generates Cypher queries
- Returns synthesized answers

### Phase 9: User Authentication
- FastAPI-Users integration
- Multi-user support
- Activity attribution

### Phase 10: Polish & Performance
- Error handling
- Loading states
- Performance optimization
- Testing suite

---

## Summary

This roadmap takes you from empty project to functional research connection tool in approximately 45 days of development time. Each phase builds on the previous, with clear checkpoints to verify progress.

Key milestones:
- Day 2: Development environment ready
- Day 7: Backend can create/retrieve nodes
- Day 12: Frontend displays data from backend
- Day 18: Interactive graph visualization working
- Day 24: Semantic similarity search functional
- Day 32: LLM analyzes and suggests connections
- Day 38: Real-time updates and user review
- Day 45: Feedback loop collecting training data

Adjust timeline based on your availability and experience. Each phase can be extended or compressed as needed.
