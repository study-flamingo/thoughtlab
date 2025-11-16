# Technical Setup Guide

This guide provides detailed, step-by-step instructions for setting up the development environment for the Research Connection Graph application.

---

## Prerequisites

### Required Software

| Software | Minimum Version | Installation |
|----------|----------------|--------------|
| Python | 3.11+ | [python.org](https://python.org) or pyenv |
| Node.js | 18+ | [nodejs.org](https://nodejs.org) or nvm |
| Docker | 20+ | [docker.com](https://docker.com/get-started) |
| Docker Compose | 2.0+ | Included with Docker Desktop |
| Git | 2.30+ | [git-scm.com](https://git-scm.com) |

### Recommended Tools

- **VS Code** with extensions:
  - Python (Microsoft)
  - Pylance
  - TypeScript and JavaScript Language Features
  - Tailwind CSS IntelliSense
  - ES7+ React/Redux/React-Native snippets
  - Docker
  - GitLens

- **Database GUIs:**
  - Neo4j Desktop (or use Browser at localhost:7474)
  - pgAdmin or DBeaver for PostgreSQL
  - Redis Insight (optional)

---

## Step 1: Clone or Initialize Repository

```bash
# If starting fresh
mkdir research-graph
cd research-graph
git init

# Create .gitignore
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
venv/
.env

# Node
node_modules/
dist/
.DS_Store

# IDE
.vscode/
.idea/
*.swp
*.swo

# Docker volumes
postgres_data/
neo4j_data/
redis_data/

# Logs
*.log
logs/

# OS
.DS_Store
Thumbs.db
EOF
```

---

## Step 2: Set Up Docker Services

### 2.1 Create Docker Compose File

Create `docker-compose.yml` in project root:

```yaml
version: '3.8'

services:
  neo4j:
    image: neo4j:5.15-community
    container_name: research-graph-neo4j
    restart: unless-stopped
    ports:
      - "7474:7474"  # HTTP (Browser)
      - "7687:7687"  # Bolt protocol
    environment:
      - NEO4J_AUTH=neo4j/research_graph_password
      - NEO4J_PLUGINS=["apoc"]
      - NEO4J_dbms_security_procedures_unrestricted=apoc.*
      - NEO4J_dbms_memory_heap_initial__size=512m
      - NEO4J_dbms_memory_heap_max__size=1G
    volumes:
      - neo4j_data:/data
      - neo4j_logs:/logs
      - neo4j_import:/var/lib/neo4j/import
    healthcheck:
      test: ["CMD", "cypher-shell", "-u", "neo4j", "-p", "research_graph_password", "RETURN 1"]
      interval: 10s
      timeout: 10s
      retries: 10

  postgres:
    image: postgres:15-alpine
    container_name: research-graph-postgres
    restart: unless-stopped
    ports:
      - "5432:5432"
    environment:
      - POSTGRES_USER=research_user
      - POSTGRES_PASSWORD=research_db_password
      - POSTGRES_DB=research_graph
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U research_user -d research_graph"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    container_name: research-graph-redis
    restart: unless-stopped
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

volumes:
  neo4j_data:
  neo4j_logs:
  neo4j_import:
  postgres_data:
  redis_data:
```

### 2.2 Start Services

```bash
# Start all services in background
docker-compose up -d

# Check status
docker-compose ps

# View logs (if issues)
docker-compose logs neo4j
docker-compose logs postgres
docker-compose logs redis

# Wait for services to be healthy
docker-compose ps  # All should show "healthy"
```

### 2.3 Verify Connections

**Neo4j:**
1. Open browser: http://localhost:7474
2. Login with `neo4j` / `research_graph_password`
3. Run: `RETURN 1` to test

**PostgreSQL:**
```bash
# Using psql (if installed)
psql -h localhost -U research_user -d research_graph
# Password: research_db_password

# Or via Docker
docker exec -it research-graph-postgres psql -U research_user -d research_graph
```

**Redis:**
```bash
# Using redis-cli (if installed)
redis-cli ping
# Should return: PONG

# Or via Docker
docker exec -it research-graph-redis redis-cli ping
```

---

## Step 3: Backend Setup

### 3.1 Create Directory Structure

```bash
mkdir -p backend/app/{api/routes,core,db,models,services,workers}
touch backend/app/__init__.py
touch backend/app/api/__init__.py
touch backend/app/api/routes/__init__.py
touch backend/app/core/__init__.py
touch backend/app/db/__init__.py
touch backend/app/models/__init__.py
touch backend/app/services/__init__.py
touch backend/app/workers/__init__.py
```

### 3.2 Set Up Python Environment

```bash
cd backend

# Create virtual environment
python -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate

# Activate (Windows)
# venv\Scripts\activate

# Verify Python version
python --version  # Should be 3.11+
```

### 3.3 Install Dependencies

Create `backend/requirements.txt`:

```txt
# Web Framework
fastapi==0.109.0
uvicorn[standard]==0.27.0
python-multipart==0.0.6

# Database Drivers
neo4j==5.17.0
asyncpg==0.29.0
sqlalchemy[asyncio]==2.0.25
redis==5.0.1

# Configuration
python-dotenv==1.0.1
pydantic-settings==2.1.0

# AI/ML
sentence-transformers==2.3.1
litellm==1.17.9

# Background Tasks
arq==0.25.0

# Utilities
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
numpy==1.26.3

# Development
pytest==8.0.0
pytest-asyncio==0.23.3
httpx==0.26.0
```

Install:

```bash
pip install -r requirements.txt
```

### 3.4 Create Environment File

Create `backend/.env`:

```bash
# Database Connections
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=research_graph_password

DATABASE_URL=postgresql+asyncpg://research_user:research_db_password@localhost/research_graph

REDIS_URL=redis://localhost:6379

# Security
SECRET_KEY=your-secret-key-here-generate-a-random-string

# LLM Configuration (choose one)
OPENAI_API_KEY=sk-your-key-here
# ANTHROPIC_API_KEY=your-key-here
# Or for local: OLLAMA_API_BASE=http://localhost:11434

# Application Settings
DEBUG=true
ENVIRONMENT=development
```

**Generate a secure SECRET_KEY:**
```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

### 3.5 Create Core Application Files

**`backend/app/core/config.py`:**

```python
from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    # Database
    neo4j_uri: str
    neo4j_user: str
    neo4j_password: str
    database_url: str
    redis_url: str
    
    # Security
    secret_key: str
    
    # Application
    debug: bool = False
    environment: str = "development"
    
    # LLM
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    
    class Config:
        env_file = ".env"
        case_sensitive = False

@lru_cache
def get_settings() -> Settings:
    return Settings()

settings = get_settings()
```

**`backend/app/main.py`:**

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Starting up...")
    # Database connections will be initialized here
    yield
    # Shutdown
    print("Shutting down...")

app = FastAPI(
    title="Research Connection Graph API",
    description="API for managing research knowledge graphs with AI-powered connection discovery",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite dev server
        "http://localhost:3000",  # Alternative React port
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "message": "Research Connection Graph API",
        "version": "0.1.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "environment": settings.environment
    }
```

### 3.6 Run Backend Server

```bash
cd backend
source venv/bin/activate

# Run with auto-reload for development
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Verify:**
- http://localhost:8000 → JSON response
- http://localhost:8000/docs → Swagger UI
- http://localhost:8000/redoc → ReDoc documentation
- http://localhost:8000/health → Health check

---

## Step 4: Frontend Setup

### 4.1 Create React Application

```bash
cd ..  # Back to project root
npm create vite@latest frontend -- --template react-ts
cd frontend
```

### 4.2 Install Dependencies

```bash
# Core dependencies
npm install @tanstack/react-query@5 zustand axios

# Graph visualization
npm install cytoscape react-cytoscapejs
npm install -D @types/cytoscape

# UI
npm install -D tailwindcss postcss autoprefixer
npx tailwindcss init -p

# Icons (optional but recommended)
npm install lucide-react
```

### 4.3 Configure Tailwind CSS

Update `frontend/tailwind.config.js`:

```javascript
/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Custom colors for node types
        'node-observation': '#3B82F6',
        'node-hypothesis': '#10B981',
        'node-source': '#F59E0B',
        'node-concept': '#8B5CF6',
        'node-entity': '#EF4444',
      },
    },
  },
  plugins: [],
}
```

Update `frontend/src/index.css`:

```css
@tailwind base;
@tailwind components;
@tailwind utilities;

body {
  margin: 0;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen',
    'Ubuntu', 'Cantarell', 'Fira Sans', 'Droid Sans', 'Helvetica Neue',
    sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

/* Custom scrollbar (optional) */
::-webkit-scrollbar {
  width: 8px;
  height: 8px;
}

::-webkit-scrollbar-track {
  background: #f1f1f1;
}

::-webkit-scrollbar-thumb {
  background: #888;
  border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
  background: #555;
}
```

### 4.4 Create Directory Structure

```bash
cd frontend/src
mkdir -p {components,hooks,services,stores,types,utils}
```

### 4.5 Create Type Definitions

Create `frontend/src/types/graph.ts`:

```typescript
export type NodeType = 'Observation' | 'Hypothesis' | 'Source' | 'Concept' | 'Entity';

export type RelationshipType = 
  | 'SUPPORTS' 
  | 'CONTRADICTS' 
  | 'RELATES_TO' 
  | 'OBSERVED_IN' 
  | 'DISCUSSES';

export interface GraphNode {
  id: string;
  type: NodeType;
  text?: string;
  title?: string;
  confidence?: number;
  concept_names?: string[];
  created_at: string;
  updated_at?: string;
}

export interface GraphEdge {
  id: string;
  source: string;
  target: string;
  type: RelationshipType;
  confidence?: number;
  notes?: string;
}

export interface GraphData {
  nodes: GraphNode[];
  edges: GraphEdge[];
}

export interface CreateObservationData {
  text: string;
  confidence?: number;
  subject_ids?: string[];
  concept_names?: string[];
}

export interface ConnectionSuggestion {
  id: string;
  from_id: string;
  to_id: string;
  relationship_type: RelationshipType;
  confidence: number;
  reasoning: string;
  status: 'pending' | 'approved' | 'rejected';
}

export interface ActivityEvent {
  id: string;
  type: 'analysis_started' | 'connection_approved' | 'needs_review' | 'analysis_complete';
  data: any;
  timestamp: string;
}
```

### 4.6 Configure API Client

Create `frontend/src/services/api.ts`:

```typescript
import axios from 'axios';
import type { GraphData, CreateObservationData, GraphNode } from '../types/graph';

const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL || 'http://localhost:8000/api/v1',
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add request interceptor for auth (when implemented)
api.interceptors.request.use((config) => {
  // const token = localStorage.getItem('token');
  // if (token) {
  //   config.headers.Authorization = `Bearer ${token}`;
  // }
  return config;
});

// Add response interceptor for error handling
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      // Handle unauthorized
    }
    return Promise.reject(error);
  }
);

export const graphApi = {
  getFullGraph: () => api.get<GraphData>('/graph/full'),
  
  createObservation: (data: CreateObservationData) =>
    api.post<{ id: string }>('/nodes/observations', data),
  
  getNode: (id: string) => api.get<GraphNode>(`/nodes/${id}`),
  
  getConnections: (id: string, maxDepth = 2) =>
    api.get(`/nodes/${id}/connections`, { params: { max_depth: maxDepth } }),
  
  createRelationship: (fromId: string, toId: string, type: string) =>
    api.post('/relationships', {
      from_id: fromId,
      to_id: toId,
      relationship_type: type,
    }),
  
  approveSuggestion: (suggestionId: string) =>
    api.post(`/suggestions/${suggestionId}/approve`),
  
  rejectSuggestion: (suggestionId: string) =>
    api.post(`/suggestions/${suggestionId}/reject`),
};

export default api;
```

Create `frontend/.env`:

```bash
VITE_API_URL=http://localhost:8000/api/v1
VITE_WS_URL=ws://localhost:8000/ws
```

### 4.7 Set Up React Query

Update `frontend/src/main.tsx`:

```typescript
import React from 'react';
import ReactDOM from 'react-dom/client';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ReactQueryDevtools } from '@tanstack/react-query-devtools';
import App from './App';
import './index.css';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 1000 * 60, // 1 minute
      retry: 1,
      refetchOnWindowFocus: false,
    },
  },
});

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      <App />
      <ReactQueryDevtools initialIsOpen={false} />
    </QueryClientProvider>
  </React.StrictMode>
);
```

### 4.8 Run Frontend Development Server

```bash
cd frontend
npm run dev
```

**Verify:**
- http://localhost:5173 → React app loads
- Browser console has no errors
- Network tab shows API requests (may 404 initially, that's expected)

---

## Step 5: Create Neo4j Indexes and Constraints

Connect to Neo4j Browser (http://localhost:7474) and run:

```cypher
// Unique constraints on IDs
CREATE CONSTRAINT observation_id IF NOT EXISTS
FOR (o:Observation) REQUIRE o.id IS UNIQUE;

CREATE CONSTRAINT hypothesis_id IF NOT EXISTS
FOR (h:Hypothesis) REQUIRE h.id IS UNIQUE;

CREATE CONSTRAINT source_id IF NOT EXISTS
FOR (s:Source) REQUIRE s.id IS UNIQUE;

CREATE CONSTRAINT concept_id IF NOT EXISTS
FOR (c:Concept) REQUIRE c.id IS UNIQUE;

CREATE CONSTRAINT entity_id IF NOT EXISTS
FOR (e:Entity) REQUIRE e.id IS UNIQUE;

// Index for text search
CREATE TEXT INDEX observation_text IF NOT EXISTS
FOR (o:Observation) ON o.text;

CREATE TEXT INDEX hypothesis_claim IF NOT EXISTS
FOR (h:Hypothesis) ON h.claim;

// Index for temporal queries
CREATE INDEX observation_created IF NOT EXISTS
FOR (o:Observation) ON o.created_at;

// Vector index for embeddings (Neo4j 5.11+)
// Note: This will be created after we have dimension info
// CREATE VECTOR INDEX observation_embeddings IF NOT EXISTS
// FOR (o:Observation) ON o.embedding
// OPTIONS {indexConfig: {`vector.dimensions`: 384, `vector.similarity_function`: 'cosine'}}
```

**Verify indexes:**
```cypher
SHOW INDEXES;
SHOW CONSTRAINTS;
```

---

## Step 6: Create PostgreSQL Schema

Connect to PostgreSQL and run:

```sql
-- Users table (for future auth)
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    hashed_password VARCHAR(255) NOT NULL,
    full_name VARCHAR(255),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Activity log
CREATE TABLE IF NOT EXISTS activity_log (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    action VARCHAR(50) NOT NULL,
    node_id VARCHAR(255),
    node_type VARCHAR(50),
    details JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Connection suggestions (for review queue)
CREATE TABLE IF NOT EXISTS connection_suggestions (
    id SERIAL PRIMARY KEY,
    suggestion_uuid VARCHAR(255) UNIQUE NOT NULL,
    from_node_id VARCHAR(255) NOT NULL,
    to_node_id VARCHAR(255) NOT NULL,
    relationship_type VARCHAR(50) NOT NULL,
    confidence FLOAT NOT NULL,
    reasoning TEXT,
    status VARCHAR(20) DEFAULT 'pending',  -- pending, approved, rejected
    created_by VARCHAR(50) DEFAULT 'system',
    reviewed_by INTEGER REFERENCES users(id),
    created_at TIMESTAMP DEFAULT NOW(),
    reviewed_at TIMESTAMP
);

-- Feedback for learning
CREATE TABLE IF NOT EXISTS feedback (
    id SERIAL PRIMARY KEY,
    suggestion_id INTEGER REFERENCES connection_suggestions(id),
    user_id INTEGER REFERENCES users(id),
    action VARCHAR(20) NOT NULL,  -- approved, rejected, undone, edited
    original_confidence FLOAT,
    reasoning_shown TEXT,
    user_notes TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Indexes for common queries
CREATE INDEX idx_activity_user ON activity_log(user_id);
CREATE INDEX idx_activity_time ON activity_log(created_at DESC);
CREATE INDEX idx_suggestions_status ON connection_suggestions(status);
CREATE INDEX idx_suggestions_created ON connection_suggestions(created_at DESC);
CREATE INDEX idx_feedback_time ON feedback(created_at DESC);
```

---

## Step 7: Verify Complete Setup

### Backend Checklist

```bash
cd backend
source venv/bin/activate

# 1. Run server
uvicorn app.main:app --reload

# 2. Test endpoints
curl http://localhost:8000/
curl http://localhost:8000/health
curl http://localhost:8000/docs  # Opens Swagger UI

# 3. Check database connections (will add specific health checks)
```

### Frontend Checklist

```bash
cd frontend

# 1. Run dev server
npm run dev

# 2. Open http://localhost:5173
# 3. Check browser console for errors
# 4. Verify API calls in Network tab
```

### Docker Services Checklist

```bash
# All healthy?
docker-compose ps

# Neo4j Browser accessible?
# http://localhost:7474

# PostgreSQL accepting connections?
docker exec -it research-graph-postgres psql -U research_user -d research_graph -c "SELECT 1"

# Redis responding?
docker exec -it research-graph-redis redis-cli ping
```

---

## Troubleshooting

### Common Issues

**Port already in use:**
```bash
# Find what's using the port
lsof -i :8000
# Kill it or change port in uvicorn command
```

**Docker services not starting:**
```bash
# Check logs
docker-compose logs neo4j
# Reset if needed
docker-compose down -v  # Warning: deletes data
docker-compose up -d
```

**Neo4j connection refused:**
- Wait longer (Neo4j takes ~30s to start)
- Check credentials match docker-compose.yml
- Ensure bolt port (7687) is mapped

**Python package issues:**
```bash
# Ensure virtual environment is active
which python  # Should show venv path
pip list  # Check installed packages
```

**Frontend build errors:**
```bash
# Clear cache and reinstall
rm -rf node_modules package-lock.json
npm install
```

**CORS errors in browser:**
- Check backend CORS configuration includes frontend URL
- Ensure both servers are running
- Check browser console for specific error

---

## Development Workflow

Once setup is complete, typical workflow:

1. **Start databases:**
   ```bash
   docker-compose up -d
   ```

2. **Start backend (terminal 1):**
   ```bash
   cd backend
   source venv/bin/activate
   uvicorn app.main:app --reload
   ```

3. **Start frontend (terminal 2):**
   ```bash
   cd frontend
   npm run dev
   ```

4. **Start background worker (terminal 3, when implemented):**
   ```bash
   cd backend
   source venv/bin/activate
   arq app.workers.tasks.WorkerSettings
   ```

5. **Monitor Neo4j (optional):**
   - Open http://localhost:7474
   - Run queries to inspect data

6. **Code changes:**
   - Backend: Auto-reloads with uvicorn --reload
   - Frontend: Hot module replacement via Vite
   - Database schemas: Manual migration

---

## Next Steps

With the environment set up, proceed to **Phase 1** of the Development Roadmap:
1. Implement database connection modules
2. Create Pydantic models for data validation
3. Build basic CRUD endpoints
4. Connect frontend to backend API
5. Verify end-to-end data flow

Refer to `03-DEVELOPMENT_ROADMAP.md` for detailed implementation steps.
