# ThoughtLab Development Guide

Comprehensive guide for developers working on ThoughtLab, including architecture, setup, workflows, and modification guidance.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Architecture Overview](#architecture-overview)
3. [Technology Stack](#technology-stack)
4. [Development Setup](#development-setup)
5. [Testing](#testing)
6. [AI Integration](#ai-integration)
7. [Common Development Workflows](#common-development-workflows)
8. [Extending the System](#extending-the-system)
9. [Dependency Management](#dependency-management)
10. [Development Principles](#development-principles)

---

## Quick Start

### Prerequisites

- **Python 3.11+** (pyenv recommended on Linux/macOS, direct install on Windows)
- **Node.js 18+** (nvm recommended on Linux/macOS, direct install on Windows)
- **Docker & Docker Compose**
- **uv** (Python package manager)
- **PowerShell 5.1+** (Windows only - included with Windows 10+)

**Install uv:**

Linux/macOS/WSL:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Windows PowerShell:
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### One-Command Setup

**Linux/macOS/WSL/Git Bash:**
```bash
./setup.sh
```

**Windows PowerShell:**
```powershell
.\scripts\setup.ps1
```

This handles all setup: dependencies, environment files, Docker services, and database initialization.

### Start Development

**Linux/macOS/WSL/Git Bash:**
```bash
./start.sh      # Start all services (backend + frontend)
./stop.sh       # Stop all services
./restart.sh    # Restart all services
```

**Windows PowerShell (Recommended for Windows):**
```powershell
.\start.ps1     # Start all services (backend + frontend)
.\stop.ps1      # Stop all services
.\restart.ps1   # Restart all services
```

> **Note**: PowerShell scripts provide better process management on Windows than Git Bash scripts.

---

## Architecture Overview

### High-Level Architecture

```
Frontend (React + TypeScript)
    ↓ HTTP/REST
Backend (FastAPI + Python)
    ↓
Neo4j (Graph DB - Required) + Redis (Cache - Optional) + PostgreSQL (Users/Logs - Scaffolded)
    ↓
AI Layer (LangChain + LangGraph + OpenAI)
```

### Component Layers

1. **Frontend**: React + TypeScript + Vite + Tailwind + React Query + Cytoscape.js
2. **Backend**: FastAPI (async) with Pydantic models
3. **Databases**: Neo4j (primary graph - **required**), Redis (cache/RT - **optional**), PostgreSQL (scaffolded - **future**)
4. **AI**: LangChain/LangGraph for embeddings, similarity search, and intelligent agents

### Key Architectural Decisions

#### Database Strategy

- **Neo4j** (Required): Knowledge graph (nodes, relationships), vector indexes for embeddings
- **Redis** (Optional): Caching, real-time features when available - backend runs fine without it
- **PostgreSQL** (Future): User accounts, activity logs, settings (scaffolded for future use)

**Why Neo4j?**
- Native graph operations optimize relationship traversal
- Built-in vector indexing (5.13+) eliminates need for separate vector DB
- Cypher query language perfect for "find paths" and "what connects X to Y" queries

**Redis Optional Design:**
- Backend starts successfully with only Neo4j running
- Redis provides performance benefits (caching) when available
- Health check shows Redis as "not configured (optional)" when unavailable
- Overall system health does not depend on Redis

#### Identifier Strategy

Use **UUIDv4 strings** as canonical IDs for all entities and relationships:
- Neo4j's internal `id()` is unstable across export/import
- UUIDs are portable and work consistently everywhere (API, frontend, exports)

```python
# Always use property-based IDs
node_id = str(uuid.uuid4())
```

```cypher
-- Always query by property 'id', never by id()
MATCH (n:Observation {id: $id}) RETURN n
```

#### Modularity Principles

1. **Single Responsibility**: Each module handles one domain area
2. **Clear Interfaces**: Pydantic models (backend), TypeScript types (frontend)
3. **Separation of Concerns**: Routes → Services → Database connectors
4. **Minimal Coupling**: Props/callbacks over global state
5. **Easy Extension Points**: New node types require isolated changes

#### Background Processing

**Current**: Synchronous AI processing (3-5s per node) is acceptable for research workflows

**Future**: Implement ARQ/Celery when:
- API timeouts become common
- Processing time exceeds 10 seconds regularly
- Multi-user deployments show resource contention
- Batch operations needed (re-analyze entire graph)

#### CORS Configuration

**Development Setup**: Backend allows frontend connections from multiple Vite dev server ports:
- Explicit origins: `http://localhost:5173` through `http://localhost:5179`
- Regex pattern: Any localhost or 127.0.0.1 with any port
- Supports `strictPort: true` in Vite config (prevents auto-incrementing)

**Why multiple ports?**
- Vite auto-increments ports when 5173 is occupied
- Range 5173-5179 covers common development scenarios
- Prevents CORS errors during multi-developer scenarios or port conflicts

**Production**: Configure CORS to allow only your specific frontend domain(s)

---

## Technology Stack

### Backend

| Package | Version | Purpose |
|---------|---------|---------|
| fastapi | 0.121+ | Async web framework |
| uvicorn | 0.32+ | ASGI server |
| neo4j | 5.26+ | Graph database driver |
| redis | 5.2+ | Redis client |
| pydantic | 2.0+ | Data validation |
| langchain | 1.2+ | AI framework |
| langgraph | 1.0+ | Agent workflows |
| langchain-openai | 1.1+ | OpenAI integration |
| fastmcp | 0.2+ | MCP server framework |
| pytest | 8.3+ | Testing |

### Frontend

| Package | Version | Purpose |
|---------|---------|---------|
| react | 18.3+ | UI framework |
| typescript | 5.7+ | Type safety |
| vite | 5.4+ | Build tool |
| @tanstack/react-query | 5.62+ | Server state |
| axios | 1.7+ | HTTP client |
| cytoscape | 3.31+ | Graph visualization |
| tailwindcss | 3.4+ | Styling |
| vitest | 2.1+ | Testing |

---

## Development Setup

### 1. Clone and Install

```bash
git clone <repository-url>
cd thoughtlab
./setup.sh
```

### 2. Configure Environment

**Backend** (`backend/.env`):
```bash
# Database (Required)
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=research_graph_password

# Cache (Optional - backend runs without Redis)
REDIS_URL=redis://localhost:6379

# Security
SECRET_KEY=<generate-with-secrets.token_urlsafe(32)>

# AI (Optional but recommended for AI features)
THOUGHTLAB_OPENAI_API_KEY=sk-...
THOUGHTLAB_LLM_MODEL=gpt-4o-mini
THOUGHTLAB_EMBEDDING_MODEL=text-embedding-3-small
```

> **Note**: Only Neo4j is required. Backend starts successfully without Redis or OpenAI API keys.

**Frontend** (`frontend/.env`):
```bash
VITE_API_URL=http://localhost:8000/api/v1
```

> **Note**: Frontend is configured with `strictPort: true` to always use port 5173. If port is occupied, Vite will fail with an error instead of auto-incrementing to 5174, 5175, etc.

### 3. Initialize Neo4j

Run constraints and indexes:

```bash
docker exec research-graph-neo4j cypher-shell -u neo4j -p research_graph_password < docker/neo4j/init.cypher
```

Or run in Neo4j Browser (http://localhost:7474):

```cypher
// Unique constraints
CREATE CONSTRAINT observation_id IF NOT EXISTS FOR (o:Observation) REQUIRE o.id IS UNIQUE;
CREATE CONSTRAINT hypothesis_id IF NOT EXISTS FOR (h:Hypothesis) REQUIRE h.id IS UNIQUE;
CREATE CONSTRAINT source_id IF NOT EXISTS FOR (s:Source) REQUIRE s.id IS UNIQUE;
CREATE CONSTRAINT concept_id IF NOT EXISTS FOR (c:Concept) REQUIRE c.id IS UNIQUE;
CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE;

// Vector indexes (for AI features)
CREATE VECTOR INDEX observation_embedding IF NOT EXISTS
FOR (o:Observation) ON o.embedding
OPTIONS {indexConfig: {
  `vector.dimensions`: 1536,
  `vector.similarity_function`: 'cosine'
}};
-- Repeat for hypothesis, source, concept, entity
```

### 4. Run Development Servers

**Option A - Use Start Scripts (Recommended):**

Linux/macOS/WSL/Git Bash:
```bash
./start.sh      # Starts both backend and frontend
```

Windows PowerShell:
```powershell
.\start.ps1     # Starts both backend and frontend
```

**Option B - Manual Control (Better for Debugging):**

**Terminal 1 - Backend**:

Linux/macOS/WSL:
```bash
cd backend
source .venv/bin/activate
uvicorn app.main:app --reload
# API: http://localhost:8000
# Docs: http://localhost:8000/docs
```

Windows PowerShell:
```powershell
cd backend
.venv\Scripts\Activate.ps1
uvicorn app.main:app --reload
# API: http://localhost:8000
# Docs: http://localhost:8000/docs
```

Windows Git Bash:
```bash
cd backend
source .venv/Scripts/activate
uvicorn app.main:app --reload
```

**Terminal 2 - Frontend**:
```bash
cd frontend
npm run dev
# App: http://localhost:5173
```

---

## Testing

### Backend Tests (pytest)

```bash
cd backend
source .venv/bin/activate

# Run all tests
pytest

# Run specific file
pytest tests/test_api_nodes.py

# Run with coverage
pytest --cov=app --cov-report=html

# View coverage report
open htmlcov/index.html
```

**Test Files**:
- `tests/test_api_nodes.py` - API endpoint tests
- `tests/test_api_activities.py` - Activity feed tests
- `tests/test_graph_service.py` - Service layer tests
- `tests/conftest.py` - Shared fixtures

### Frontend Tests (vitest)

```bash
cd frontend

# Run in watch mode
npm test

# Run once
npm test -- --run

# Run with coverage
npm run test:coverage

# Visual UI
npm run test:ui
```

**Test Files**:
- `src/App.test.tsx` - Main app
- `src/components/__tests__/*.test.tsx` - Component tests
- `src/services/__tests__/api.test.ts` - API client tests

### Best Practices

1. **Test behavior, not implementation** - Focus on what users see/do
2. **Keep tests isolated** - Each test should be independent
3. **Use descriptive names** - Test names explain what they verify
4. **Mock external dependencies** - Don't hit real APIs in tests
5. **Clean up** - Use fixtures to reset state between tests

---

## AI Integration

ThoughtLab uses a **three-layer AI architecture** with a shared core service:

```
Claude Desktop → MCP Server (/mcp) ─┐
LangGraph Agent ────────────────────┼→ ToolService (in-process)
Frontend UI ─→ REST API (/api/v1) ──┘
```

Both MCP and LangGraph call ToolService directly (in-process, no HTTP), sharing:
- Tool definitions from `app/tools/tool_definitions.py`
- Business logic from `app/services/tools/`
- Job queue from `app/services/job_service.py` (for async operations)
- Report storage from `app/services/report_service.py` (for LangGraph results)

### AI Capabilities

1. **Embedding Generation**: Convert node content to vectors using OpenAI
2. **Similarity Search**: Find semantically related nodes using Neo4j vector indexes
3. **Relationship Classification**: LLM identifies relationship types (SUPPORTS, CONTRADICTS, etc.)
4. **Automated Suggestions**: Creates suggestions based on confidence thresholds
5. **Intelligent Tools**: LLM-powered operations (find related, summarize, recalculate confidence)

### Layer 1: Backend API

**Location**: `backend/app/api/routes/tools.py`, `backend/app/services/tool_service.py`

**Endpoints**:
- `POST /api/v1/tools/nodes/{id}/find-related` - Semantic similarity search
- `POST /api/v1/tools/nodes/{id}/summarize` - AI-powered summaries
- `POST /api/v1/tools/nodes/{id}/summarize-with-context` - Context-aware analysis
- `POST /api/v1/tools/nodes/{id}/recalculate-confidence` - Reliability assessment
- `POST /api/v1/tools/relationships/{id}/summarize` - Explain connections
- `GET /api/v1/tools/health` - Backend status
- `GET /api/v1/tools/capabilities` - List available tools

**Test Backend API**:
```bash
cd backend
python validate_tools.py

# Or manually:
curl http://localhost:8000/api/v1/tools/health
```

### Layer 2: LangGraph Agents

**Location**: `backend/app/agents/`

**Key Files**:
- `agent.py` - Agent creation with ReAct pattern
- `agent_tools.py` - 10 tools calling ToolService directly (in-process)
- `config.py` - Agent configuration
- `state.py` - Agent state schema

**Features**:
- ReAct pattern for intelligent tool selection
- 10 tools calling ToolService directly (no HTTP)
- Results saved to ReportService for later viewing
- Multi-step reasoning with automatic tool chaining

**Usage**:
```bash
cd backend
python validate_agent.py

# Programmatic
from app.agents import create_thoughtlab_agent, run_agent
agent = create_thoughtlab_agent()
response = await run_agent(agent, "Find nodes related to obs-123")
```

### Layer 3: MCP Server

**Location**: `backend/app/mcp/`

**Key Files**:
- `server.py` - MCP server setup with FastMCP
- `mcp_tools.py` - Tool wrappers calling ToolService directly (in-process)

**Features**:
- Mounted at `/mcp` endpoint with Streamable HTTP transport
- 10+ MCP tools for Claude Desktop integration
- Calls ToolService directly (no HTTP, same as LangGraph)
- Dangerous tools gated by `THOUGHTLAB_MCP_ADMIN_MODE` env var
- Uses shared tool definitions from `app/tools/tool_definitions.py`

**Setup for Claude Desktop**:
```bash
cd backend
fastmcp install claude-desktop mcp_server.py \
  --server-name "ThoughtLab" \
  --env THOUGHTLAB_OPENAI_API_KEY="sk-..."
```

**Test MCP Server**:
```bash
cd backend
python validate_mcp.py

# Or access directly at /mcp endpoint when backend is running
```

### AI Configuration

**Environment Variables**:
```bash
# Required for AI features
THOUGHTLAB_OPENAI_API_KEY=sk-...

# Optional (with defaults)
THOUGHTLAB_LLM_MODEL=gpt-4o-mini
THOUGHTLAB_EMBEDDING_MODEL=text-embedding-3-small
THOUGHTLAB_EMBEDDING_DIMENSIONS=1536
THOUGHTLAB_AUTO_CREATE_THRESHOLD=0.8
THOUGHTLAB_SUGGEST_THRESHOLD=0.6
THOUGHTLAB_SIMILARITY_MIN_SCORE=0.5
THOUGHTLAB_MAX_SIMILAR_NODES=20
```

**Confidence Thresholds**:
- **≥ 0.8**: Auto-create relationship (marked as `created_by: system-llm`)
- **0.6 - 0.8**: Create suggestion in Activity Feed for user review
- **< 0.6**: Discard silently

### AI Workflow

When a new node is created:
1. Generate embeddings using OpenAI `text-embedding-3-small`
2. Store embeddings in Neo4j vector index
3. Find similar nodes using vector similarity search
4. For each candidate, LLM classifies relationship type and confidence
5. High confidence (≥0.8): Auto-create relationship
6. Medium confidence (0.6-0.8): Create suggestion for user review
7. Low confidence (<0.6): Discard

### AI Tools Reference

ThoughtLab provides 10 AI-powered tools for analyzing and modifying the knowledge graph. These tools are accessible via:
- **Frontend UI**: Buttons in NodeInspector and RelationInspector
- **REST API**: Direct HTTP calls to `/api/v1/tools/...`
- **LangGraph Agent**: Programmatic agent invocation
- **MCP Server**: Claude Desktop integration

#### Node Analysis Tools

##### 1. Find Related Nodes

**Purpose**: Discover semantically similar nodes using vector embeddings.

**Endpoint**: `POST /api/v1/tools/nodes/{node_id}/find-related`

**Request Body**:
```json
{
  "limit": 10,
  "min_similarity": 0.5,
  "node_types": ["Observation", "Hypothesis"],
  "auto_link": false
}
```

**Response**:
```json
{
  "success": true,
  "node_id": "obs-123",
  "related_nodes": [
    {
      "id": "obs-456",
      "type": "Observation",
      "content": "Text preview...",
      "similarity_score": 0.87,
      "suggested_relationship": "RELATES_TO",
      "reasoning": "Both discuss quantum entanglement..."
    }
  ],
  "links_created": 0,
  "message": "Found 5 related nodes"
}
```

**How It Works**:
1. Extracts content from the source node (text, description, title, or name field)
2. Uses the node's embedding to query Neo4j vector index
3. Filters results by `min_similarity` threshold and optional `node_types`
4. For each candidate, LLM classifier suggests relationship type and reasoning
5. If `auto_link: true`, automatically creates relationships for high-confidence matches

**Frontend Location**: NodeInspector → AI Tools → "Find Related Nodes" button

---

##### 2. Summarize Node

**Purpose**: Generate a concise AI summary of a node's content.

**Endpoint**: `POST /api/v1/tools/nodes/{node_id}/summarize`

**Request Body**:
```json
{
  "max_length": 200,
  "style": "concise"
}
```

**Styles**: `"concise"` | `"detailed"` | `"bullet_points"`

**Response**:
```json
{
  "success": true,
  "node_id": "obs-123",
  "summary": "This observation discusses...",
  "key_points": [
    "Main finding about X",
    "Secondary observation about Y"
  ],
  "word_count": 42
}
```

**How It Works**:
1. Extracts node content
2. Constructs prompt based on style preference
3. Calls LLM (GPT-4o-mini by default) to generate summary
4. Extracts key points from summary or generates them separately

**Frontend Location**: NodeInspector → AI Tools → "Summarize" button

---

##### 3. Summarize Node with Context

**Purpose**: Generate a summary that includes information about connected nodes.

**Endpoint**: `POST /api/v1/tools/nodes/{node_id}/summarize-with-context`

**Request Body**:
```json
{
  "depth": 1,
  "relationship_types": ["SUPPORTS", "CONTRADICTS"],
  "max_length": 300
}
```

**Response**:
```json
{
  "success": true,
  "node_id": "hyp-789",
  "summary": "This hypothesis proposes...",
  "context": {
    "supports": ["Observation obs-123: Evidence showing..."],
    "contradicts": ["Source src-456: Study found opposite results..."],
    "related": ["Concept con-789: Related to quantum mechanics..."]
  },
  "synthesis": "Overall, this hypothesis has strong support from 3 observations but conflicts with recent experimental data.",
  "relationship_count": 7
}
```

**How It Works**:
1. Retrieves the node and all connected nodes within specified depth
2. Categorizes connections by relationship type (supports, contradicts, related)
3. Builds comprehensive prompt including node content and context
4. LLM generates context-aware summary and synthesis

**Frontend Location**: NodeInspector → AI Tools → "Summarize with Context" button

---

##### 4. Recalculate Node Confidence

**Purpose**: Re-assess a node's confidence score based on content quality and graph context.

**Endpoint**: `POST /api/v1/tools/nodes/{node_id}/recalculate-confidence`

**Request Body**:
```json
{
  "factor_in_relationships": true
}
```

**Response**:
```json
{
  "success": true,
  "node_id": "obs-123",
  "old_confidence": 0.7,
  "new_confidence": 0.85,
  "reasoning": "Confidence increased due to supporting evidence from 3 new sources",
  "factors": [
    {"factor": "Supporting evidence", "impact": "+0.1"},
    {"factor": "Source credibility", "impact": "+0.05"}
  ]
}
```

**How It Works**:
1. Retrieves current node and its confidence value
2. If `factor_in_relationships: true`, counts supporting and contradicting relationships
3. LLM evaluates content quality, clarity, and evidence support
4. Parses LLM response for new confidence score and factors
5. Updates node's confidence in Neo4j

**Frontend Location**: NodeInspector → AI Tools → "Recalculate Confidence" button

---

##### 5. Reclassify Node

**Purpose**: Change a node's type (e.g., Observation → Hypothesis).

**Endpoint**: `POST /api/v1/tools/nodes/{node_id}/reclassify`

**Request Body**:
```json
{
  "new_type": "Hypothesis",
  "preserve_relationships": true
}
```

**Valid Types**: `"Observation"` | `"Hypothesis"` | `"Question"` | `"Source"` | `"Note"`

**Response**:
```json
{
  "success": true,
  "node_id": "obs-123",
  "old_type": "Observation",
  "new_type": "Hypothesis",
  "properties_preserved": ["text", "confidence", "created_at"],
  "relationships_preserved": 5,
  "message": "Successfully reclassified from Observation to Hypothesis"
}
```

**How It Works**:
1. Validates new type against allowed node types
2. Retrieves current node and relationship count
3. Updates Neo4j node label (removes old label, adds new label)
4. Preserves all properties and relationships

**Frontend Location**: NodeInspector → AI Tools → "Reclassify Node" dropdown

---

##### 6. Search Web for Evidence

**Purpose**: Search the web for supporting or contradicting evidence (requires Tavily API key).

**Endpoint**: `POST /api/v1/tools/nodes/{node_id}/search-web-evidence`

**Request Body**:
```json
{
  "evidence_type": "supporting",
  "max_results": 5,
  "auto_create_sources": false
}
```

**Evidence Types**: `"supporting"` | `"contradicting"` | `"all"`

**Response** (when configured):
```json
{
  "success": true,
  "node_id": "hyp-123",
  "query_used": "evidence for: quantum entanglement in neurons",
  "results": [
    {
      "title": "Recent findings on quantum entanglement",
      "url": "https://example.com/paper",
      "snippet": "Study confirms that...",
      "relevance_score": 0.89
    }
  ],
  "sources_created": 0,
  "message": "Found 5 relevant results"
}
```

**Response** (when not configured):
```json
{
  "success": false,
  "node_id": "hyp-123",
  "message": "Web search not configured",
  "error": "TAVILY_API_KEY environment variable not set"
}
```

**How It Works**:
1. Checks for `TAVILY_API_KEY` environment variable
2. Extracts node content to build search query
3. Calls Tavily API for web search (when configured)
4. If `auto_create_sources: true`, creates Source nodes for results

**Frontend Location**: NodeInspector → AI Tools → "Search Web for Evidence" button

---

##### 7. Merge Nodes

**Purpose**: Combine two nodes of the same type into one, transferring relationships.

**Endpoint**: `POST /api/v1/tools/nodes/merge`

**Request Body**:
```json
{
  "primary_node_id": "obs-123",
  "secondary_node_id": "obs-456",
  "merge_strategy": "combine"
}
```

**Strategies**:
- `"combine"`: Merge text content from both nodes (appends with separator)
- `"keep_primary"`: Keep primary node's content, only transfer relationships
- `"keep_secondary"`: Use secondary node's content, then transfer relationships
- `"smart"`: Use AI to intelligently combine text content, preserving all data points and removing duplicates

**Response**:
```json
{
  "success": true,
  "primary_node_id": "obs-123",
  "secondary_node_id": "obs-456",
  "merged_properties": ["text", "notes"],
  "relationships_transferred": 3,
  "message": "Successfully merged nodes. Transferred 3 relationships."
}
```

**How It Works**:
1. Validates both nodes exist and have the same type
2. Merges properties based on strategy:
   - `combine`: Appends text fields with `\n\n---\n\n` separator
   - `keep_primary`: No property changes
   - `keep_secondary`: Overwrites primary with secondary values
   - `smart`: Uses LLM to intelligently merge text, preserving unique information and removing redundancy
3. Transfers all relationships from secondary to primary node
4. Deletes secondary node

**Frontend Location**: NodeInspector → AI Tools → "Merge with Another Node" button (opens modal)

---

#### Relationship Analysis Tools

##### 8. Summarize Relationship

**Purpose**: Explain the connection between two nodes in plain language.

**Endpoint**: `POST /api/v1/tools/relationships/{edge_id}/summarize`

**Request Body**:
```json
{
  "include_evidence": true
}
```

**Response**:
```json
{
  "success": true,
  "edge_id": "rel-abc",
  "from_node": {
    "id": "obs-123",
    "type": "Observation",
    "content": "Particles showed entanglement..."
  },
  "to_node": {
    "id": "hyp-456",
    "type": "Hypothesis",
    "content": "Quantum mechanics predicts..."
  },
  "relationship_type": "SUPPORTS",
  "summary": "This observation provides experimental evidence that supports the hypothesis about quantum entanglement",
  "evidence": [
    "Direct measurement of entangled state",
    "Results match theoretical predictions"
  ],
  "strength_assessment": "strong"
}
```

**Strength Assessments**: `"strong"` (≥0.8) | `"moderate"` (0.6-0.8) | `"weak"` (<0.6)

**How It Works**:
1. Retrieves relationship and both connected nodes
2. Extracts content from both nodes
3. LLM generates explanation of the connection
4. If `include_evidence: true`, extracts specific evidence points
5. Assesses strength based on relationship confidence

**Frontend Location**: RelationInspector → AI Tools → "Summarize Relationship" button

---

##### 9. Recalculate Edge Confidence

**Purpose**: Re-evaluate relationship strength based on connected nodes and graph context.

**Endpoint**: `POST /api/v1/tools/relationships/{edge_id}/recalculate-confidence`

**Request Body**:
```json
{
  "consider_graph_structure": true
}
```

**Response**:
```json
{
  "success": true,
  "edge_id": "rel-abc",
  "old_confidence": 0.6,
  "new_confidence": 0.75,
  "reasoning": "Relationship strengthened by additional supporting observations in the graph",
  "factors": [
    {"factor": "Content alignment", "impact": "good"},
    {"factor": "Logical connection", "impact": "clear"}
  ]
}
```

**How It Works**:
1. Retrieves relationship and both connected nodes
2. If `consider_graph_structure: true`, counts relationships on both nodes
3. LLM evaluates semantic connection between node contents
4. Parses response for new confidence and contributing factors
5. Updates relationship confidence in Neo4j

**Frontend Location**: RelationInspector → AI Tools → "Recalculate Confidence" button

---

##### 10. Reclassify Relationship

**Purpose**: Change relationship type or let AI suggest the best type.

**Endpoint**: `POST /api/v1/tools/relationships/{edge_id}/reclassify`

**Request Body**:
```json
{
  "new_type": null,
  "preserve_notes": true
}
```

If `new_type` is `null`, AI suggests the best relationship type.

**Valid Types**: `"SUPPORTS"` | `"CONTRADICTS"` | `"RELATES_TO"` | `"DERIVED_FROM"` | `"CITES"`

**Response**:
```json
{
  "success": true,
  "edge_id": "rel-abc",
  "old_type": "RELATES_TO",
  "new_type": "SUPPORTS",
  "suggested_by_ai": true,
  "reasoning": "Analysis shows the source provides direct evidence for the hypothesis",
  "notes_preserved": true
}
```

**How It Works**:
1. Retrieves relationship and both connected nodes
2. If `new_type` is provided, uses that type
3. If `new_type` is `null`, LLM analyzes both nodes and suggests best type
4. Recreates relationship with new type (Neo4j doesn't allow in-place type changes)
5. Preserves notes if requested

**Frontend Location**: RelationInspector → AI Tools → "Reclassify Relationship" dropdown

---

#### Tool Health & Capabilities

##### Health Check

**Endpoint**: `GET /api/v1/tools/health`

**Response**:
```json
{
  "status": "healthy",
  "ai_configured": true,
  "llm_model": "gpt-4o-mini",
  "embedding_model": "text-embedding-3-small"
}
```

Status can be:
- `"healthy"`: AI features fully available
- `"degraded"`: AI not configured (missing OpenAI API key)

---

##### Capabilities List

**Endpoint**: `GET /api/v1/tools/capabilities`

**Response**:
```json
{
  "node_analysis": [
    {"operation": "find_related_nodes", "description": "Find semantically similar nodes"},
    {"operation": "summarize_node", "description": "Generate AI summary"},
    {"operation": "summarize_node_with_context", "description": "Summary including relationships"},
    {"operation": "recalculate_confidence", "description": "Re-assess node confidence"}
  ],
  "node_modification": [
    {"operation": "reclassify_node", "description": "Change node type"},
    {"operation": "search_web_evidence", "description": "Search web for evidence"},
    {"operation": "merge_nodes", "description": "Combine two nodes"}
  ],
  "relationship_analysis": [
    {"operation": "summarize_relationship", "description": "Explain connection"},
    {"operation": "recalculate_edge_confidence", "description": "Re-assess relationship strength"},
    {"operation": "reclassify_relationship", "description": "Change relationship type"}
  ]
}
```

---

#### Error Handling

All tool endpoints follow consistent error patterns:

**404 Not Found** (node/relationship doesn't exist):
```json
{
  "detail": "Node not found"
}
```

**400 Bad Request** (invalid parameters):
```json
{
  "success": false,
  "error": "Invalid parameters",
  "message": "min_similarity must be between 0 and 1"
}
```

**500 Internal Server Error** (LLM or database issues):
```json
{
  "success": false,
  "error": "OpenAI API error",
  "message": "Rate limit exceeded"
}
```

---

#### Frontend Integration

Tools are integrated into the UI via the `AIToolsSection` component:

**NodeInspector** (6 tools):
1. Find Related Nodes
2. Summarize
3. Summarize with Context
4. Recalculate Confidence
5. Reclassify Node (dropdown)
6. Search Web for Evidence
7. Merge with Another Node (opens modal)

**RelationInspector** (3 tools):
1. Summarize Relationship
2. Recalculate Confidence
3. Reclassify Relationship (dropdown with "Let AI Suggest" option)

**UI Patterns**:
- Tools show loading state while processing
- All tools disabled when any tool is running (prevents conflicts)
- Success/error messages shown via toast notifications
- Dropdowns close on click-outside
- Merge nodes shows confirmation modal with strategy selection

---

## Common Development Workflows

### Adding a New Node Type

1. **Update models** (`backend/app/models/nodes.py`):
```python
class NewNodeType(BaseModel):
    id: str
    title: str
    created_at: datetime
    # ... type-specific fields
```

2. **Add to graph service** (`backend/app/services/graph_service.py`):
```python
async def create_new_node(data: NewNodeCreate):
    node_id = str(uuid.uuid4())
    # ... Cypher query to create node
```

3. **Create API route** (`backend/app/api/routes/nodes.py`):
```python
@router.post("/nodes/newtype")
async def create_new_type(data: NewNodeCreate):
    node_id = await graph_service.create_new_node(data)
    return {"id": node_id}
```

4. **Add to frontend types** (`frontend/src/types/graph.ts`):
```typescript
export interface NewNodeType {
  id: string;
  title: string;
  // ...
}
```

5. **Update visualization** (`frontend/src/components/GraphVisualizer.tsx`):
- Add node color/shape in Cytoscape stylesheet

6. **Add Neo4j constraint**:
```cypher
CREATE CONSTRAINT newtype_id IF NOT EXISTS
FOR (n:NewNodeType) REQUIRE n.id IS UNIQUE;
```

### Adding a New API Endpoint

1. **Create route** in appropriate file under `backend/app/api/routes/`
2. **Implement service logic** in `backend/app/services/`
3. **Write tests** in `backend/tests/`
4. **Update API client** (`frontend/src/services/api.ts`)
5. **Update TypeScript types** (`frontend/src/types/`)

### Adding a New Frontend Component

1. **Create component** in `frontend/src/components/`
2. **Write tests** in `frontend/src/components/__tests__/`
3. **Import and use** in parent component
4. **Add to React Query** if it fetches data

### Modifying the Graph Schema

**For nodes**:
1. Update Pydantic models in `backend/app/models/nodes.py`
2. Update graph service Cypher queries
3. Update frontend TypeScript types
4. Add migration script if needed

**For relationships**:
1. Update relationship types enum
2. Update Cytoscape styles for visualization
3. Update classification logic in AI classifier if using AI

### Running Background Tasks

Currently synchronous. When ARQ is needed:
1. Create task in `backend/app/tasks/`
2. Add to ARQ worker configuration
3. Trigger from API endpoint
4. Poll for completion or use WebSocket

---

## Extending the System

### Adding New AI Tools

ThoughtLab uses a unified tool architecture. To add a new tool:

1. **Add Tool Definition** (`backend/app/tools/tool_definitions.py`):
```python
ToolDefinition(
    name="new_operation",
    description="Description for LLM",
    parameters={
        "node_id": {"type": "string", "required": True, "description": "Target node ID"},
        # ... other parameters
    },
    category=ToolCategory.NODE_ANALYSIS,  # or NODE_MODIFICATION, RELATIONSHIP_ANALYSIS
    execution_mode=MCPExecutionMode.SYNC,  # or ASYNC for long-running ops
    is_dangerous=False,  # True if modifies data (requires confirmation)
)
```

2. **Implement Service Logic** (`backend/app/services/tools/operations/`):
```python
# In node_analysis.py, node_modification.py, or relationship_analysis.py:
async def new_operation(self, node_id: str, params: NewOpParams) -> dict:
    """Implement the actual operation logic."""
    # Use self.neo4j_driver, self.llm, self.embedding_model as needed
    return {"success": True, ...}
```

3. **Add to ToolService Facade** (`backend/app/services/tools/service.py`):
```python
async def new_operation(self, node_id: str, params: NewOpParams) -> dict:
    return await self._node_analysis.new_operation(node_id, params)
```

4. **Add API Endpoint** (`backend/app/api/routes/tools.py`):
```python
@router.post("/tools/nodes/{node_id}/new-operation")
async def new_operation(node_id: str, params: NewOpParams):
    result = await tool_service.new_operation(node_id, params)
    return result
```

5. **Add to LangGraph Agent Tools** (`backend/app/agents/agent_tools.py`):
```python
@tool
async def new_operation(node_id: str, ...) -> str:
    """Tool description for LLM."""
    result = await tool_service.new_operation(node_id, ...)
    report_service.save_report(...)  # Optional: save results
    return format_result(result)
```

6. **Add to MCP Tools** (`backend/app/mcp/mcp_tools.py`):
The tool is automatically registered from tool_definitions.py. Just add the result formatter:
```python
def _format_new_operation_result(result: dict) -> str:
    """Format result for Claude Desktop display."""
    return f"Result: {result['message']}"
```

The key benefit: Tool definitions, parameters, and descriptions are shared across REST API, LangGraph, and MCP.

### Integrating External Services

1. **Add configuration** to `backend/app/core/config.py`
2. **Create service wrapper** in `backend/app/services/`
3. **Add environment variables** to `.env.example`
4. **Document in README** and this guide

### Adding Authentication

When implementing FastAPI-Users:
1. Create user models in `backend/app/models/user.py`
2. Set up SQLAlchemy with PostgreSQL
3. Configure FastAPI-Users with JWT
4. Add authentication dependency to protected routes
5. Update frontend to handle auth tokens

---

## Dependency Management

### Backend: uv (Python)

**Why uv**: 10-100x faster than pip, Rust-based, modern resolver

**Common commands**:
```bash
cd backend

# Install dependencies
uv pip install -r requirements.txt

# Add new package
uv pip install package-name

# Update all dependencies
uv pip install --upgrade -r requirements.txt

# Sync with extras
uv sync --all-extras
```

**Adding dependencies**:
1. Add to `backend/pyproject.toml` under `[project.dependencies]`
2. Run `uv sync`
3. Update `backend/requirements.txt` (uv pip compile if needed)

**Strategy**: Minimal constraints with `>=` for core packages, let uv resolve transitives

### Frontend: npm (Node.js)

**Common commands**:
```bash
cd frontend

# Install dependencies
npm install

# Add new package
npm install package-name

# Update all dependencies
npm update

# Audit security
npm audit
```

**Adding dependencies**:
```bash
npm install package-name
# or for dev dependencies
npm install -D package-name
```

### Updating Dependencies

**Backend**:
```bash
cd backend
uv pip install --upgrade -r requirements.txt
pytest  # Verify tests pass
```

**Frontend**:
```bash
cd frontend
npm update
npm test -- --run  # Verify tests pass
```

### Version Pinning

- **Development**: Use flexible versions (>=) for faster iteration
- **Production**: Pin exact versions for reproducibility
- Lock files: `uv pip compile` for Python, `package-lock.json` for Node

---

## Development Principles

### Modularity First

- **Single Responsibility**: Each module does one thing well
- **Clear Interfaces**: Pydantic models (backend), TypeScript types (frontend)
- **Separation of Concerns**: Routes → Services → Database
- **Minimal Coupling**: Props/callbacks over global state
- **Easy Extension**: New features require isolated changes

### Code Quality Standards

- Write tests for new features (pytest/vitest)
- Use type hints throughout (Python 3.11+, TypeScript)
- Keep functions small and focused (<50 lines ideal)
- Document non-obvious decisions
- Follow existing codebase patterns

### Avoid Over-Engineering

- Don't add features beyond what was asked
- Don't refactor unrelated code during bug fixes
- Don't add error handling for impossible scenarios
- Don't create abstractions for single-use code
- Three similar lines > premature abstraction

### Git Workflow

1. Create feature branch from main
2. Make focused, incremental commits
3. Write clear commit messages
4. Run tests before committing
5. Submit PR with description

### Code Review Checklist

- [ ] Tests written and passing
- [ ] Types annotated (Python/TypeScript)
- [ ] Error handling for user-facing operations
- [ ] Documentation updated if needed
- [ ] No secrets committed
- [ ] Follows existing patterns

---

## Troubleshooting

### Backend Issues

**Port 8000 already in use**:

Linux/macOS/WSL:
```bash
lsof -i :8000  # Find process
kill -9 <PID>  # Kill it
```

Windows PowerShell:
```powershell
Get-NetTCPConnection -LocalPort 8000 | Select-Object -ExpandProperty OwningProcess | ForEach-Object { Stop-Process -Id $_ -Force }
```

Or use stop script:
```powershell
.\stop.ps1      # Kills all backend and frontend processes
```

**Database connection errors**:
```bash
docker-compose ps  # Check services are healthy
docker-compose logs neo4j  # Check logs
```

**Redis connection failed (Backend won't start)**:
- **This should NOT happen anymore** - Redis is optional as of recent updates
- If you see this error, you have an old backend process running
- Solution: Run `.\stop.ps1` (Windows) or `./stop.sh` (Linux/macOS) and restart
- Backend now shows "Redis: not configured (optional)" and continues without it

**Module import errors**:
```bash
cd backend
source .venv/bin/activate  # Activate venv
uv sync --all-extras  # Reinstall dependencies
```

### Frontend Issues

**CORS errors (Access-Control-Allow-Origin header not present)**:
1. **Verify backend is actually running**: Open http://localhost:8000/docs
2. **Check backend logs**: You should see request logs when frontend loads
3. **Kill orphaned processes**: Run `.\stop.ps1` (Windows) or `./stop.sh` (Linux/macOS)
4. **Restart both servers**: Backend must be running for frontend to work
5. **Check Vite port**: Frontend should be on port 5173 (or 5174-5179 if auto-incremented)
6. **Verify CORS config**: Backend allows ports 5173-5179 by default

**Port 5173 already in use**:

Using stop script:
```bash
.\stop.ps1      # Windows PowerShell - kills ports 5173-5179
./stop.sh       # Linux/macOS - kills port 5173
```

Manual kill (Windows):
```powershell
Get-NetTCPConnection -LocalPort 5173 | Select-Object -ExpandProperty OwningProcess | ForEach-Object { Stop-Process -Id $_ -Force }
```

Manual kill (Linux/macOS):
```bash
lsof -i :5173 | grep LISTEN | awk '{print $2}' | xargs kill -9
```

**API connection errors**:
- Check backend is running: http://localhost:8000/health
- Verify `VITE_API_URL` in `frontend/.env`
- Ensure no CORS errors in browser console

**Build errors**:
```bash
rm -rf node_modules package-lock.json
npm install
```

### Docker Issues

**Services not starting**:
```bash
docker-compose down -v  # WARNING: Deletes data
docker-compose up -d
docker-compose ps
```

**Neo4j slow to start**:
- Wait ~30 seconds for initialization
- Check logs: `docker-compose logs neo4j`

### AI Issues

**"OpenAI API key not set"**:
```bash
export THOUGHTLAB_OPENAI_API_KEY="sk-..."
```

**Slow AI responses**:
- Use `gpt-4o-mini` instead of `gpt-4o`
- Reduce `max_iterations` in agent config

### Windows-Specific Issues

**PowerShell scripts won't run (execution policy)**:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Git Bash scripts fail to kill processes**:
- **Solution**: Use PowerShell scripts instead (`.\start.ps1`, `.\stop.ps1`)
- PowerShell provides more reliable process management on Windows
- Git Bash has issues with Windows process termination

**Backend changes not reflecting after restart**:
- Old `uvicorn` processes may still be running
- **Solution**: Use `.\stop.ps1` to ensure all Python processes are killed
- Check Task Manager for orphaned `python.exe` processes

**Vite incrementing to port 5174, 5175, etc.**:
- Frontend config has `strictPort: true` to prevent this
- If still happening, old Vite processes on 5173 are running
- **Solution**: Run `.\stop.ps1` to kill all Vite processes on ports 5173-5179

**"command not found" errors in Git Bash**:
- Ensure Git Bash utilities are in PATH
- Scripts add `/usr/bin` and `/mingw64/bin` automatically
- If issues persist, use PowerShell scripts instead

---

## Additional Resources

### Documentation

- [PROJECT_MAP.md](./PROJECT_MAP.md) - Code structure and file locations
- [SETUP.md](./docs/SETUP.md) - Detailed setup instructions
- [TESTING.md](./docs/TESTING.md) - Comprehensive testing guide
- [MCP_SERVER_GUIDE.md](./docs/MCP_SERVER_GUIDE.md) - MCP server reference
- API Docs: http://localhost:8000/docs (when backend running)

### External Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Neo4j Cypher Manual](https://neo4j.com/docs/cypher-manual/current/)
- [LangChain Documentation](https://python.langchain.com/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [React Query Documentation](https://tanstack.com/query/latest)

---

**Last Updated**: 2026-01-03
