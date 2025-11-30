# Architecture Decisions

This document captures the technical architecture for the Research Connection Graph project, organized by concern.

---

## Quick Reference

| Concern | Decision | Key Rationale |
|---------|----------|---------------|
| Graph Database | Neo4j | Native graph ops, vector indexes, Cypher |
| Relational Database | PostgreSQL | Users, logs, settings; battle-tested |
| Identifiers | UUIDv4 strings | Portable, stable across export/import |
| Backend Framework | FastAPI | Async-native, auto-docs, WebSocket built-in |
| Background Jobs | ARQ (Redis) | Async-native, matches FastAPI patterns |
| Real-Time | FastAPI WebSocket | Zero dependencies, bi-directional |
| Authentication | FastAPI-Users + JWT | Self-hosted, async-compatible |
| LLM & Embeddings | LangChain + OpenAI | Unified AI layer, hybrid graph+vector queries |
| Frontend Framework | React + TypeScript + Vite | Ecosystem, type safety, fast builds |
| Graph Visualization | Cytoscape.js | Purpose-built, layout algorithms |

---

## Data Layer

### Graph Database: Neo4j

**Decision:** Use Neo4j (Community Edition) as the primary graph database.

**Why Neo4j:**
- First-class graph operations — Cypher queries for path finding are exactly what we need
- Native vector indexing (5.13+) — Store embeddings directly, no separate vector DB
- Python ecosystem — Official async-compatible driver
- Cost — Community edition is free; Aura cloud has free tier

**Trade-offs:**
- ✅ Optimized for relationship traversal and discovery
- ✅ Single database for graph + vectors
- ⚠️ Team needs to learn Cypher
- ⚠️ Memory-intensive for large graphs
- ⚠️ Community edition lacks clustering

**Alternatives Rejected:**
- *ArangoDB* — Less mature vector support, smaller community
- *Amazon Neptune* — Expensive, vendor lock-in, no vector search
- *PostgreSQL + Apache AGE* — Graph features less mature, would need pgvector separately

---

### Relational Database: PostgreSQL

**Decision:** Use PostgreSQL for relational data (users, activity logs, settings).

**Why PostgreSQL:**
- Battle-tested, ACID compliant
- Every auth library, ORM, and migration tool supports it
- asyncpg is one of the fastest Python database drivers
- pgvector available as backup if vector needs expand

**Trade-offs:**
- ✅ Standard patterns, excellent tooling
- ✅ Scales well, easy backup/restore
- ⚠️ Another service to run
- ⚠️ Two databases to manage

**What Goes Where:**

| Neo4j | PostgreSQL |
|-------|------------|
| Knowledge nodes (Observation, Hypothesis, Source, Concept, Entity) | User accounts |
| Relationships (SUPPORTS, CONTRADICTS, RELATES_TO) | Activity logs |
| Embeddings | Application settings |
| Graph analytics | Feedback data |

---

### Identifier Strategy: UUIDv4

**Decision:** Use UUIDv4 strings as canonical identifiers for all entities and relationships.

**Why:**
- Neo4j's internal `id()` is unstable across export/import and can be reused
- Portable — IDs work the same everywhere (API, frontend, exports)
- No collisions — Safe for distributed systems

**Implementation:**

```python
# Backend: Generate on creation
node_id = str(uuid.uuid4())
```

```cypher
// Cypher: Always use property id, never id()
MATCH (n:Observation {id: $id}) RETURN n
MATCH ()-[r:SUPPORTS {id: $rel_id}]->() RETURN r
```

**Indexes Required:**

```cypher
-- Relationship property indexes (Neo4j 5+)
CREATE INDEX rel_id_supports IF NOT EXISTS FOR ()-[r:SUPPORTS]-() ON (r.id);
CREATE INDEX rel_id_contradicts IF NOT EXISTS FOR ()-[r:CONTRADICTS]-() ON (r.id);
CREATE INDEX rel_id_relates IF NOT EXISTS FOR ()-[r:RELATES_TO]-() ON (r.id);
```

---

## AI & Machine Learning

### LangChain + OpenAI

**Decision:** Use **LangChain** as the unified AI framework with **OpenAI** as the provider for both LLM chat/completion and embeddings. Vector embeddings stored in Neo4j's native vector indexes.

**What LangChain Handles:**

| Capability | OpenAI Model | Use Case |
|------------|--------------|----------|
| Chat/Completion | `gpt-4o` / `gpt-4o-mini` | Connection analysis, relationship classification |
| Embeddings | `text-embedding-3-small` | Semantic similarity search |

**Why LangChain + OpenAI:**
- **Unified framework** — Single library for LLM calls, embeddings, and retrieval
- **LangGraph ready** — Natural progression to agent workflows when needed
- **OpenAI quality** — Best-in-class models for both reasoning and embeddings
- **Neo4j integration** — `Neo4jVector` store enables hybrid graph+vector queries

**Hybrid Query Power:**

Neo4j uniquely enables vector + graph queries together:

```cypher
// Find similar observations connected to a specific hypothesis
CALL db.index.vector.queryNodes('observation_embedding', 20, $queryVector)
YIELD node AS obs, score
WHERE score > 0.7
MATCH (obs)-[:SUPPORTS]->(h:Hypothesis {id: $hypothesisId})
RETURN obs, score
```

Impossible with a separate vector database.

**Architecture:**

```text
┌───────────────────────────────────────────────────────────┐
│                     Backend (FastAPI)                      │
│  ┌──────────────┐    ┌─────────────────────────────────┐  │
│  │ Graph Service │───▶│         LangChain               │  │
│  │              │    │  ┌───────────┬───────────────┐  │  │
│  │ Node Service │───▶│  │ ChatOpenAI│ OpenAIEmbeddings│ │  │
│  └──────────────┘    │  │ (gpt-4o)  │ (text-emb-3)  │  │  │
│                      │  └───────────┴───────────────┘  │  │
│                      └─────────────────────────────────┘  │
│                                    │                       │
│                                    ▼                       │
│  ┌─────────────────────────────────────────────────────┐  │
│  │                      Neo4j                           │  │
│  │    Nodes + Relationships + Vector Indexes            │  │
│  └─────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────┘
```

**Configuration:**

```python
# backend/app/core/config.py
OPENAI_API_KEY: str  # Required
LLM_MODEL: str = "gpt-4o-mini"  # or "gpt-4o" for complex analysis
EMBEDDING_MODEL: str = "text-embedding-3-small"
EMBEDDING_DIMENSIONS: int = 1536
```

**Vector Index Schema:**

```cypher
CREATE VECTOR INDEX observation_embedding IF NOT EXISTS
FOR (o:Observation) ON o.embedding
OPTIONS {indexConfig: {
  `vector.dimensions`: 1536,
  `vector.similarity_function`: 'cosine'
}}
-- Repeat for: hypothesis_embedding, source_embedding, concept_embedding, entity_embedding
```

**Key Query Patterns:**

| Pattern | Use Case |
|---------|----------|
| Pure vector | "Find content similar to this query" |
| Vector + graph | "Find similar content connected to Source X" |
| Vector + time | "Find similar content from the last 30 days" |
| Vector + filters | "Find similar observations with confidence > 0.8" |

**Trade-offs:**
- ✅ Unified AI layer (LLM + embeddings in one framework)
- ✅ Hybrid graph+vector queries in Neo4j
- ✅ LangGraph-ready for future agent workflows
- ⚠️ Requires OpenAI API key and ongoing API costs
- ⚠️ Neo4j vector indexes are newer (5.13+)

**Alternatives Rejected:**
- *LiteLLM* — Provider abstraction not needed when committed to OpenAI
- *Direct OpenAI SDK* — Less integration with Neo4j, no LangGraph path
- *Separate vector DB* — Can't do hybrid graph+vector queries

**Future: Model Migration**

```cypher
-- If switching from OpenAI (1536d) to local (384d):
DROP INDEX observation_embedding IF EXISTS;
-- Python re-embeds all nodes...
CREATE VECTOR INDEX observation_embedding FOR (o:Observation) ON o.embedding
OPTIONS {indexConfig: {`vector.dimensions`: 384, `vector.similarity_function`: 'cosine'}}
```

---

## Backend

### Framework: FastAPI

**Decision:** Use FastAPI for the Python backend.

**Why FastAPI:**
- Async-native — LLM calls, database queries, WebSocket all benefit
- Type safety — Pydantic models catch errors early, self-documenting
- Auto documentation — OpenAPI spec at `/docs`
- WebSocket support — Built-in, no extensions needed
- Performance — One of the fastest Python frameworks

**Trade-offs:**
- ✅ Modern Python patterns (type hints, async/await)
- ✅ Self-documenting API
- ⚠️ Less mature than Flask/Django
- ⚠️ Team needs async/await familiarity

**Alternatives Rejected:**
- *Flask* — Not async-native, needs workarounds
- *Django* — ORM conflicts with Neo4j patterns, heavier weight
- *Starlette* — Too manual, no auto-docs

---

### Background Processing: ARQ

**Decision:** Use ARQ (Async Redis Queue) for background job processing.

**Why ARQ:**
- Async-native — Matches FastAPI perfectly
- Redis-backed — Already using Redis
- Simple — Easy to set up, lightweight
- Reliable — Job persistence, retries, timeouts built-in

**Use Cases:**
- LLM analysis of connections (shouldn't block API)
- Embedding generation for new nodes
- Batch processing operations

**Trade-offs:**
- ✅ Jobs survive server restarts
- ✅ Configurable retries/timeouts
- ⚠️ Smaller community than Celery
- ⚠️ Fewer monitoring tools

**Alternatives Rejected:**
- *Celery* — Not async-native, heavier weight
- *FastAPI Background Tasks* — No persistence, can't survive restarts

---

### Real-Time: FastAPI WebSocket

**Decision:** Use FastAPI's native WebSocket support for real-time updates.

**Why:**
- Built-in — Zero additional dependencies
- Bi-directional — Can send and receive
- Control — Direct implementation, no abstraction overhead
- Performance — Efficient for live updates

**Use Cases:**
- Activity feed updates when analysis completes
- Connection suggestions as they're discovered
- Graph changes from other users (future)

**Trade-offs:**
- ✅ Integrates naturally with FastAPI
- ⚠️ Need to handle reconnection logic manually
- ⚠️ No built-in rooms/namespaces

---

### Authentication: FastAPI-Users + JWT

**Decision:** Use FastAPI-Users library for authentication with JWT tokens.

**Why:**
- Purpose-built for FastAPI
- Complete — Registration, login, password reset included
- Self-hosted — No external service dependency
- JWT — Stateless, async-friendly

**Trade-offs:**
- ✅ Security best practices built-in
- ✅ Full control over user data
- ⚠️ Must implement email for password reset
- ⚠️ No built-in MFA (can add later)

**Alternatives Rejected:**
- *Auth0/Supabase* — External dependency, costs, data leaves system
- *Build from scratch* — Security risks, reinventing solved problems

---

## Frontend

### Framework: React + TypeScript + Vite

**Decision:** Use React with TypeScript and Vite as build tool.

**Why React + TypeScript:**
- Ecosystem — Most graph visualization libraries support React
- Type safety — Graph data structures are complex, TypeScript prevents bugs
- Component model — UI elements map well to components
- State management — React Query for server state

**Why Vite:**
- Faster development server (ES modules)
- Faster builds
- Better TypeScript support
- More actively maintained than Create React App

**Trade-offs:**
- ✅ Mature patterns, extensive ecosystem
- ✅ TypeScript catches bugs at compile time
- ⚠️ More boilerplate than Vue/Svelte
- ⚠️ Need to learn React hooks patterns

---

### Graph Visualization: Cytoscape.js

**Decision:** Use Cytoscape.js with react-cytoscapejs wrapper.

**Why Cytoscape:**
- Purpose-built for network/graph visualization
- Layout algorithms — Hierarchical, force-directed, circle, grid
- Performance — Handles hundreds to thousands of nodes
- Interactivity — Click, hover, select, zoom, pan built-in
- Styling — CSS-like selectors for node/edge appearance
- Maturity — Long history, stable API

**Trade-offs:**
- ✅ Handles complex layouts automatically
- ✅ Can style nodes by type (different colors per node type)
- ⚠️ React wrapper adds some indirection
- ⚠️ Large library size

**Alternatives Rejected:**
- *React Flow* — Less optimized for large networks
- *D3.js* — Not React-native, DOM manipulation conflicts
- *Sigma.js* — Less interactive features, steeper learning curve

---

## Development Principles

### Modularity Guidelines

**Core Principles:**

1. **Single Responsibility**
   - Backend services handle one domain area
   - Frontend components handle one UI concern
   - API routes group related endpoints logically

2. **Clear Interfaces**
   - Backend: Pydantic models define request/response contracts
   - Frontend: TypeScript types define data structures
   - API: Consistent naming and response patterns

3. **Separation of Concerns**
   - Backend: Routes → Services → Database connectors
   - Frontend: Components → Hooks/Services → Types

4. **Minimal Coupling**
   - Props/callbacks over direct imports
   - React Query for server state
   - No circular dependencies

5. **Easy Extension Points**
   - New node types: Add to enum, create handler, update inspector
   - New relationship types: Add to enum, update Cytoscape styles
   - New embedding models: Swap via configuration

**Code Patterns:**

```python
# Backend: Service handles logic, route handles HTTP
@router.post("/nodes/observations")
async def create_observation(data: ObservationCreate):
    node_id = await graph_service.create_observation(data)
    return {"id": node_id}
```

```typescript
// Frontend: Props in, callbacks out
function NodeInspector({ nodeId, onClose }: Props) {
  // Component is reusable and testable
}

// Stable callback references
const handleNodeSelect = useCallback((id: string | null) => {
  setSelectedNodeId(id);
}, []);
```

---

## References

**Neo4j:**
- [Vector Indexes](https://neo4j.com/docs/cypher-manual/current/indexes/semantic-indexes/vector-indexes/)
- [Cypher Manual](https://neo4j.com/docs/cypher-manual/current/)

**LangChain:**
- [Neo4jVector Store](https://python.langchain.com/docs/integrations/vectorstores/neo4jvector)

**FastAPI:**
- [Documentation](https://fastapi.tiangolo.com/)
- [WebSocket](https://fastapi.tiangolo.com/advanced/websockets/)

**Cytoscape.js:**
- [Documentation](https://js.cytoscape.org/)
- [react-cytoscapejs](https://github.com/plotly/react-cytoscapejs)
