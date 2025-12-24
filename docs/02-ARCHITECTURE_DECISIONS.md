# Architecture Decisions

This document captures the technical architecture for the Research Connection Graph project, organized by concern.

---

## Quick Reference

🔄 = Interim decision with planned future changes

| Concern | Decision | Key Rationale |
|---------|----------|---------------|
| Graph Database | Neo4j | Native graph ops, vector indexes, Cypher |
| Relational Database | PostgreSQL | Users, logs, settings; battle-tested |
| Identifiers | UUIDv4 strings | Portable, stable across export/import |
| Backend Framework | FastAPI | Async-native, auto-docs, WebSocket built-in |
| Background Jobs | Deferred (ARQ when needed) | Sync processing acceptable; optimize when proven necessary |
| LLM Operations | Tool-based architecture | LLM agents call graph operations via unified tool layer |
| Real-Time | FastAPI WebSocket | Zero dependencies, bi-directional |
| Authentication | FastAPI-Users + JWT 🔄 | Self-hosted, async-compatible |
| AI Workflows | LangGraph | Intelligent tool selection, multi-step reasoning |
| AI Primitives | LangChain + OpenAI 🔄 | Embeddings, LLM calls, Neo4j integration |
| Tool Architecture | Unified tool layer | Single source of truth for all interfaces |
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

### LangGraph + LangChain + OpenAI

**Decision:** Use **LangGraph** for intelligent workflows, **LangChain** for AI primitives, and **OpenAI** as the provider. Vector embeddings stored in Neo4j's native vector indexes.

> 📄 **Implementation Details**: See [langchain_implementation.md](./langchain_implementation.md)

**Framework Roles:**

| Framework | Role | Use Case |
|-----------|------|----------|
| **LangGraph** | Workflow orchestration | Multi-step reasoning, tool selection, branching logic |
| **LangChain** | AI primitives | Embeddings, LLM calls, Neo4j integration |
| **OpenAI** | Model provider | GPT-4o-mini (chat), text-embedding-3-small (vectors) |

**Why LangGraph over LangChain alone:**
- **Intelligent tool selection** — Agent decides which tools to invoke
- **Complex workflows** — Multi-step: embed → search → classify → decide
- **Branching logic** — Auto-create vs suggest vs discard based on confidence
- **Human-in-the-loop** — Interrupt workflows for approval at any point
- **State management** — Persistent context across conversation turns
- **MCP compatibility** — Same tools exposable via Model Context Protocol

### Unified Tool Architecture

**Decision:** Create a shared **tool layer** that can be invoked by LangGraph agents, MCP server, and frontend.

**Why:**
- **Single source of truth** — Logic in one place, not duplicated
- **Consistent behavior** — Same validation everywhere
- **Easy testing** — Test tools once
- **Extensibility** — Add CLI, Slack bot, etc. without rewriting logic

```
┌─────────────────────────────────────────────────────────────┐
│                    TOOL LAYER (Core Logic)                   │
│   create_node │ search_similar │ classify_rel │ query_graph  │
└─────────────────────────────────────────────────────────────┘
                              │
         ┌────────────────────┼────────────────────┐
         ▼                    ▼                    ▼
   ┌──────────┐        ┌──────────┐        ┌──────────┐
   │ LangGraph│        │   MCP    │        │ Frontend │
   │  Agent   │        │  Server  │        │   API    │
   └──────────┘        └──────────┘        └──────────┘
```

### OpenAI Configuration

| Capability | OpenAI Model | Use Case |
|------------|--------------|----------|
| Chat/Completion | `gpt-4o` / `gpt-4o-mini` | Connection analysis, relationship classification |
| Embeddings | `text-embedding-3-small` | Semantic similarity search |

**Why OpenAI:**
- **Quality** — Best-in-class models for both reasoning and embeddings
- **Reliability** — Consistent API, good uptime
- **LangChain integration** — First-class support via `langchain-openai`

> **🔄 INTERIM DECISION:** Currently hardcoded to OpenAI. Future work will add provider selection via settings.

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

> **🔄 INTERIM DECISION:** Currently hardcoded to OpenAI. Future work will add provider selection via settings (OpenAI, Anthropic, local models via Ollama) to support privacy-first and cost-conscious deployments.

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

### Background Processing: Deferred (ARQ when needed)

**Decision:** Defer background job infrastructure until proven necessary by real-world usage data.

**Current State:** Synchronous AI processing (3-5 seconds per node) is acceptable for single-user research workflows.

**Rationale:**
- YAGNI principle — Don't build infrastructure before proving need
- Current performance is acceptable — No user complaints or bottlenecks
- Optimize for capability over scale — Focus on making AI smarter, not faster
- Design with data — Measure real-world performance before optimization

**Decision Criteria** — Implement ARQ/Celery when:
- API timeouts during node creation become common
- Processing time exceeds 10 seconds regularly
- Multi-user deployments show resource contention
- Batch operations needed (re-analyze entire graph)
- Performance monitoring shows clear bottleneck

**When Implemented (Future):**
- *Primary choice:* ARQ (async-native, Redis-backed, simple)
- *Alternative:* Celery (if more mature tooling needed)
- *Implementation:* Worker service, job definitions, retry logic

**Trade-offs:**
- ✅ Simpler codebase now — Fewer moving parts to maintain
- ✅ Easier debugging — Synchronous flows easier to trace
- ✅ Better design later — Queue system designed around actual needs
- ⚠️ May need refactoring if scale issues emerge

---

### LLM-Powered Graph Operations

**Decision:** Build a unified tool layer enabling LLM agents to intelligently operate on the knowledge graph via tool calls.

**Why Tool-Based Architecture:**
- User control — Manual operations (find related, summarize, merge nodes)
- AI automation — LLM agents can call tools based on natural language intent
- Single source of truth — Same logic for LLM, API, frontend, future MCP server
- Composability — Tools can chain together for complex workflows
- Safety — Destructive operations require explicit user confirmation

**Key Capabilities:**

*Node Operations:*
- Find and link semantically related nodes
- Recalculate confidence based on graph context
- Generate LLM summaries (with/without relationship context)
- Search web for supporting/contradicting evidence
- Reclassify node types

*Edge Operations:*
- Recalculate relationship confidence
- Reclassify relationship types
- Explain connections in plain language
- Merge duplicate/similar nodes (with confirmation)

**Safety Model:**
- All destructive operations (delete, merge) require user confirmation
- LLM receives feedback about user approval/denial
- Comprehensive audit trail in Activity Feed
- No silent data loss

**Architecture Layers:**

| Layer | Responsibility |
|-------|----------------|
| LangGraph Agent | Natural language → tool selection |
| Tool Layer | Shared business logic (node ops, edge ops) |
| Services | GraphService, AI workflows, database |
| Database | Neo4j (graph + vectors) + PostgreSQL (users, logs) |

**Trade-offs:**
- ✅ Powerful user-controlled operations
- ✅ Foundation for future LLM autonomy
- ✅ Reusable across interfaces (API, MCP, CLI)
- ✅ Clear safety boundaries
- ⚠️ More complex than simple CRUD
- ⚠️ Requires thoughtful UX for confirmations

**Why This Over Background Jobs:**
- More valuable to users — Direct manipulation beats faster automation
- Learn actual usage patterns — Inform future optimization decisions
- Capabilities before scale — Build features users want first

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

> **🔄 INTERIM DECISION:** FastAPI-Users with JWT is the starting point. Future work will implement **OAuth 2.1** for standards-compliant authentication, social login support, and better security practices.

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
