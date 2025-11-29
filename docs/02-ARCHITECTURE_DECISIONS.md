# Architecture Decision Records

This document captures the key technical decisions for the Research Connection Graph project, including rationale, alternatives considered, and trade-offs.

---

## ADR-001: Graph Database Selection

### Decision
Use **Neo4j** as the primary graph database.

### Context
The core of this application is discovering and traversing connections between knowledge nodes. We need a database optimized for graph operations, not relational joins.

### Options Considered

**Option A: Neo4j (Community Edition)**
- Industry standard graph database
- Cypher query language is intuitive
- Native vector index support (since version 5.11)
- Excellent Python driver
- Large community, extensive documentation
- Free community edition sufficient for single-user/small team

**Option B: ArangoDB**
- Multi-model (graph + document + key-value)
- Can handle graph and relational-style queries
- Less mature vector support
- Smaller community

**Option C: Amazon Neptune**
- Managed service, less operational overhead
- Expensive for development/small scale
- Vendor lock-in
- No vector search

**Option D: PostgreSQL with Apache AGE extension**
- Stay in Postgres ecosystem
- Graph features less mature
- Would need separate vector solution (pgvector)
- More complex setup

### Decision Rationale

Neo4j wins because:
1. **First-class graph operations** - Cypher queries for path finding are exactly what we need
2. **Native vector indexing** - Can store embeddings directly, no separate vector DB needed
3. **Python ecosystem** - Official driver is well-maintained and async-compatible
4. **Visualization tools** - Neo4j Browser helps during development
5. **Cost** - Community edition is free, Aura (cloud) has free tier for prototyping

### Trade-offs

**Pros:**
- Optimized for our core use case
- Query language designed for relationship traversal
- Can handle embeddings without another database
- Active development and community

**Cons:**
- Another database to learn and manage
- Community edition lacks some enterprise features (clustering, advanced security)
- Memory-intensive for large graphs
- Cypher has a learning curve

### Consequences

- Development team needs to learn Cypher query language
- Need to run Neo4j alongside other services (Docker Compose)
- Graph schema design becomes critical early decision
- Backup and migration strategies must be graph-aware

---

## ADR-002: Relational Database Selection

### Decision
Use **PostgreSQL** for relational data (users, logs, settings).

### Context
Not all data fits the graph model. User accounts, activity logs, application settings, and feedback data are better suited to traditional relational storage.

### Options Considered

**Option A: PostgreSQL**
- Industry standard, extremely reliable
- Excellent Python support (asyncpg, SQLAlchemy)
- Advanced features if needed (JSON, full-text search)
- pgvector extension available if we need additional vector operations

**Option B: SQLite**
- Zero setup, file-based
- Great for prototyping
- Limited concurrent writes
- No built-in user management

**Option C: MySQL/MariaDB**
- Also well-established
- Less feature-rich than Postgres
- No significant advantages for our use case

**Option D: Skip relational DB, use Neo4j for everything**
- Simpler architecture
- Graph DB not ideal for user sessions, logs, settings
- Would complicate standard auth patterns

### Decision Rationale

PostgreSQL because:
1. **Reliability** - Battle-tested, ACID compliant, won't lose data
2. **Ecosystem** - Every auth library, ORM, and migration tool supports it
3. **Flexibility** - If we need additional features (full-text search, JSON queries), it's there
4. **Async support** - asyncpg is one of the fastest Python database drivers
5. **Future-proofing** - pgvector available if vector needs expand beyond Neo4j

### Trade-offs

**Pros:**
- Standard, well-understood patterns
- Excellent tooling and hosting options
- Scales well if user base grows
- Easy to back up and restore

**Cons:**
- Another service to run (vs SQLite simplicity)
- Slight overhead for small-scale use
- Need to manage two different databases

### Consequences

- Must define and migrate relational schema
- Auth patterns are well-established (good)
- Activity logs can use standard SQL queries
- Need PostgreSQL in Docker Compose setup

---

## ADR-003: Backend Framework Selection

### Decision
Use **FastAPI** for the Python backend.

### Context
Need a Python web framework that can handle REST API, WebSocket connections, and async operations (for LLM calls and database queries).

### Options Considered

**Option A: FastAPI**
- Modern, async-native
- Automatic OpenAPI documentation
- Type hints with Pydantic validation
- Built-in WebSocket support
- Excellent performance

**Option B: Flask**
- Simple, widely known
- Large ecosystem
- Not async-native (needs workarounds)
- WebSocket support via extensions
- No built-in validation

**Option C: Django**
- Full-featured, batteries included
- Django REST Framework is mature
- Heavier weight, more opinionated
- Async support added but not native
- ORM might conflict with Neo4j patterns

**Option D: Starlette (raw)**
- What FastAPI is built on
- More manual work
- No automatic docs or validation

### Decision Rationale

FastAPI because:
1. **Async native** - LLM calls, database queries, and WebSocket all benefit from async
2. **Type safety** - Pydantic models catch data issues early, self-documenting
3. **Auto documentation** - OpenAPI spec generated from code, speeds development
4. **WebSocket support** - Built-in, no extensions needed for real-time feed
5. **Performance** - One of the fastest Python frameworks
6. **Learning curve** - Intuitive for Python developers, good docs

### Trade-offs

**Pros:**
- Modern Python patterns (type hints, async/await)
- Self-documenting API
- Great developer experience
- Active community and development

**Cons:**
- Less mature than Flask/Django
- Fewer third-party extensions
- Team needs to understand async patterns
- Some advanced patterns (middleware, dependencies) have learning curve

### Consequences

- API endpoints are auto-documented (Swagger UI at /docs)
- Request/response validation is automatic
- Background tasks can use FastAPI's built-in support or Celery
- WebSocket endpoints are straightforward to implement
- Team should be comfortable with async/await patterns

---

## ADR-004: Frontend Framework Selection

### Decision
Use **React with TypeScript** and **Vite** as build tool.

### Context
Need a frontend framework capable of rendering interactive graph visualizations, handling real-time updates, and providing responsive UI for node/relationship management.

### Options Considered

**Option A: React + TypeScript**
- Dominant market share, huge ecosystem
- Component-based architecture
- TypeScript adds type safety
- Excellent state management options
- Many graph visualization libraries available

**Option B: Vue.js**
- Simpler learning curve
- Good ecosystem
- Smaller community than React
- Fewer graph visualization options

**Option C: Svelte**
- Excellent performance
- Less boilerplate
- Smaller ecosystem
- Fewer libraries, especially for graph visualization

**Option D: Plain JavaScript + D3.js**
- Maximum control
- No framework overhead
- Much more manual work
- Harder to maintain as app grows

### Decision Rationale

React + TypeScript because:
1. **Ecosystem** - Most graph visualization libraries support React
2. **Type safety** - Graph data structures are complex, TypeScript prevents bugs
3. **Component model** - UI elements (feed items, modals, graph nodes) map well to components
4. **State management** - React Query for server state, Zustand for UI state
5. **Hiring/resources** - Largest community, most tutorials and examples

**Vite** over Create React App because:
- Faster development server (ES modules)
- Faster builds
- Better TypeScript support
- More modern and actively maintained

### Trade-offs

**Pros:**
- Mature, well-understood patterns
- TypeScript catches many bugs at compile time
- Extensive library ecosystem
- Easy to find examples and solutions

**Cons:**
- More boilerplate than Vue/Svelte
- Need to learn React patterns (hooks, effects, etc.)
- Bundle size larger than Svelte
- Can be over-engineered for simple cases

### Consequences

- Frontend development uses modern React patterns (hooks, functional components)
- TypeScript interfaces must be defined for all data structures
- Build tooling is fast and modern (Vite)
- Large selection of UI component libraries available

---

## ADR-005: Graph Visualization Library

### Decision
Use **Cytoscape.js** with react-cytoscapejs wrapper.

### Context
Core feature is interactive graph visualization. Need a library that can render potentially large graphs, handle user interactions, and integrate with React.

### Options Considered

**Option A: Cytoscape.js**
- Mature, purpose-built for graph/network visualization
- Excellent layout algorithms
- Good performance with thousands of nodes
- React wrapper available
- Extensive styling options
- Event handling for clicks, hovers, etc.

**Option B: React Flow**
- Built for React, very smooth integration
- Great for flowcharts and diagrams
- Better for smaller, editable graphs
- Less optimized for large networks
- Limited layout algorithms

**Option C: D3.js (force-directed)**
- Maximum customization
- Low-level, more work to build features
- Not React-native (DOM manipulation conflicts)
- Performance can be tricky for large graphs

**Option D: vis.js**
- Decent graph visualization
- Less maintained recently
- Limited layout options
- Smaller community

**Option E: Sigma.js**
- Very performant (WebGL rendering)
- Good for huge graphs (10k+ nodes)
- Less interactive features
- Steeper learning curve

### Decision Rationale

Cytoscape.js because:
1. **Purpose-built** - Designed specifically for network/graph visualization
2. **Layout algorithms** - Hierarchical, force-directed, circle, grid, etc.
3. **Performance** - Handles hundreds to thousands of nodes well
4. **Interactivity** - Click events, selection, zooming, panning built-in
5. **Styling** - CSS-like selectors for node/edge appearance
6. **Maturity** - Long history, well-documented, stable API

### Trade-offs

**Pros:**
- Handles complex graph layouts automatically
- Rich interaction model (click, hover, select, drag)
- Can style nodes by type (different colors for sources vs observations)
- Good documentation and examples

**Cons:**
- React wrapper adds some indirection
- Not as "React-native" as React Flow
- Large library size
- Some advanced customizations require understanding Cytoscape internals

### Consequences

- Graph layout is handled by proven algorithms
- Can support filtering, searching, highlighting within graph
- Node/edge styling is flexible and powerful
- May need to implement custom behaviors (double-click to edit, etc.)
- Performance should be acceptable for typical research graphs (hundreds of nodes)

---

## ADR-006: LLM Integration Strategy

### Decision
Use **LiteLLM** as a unified interface, supporting both cloud APIs and local models.

### Context
LLM reasoning is core to the connection discovery feature. Need flexibility to use different models (cost, privacy, performance trade-offs) without rewriting code.

### Options Considered

**Option A: LiteLLM**
- Unified API for 100+ LLM providers
- Switch between OpenAI, Anthropic, local models with config change
- Lightweight, focused library
- Active development

**Option B: LangChain**
- Full framework with chains, agents, memory
- Larger ecosystem
- More complex, steeper learning curve
- Can be overkill for our needs

**Option C: Direct OpenAI SDK**
- Simplest integration
- Locked to one provider
- No local model support without rewrite

**Option D: Build custom abstraction**
- Full control
- More work to implement and maintain
- Reinventing the wheel

### Decision Rationale

LiteLLM because:
1. **Flexibility** - Can start with OpenAI, switch to Anthropic or local models later
2. **Simplicity** - Just wraps API calls, doesn't impose framework patterns
3. **Local support** - Works with Ollama for privacy-conscious users
4. **Cost management** - Can route to cheaper models for bulk operations
5. **Future-proof** - New models supported as they release

### Trade-offs

**Pros:**
- Provider-agnostic code
- Easy to test with different models
- Supports both API and local models
- Lightweight addition to codebase

**Cons:**
- Abstraction layer adds slight complexity
- Need to handle provider-specific features manually if used
- Another dependency to manage

### Consequences

- Can develop with cheaper models, deploy with better ones
- Users can choose to use local models for privacy
- Model switching is configuration, not code change
- Must handle rate limits and costs per provider
- Fallback strategies possible (try Claude, fall back to GPT, fall back to local)

---

## ADR-007: Embedding Model Selection

### Decision
Use **sentence-transformers** library with `all-MiniLM-L6-v2` model initially, with option to use domain-specific models.

### Context
Embeddings power the semantic similarity search that identifies candidate connections. Need a model that runs locally (cost-effective) and produces meaningful similarity scores.

### Options Considered

**Option A: sentence-transformers (local)**
- Run embeddings locally, no API costs
- Many pre-trained models available
- Fast inference
- Privacy-preserving

**Option B: OpenAI Embeddings API**
- High quality embeddings
- API cost for every embedding
- Data leaves your system
- Simple integration

**Option C: Cohere Embeddings**
- Good quality
- API costs
- Less ubiquitous than OpenAI

### Model Choices within sentence-transformers:

**all-MiniLM-L6-v2**
- Fast, small model
- Good general-purpose quality
- 384 dimensions
- Well-tested

**all-mpnet-base-v2**
- Higher quality than MiniLM
- Slower, larger
- 768 dimensions

**allenai/specter2**
- Trained on scientific papers
- Better for academic content
- Slower
- Larger embeddings

### Decision Rationale

Local sentence-transformers with all-MiniLM-L6-v2 because:
1. **No API costs** - Embeddings are generated frequently, costs add up
2. **Privacy** - Data doesn't leave user's system
3. **Speed** - Local inference is fast enough for our needs
4. **Flexibility** - Can swap models based on domain (SPECTER for science, general for other)
5. **Quality** - MiniLM is surprisingly good for its size

### Trade-offs

**Pros:**
- Free to run unlimited embeddings
- Fast enough for real-time use
- No external dependencies for core feature
- Can upgrade model later without API changes

**Cons:**
- Requires compute resources (CPU/GPU)
- Initial model download
- Quality slightly lower than latest API models
- Need to manage model loading/memory

### Consequences

- Embedding generation is part of backend, not external service
- Can process new nodes immediately without API rate limits
- Model can be swapped via configuration
- Need to handle model loading on startup
- Memory usage increases (model stays in memory)

---

## ADR-008: Background Task Processing

### Decision
Use **ARQ** (Async Redis Queue) for background job processing.

### Context
LLM analysis of connections shouldn't block the API. When a user adds a node, the system should respond immediately and process connections in the background.

### Options Considered

**Option A: ARQ (Async Redis Queue)**
- Async-native (matches FastAPI)
- Lightweight
- Redis-backed (already using Redis)
- Simple API
- Built for Python async

**Option B: Celery**
- Industry standard
- Battle-tested
- More features (scheduling, monitoring, etc.)
- Not async-native
- Heavier weight, more complex setup

**Option C: FastAPI Background Tasks**
- Built-in to FastAPI
- Simplest option
- No persistence (lost if server restarts)
- No job queuing or retries
- Not suitable for long-running tasks

**Option D: Dramatiq**
- Modern Celery alternative
- Good performance
- Smaller community
- Less documentation

### Decision Rationale

ARQ because:
1. **Async-native** - Works seamlessly with FastAPI's async patterns
2. **Simple** - Easy to set up and understand
3. **Redis-backed** - Already using Redis for caching
4. **Lightweight** - Not over-engineered for our needs
5. **Reliable** - Job persistence, retries, timeouts built-in

### Trade-offs

**Pros:**
- Matches FastAPI's async model perfectly
- Single dependency (Redis) for queue
- Jobs survive server restarts
- Configurable retries and timeouts

**Cons:**
- Smaller community than Celery
- Fewer monitoring tools
- Less battle-tested at scale
- Limited scheduling features (though we don't need them)

### Consequences

- Background workers process connection analysis
- Jobs are queued immediately when nodes are added
- Failed jobs can retry automatically
- Need to monitor Redis memory usage
- Worker processes run alongside API server

---

## ADR-009: Real-Time Communication

### Decision
Use **FastAPI's native WebSocket support** for real-time updates.

### Context
Activity feed needs to update in real-time as background processing completes. User should see when analysis finishes, connections are made, etc.

### Options Considered

**Option A: FastAPI WebSocket**
- Built into FastAPI
- No additional dependencies
- Async-native
- Direct control over implementation

**Option B: Socket.IO (python-socketio)**
- Higher-level abstraction
- Automatic reconnection, rooms, namespaces
- Additional dependency
- May be overkill for our needs

**Option C: Server-Sent Events (SSE)**
- Simpler than WebSocket
- One-way communication (server to client)
- Good browser support
- Can't send messages from client

**Option D: Polling**
- Simplest to implement
- Inefficient (constant requests)
- Not truly real-time
- More server load

### Decision Rationale

FastAPI WebSocket because:
1. **Built-in** - No additional dependencies
2. **Bi-directional** - Can send and receive (useful for future features)
3. **Control** - Direct implementation, no abstraction overhead
4. **Performance** - Efficient for real-time updates
5. **Simplicity** - Our real-time needs are straightforward

### Trade-offs

**Pros:**
- Zero additional dependencies
- Full control over protocol
- Efficient for live updates
- Integrates naturally with FastAPI

**Cons:**
- Need to handle reconnection logic manually
- No built-in rooms/namespaces (need to implement if needed)
- More code than Socket.IO for complex scenarios

### Consequences

- Frontend establishes WebSocket connection on load
- Backend broadcasts events when analysis completes
- Need to handle connection lifecycle (connect, disconnect, errors)
- Activity feed updates immediately
- Can extend to other real-time features later

---

## ADR-010: Authentication Strategy

### Decision
Use **FastAPI-Users** library for authentication, with JWT tokens.

### Context
Need user authentication for multi-user scenarios. Even for single-user, authentication protects the API and enables features like activity attribution.

### Options Considered

**Option A: FastAPI-Users**
- Purpose-built for FastAPI
- Handles registration, login, JWT, OAuth
- PostgreSQL support
- Well-maintained
- Customizable

**Option B: Auth0/Supabase (external service)**
- No need to implement auth
- More features (social login, MFA)
- External dependency
- Costs at scale
- Data leaves your system

**Option C: Build from scratch**
- Full control
- More work
- Security risks if not done properly
- Reinventing solved problems

**Option D: Simple API key**
- Very simple
- No user management
- Not secure enough
- Doesn't scale to multiple users

### Decision Rationale

FastAPI-Users because:
1. **Purpose-built** - Designed for FastAPI, follows its patterns
2. **Complete** - Registration, authentication, password reset included
3. **Self-hosted** - No external service dependency
4. **JWT tokens** - Stateless auth, works well with async
5. **Customizable** - Can add fields, change behavior as needed

### Trade-offs

**Pros:**
- Security best practices built-in
- No external service costs
- Full control over user data
- Integrates with our PostgreSQL
- Async-compatible

**Cons:**
- Need to manage user data securely
- Must implement password reset emails
- No built-in MFA (can add later)
- More responsibility for security

### Consequences

- User registration and login endpoints provided
- JWT tokens for API authentication
- User model stored in PostgreSQL
- Can associate nodes/activity with users
- Need to implement email sending for password reset
- Security updates are our responsibility

---

## Summary

These decisions create a modern, Python-centric stack optimized for:
- Graph operations and discovery (Neo4j)
- AI-powered analysis (LiteLLM, sentence-transformers)
- Real-time, async operations (FastAPI, ARQ, WebSocket)
- Interactive visualization (React, Cytoscape.js)
- Type safety and maintainability (TypeScript, Pydantic)
- **Modularity and extensibility** (clear interfaces, separation of concerns)

The architecture balances power with pragmatism, avoiding over-engineering while providing room to grow. All development follows modularity principles (ADR-012) to ensure the codebase can scale with minimal refactoring.

---

## ADR-011: Identifier Strategy (UUIDv4)

### Decision
Use UUIDv4 (string) as the canonical identifier for all domain entities and relationships.

### Context
- Neo4j’s internal element IDs (`id(n)`, `id(r)`) are not stable across export/import and can be reused.
- Nodes already use a UUIDv4 in property `id`. We standardize relationships to also have a UUIDv4 property `id`.
- All API endpoints and the frontend treat IDs as opaque strings.

### Implementation
- Nodes: property `id` set server-side via Python `uuid.uuid4()`.
- Relationships: property `id` set on creation and used for all lookups and updates.
- Cypher patterns use `WHERE r.id = $rel_id` and `WHERE n.id = $id`. We do not use `id()` for application-level lookup.
- Full graph API returns `r.id` for edge identifiers.

### Migration (if upgrading existing data)
Backfill relationship IDs once:

```cypher
MATCH ()-[r]->()
WHERE r.id IS NULL
SET r.id = randomUUID();
```

Optional (Neo4j 5+): add relationship property indexes for lookup by `id`:

```cypher
CREATE INDEX rel_id_supports IF NOT EXISTS FOR ()-[r:SUPPORTS]-() ON (r.id);
CREATE INDEX rel_id_contradicts IF NOT EXISTS FOR ()-[r:CONTRADICTS]-() ON (r.id);
CREATE INDEX rel_id_relates IF NOT EXISTS FOR ()-[r:RELATES_TO]-() ON (r.id);
```

### Frontend considerations
- Treat IDs as strings and use explicit null/undefined checks in conditional rendering.

### References
- Cypher functions (randomUUID): https://neo4j.com/docs/cypher-manual/4.3/functions/
- Relationship index hint format (Neo4j 5): https://neo4j.com/docs/status-codes/current/notifications/all-notifications/

---

## ADR-012: Development Principles - Modularity and Extensibility

### Decision
Design and implement all features with **modularity** as a core principle, ensuring the codebase can grow and evolve with minimal refactoring.

### Context
This project is intended to grow significantly over time. Early architectural decisions that prioritize modularity will pay dividends as new features are added, reducing technical debt and making onboarding easier.

### Principles

#### 1. Single Responsibility
Each module, component, service, and function should have **one clear purpose**:
- Backend services handle one domain area (e.g., `graph_service` handles all Neo4j operations)
- Frontend components handle one UI concern (e.g., `NodeInspector` displays/edits node details)
- API routes group related endpoints logically (e.g., `/nodes/*`, `/graph/*`, `/settings/*`)

#### 2. Clear Interfaces
Modules communicate through **well-defined interfaces**:
- Backend: Pydantic models define all request/response contracts
- Frontend: TypeScript types define data structures
- API: REST endpoints follow consistent naming and response patterns
- Callbacks: Use stable function references (e.g., `useCallback` in React) to prevent subtle bugs

#### 3. Separation of Concerns
Keep distinct responsibilities in separate layers:
- **Backend**: Routes (HTTP handling) → Services (business logic) → Database connectors (data access)
- **Frontend**: Components (UI rendering) → Hooks/Services (data fetching) → Types (contracts)
- **State**: Selection state lives in `App.tsx`; visualization state lives in `GraphVisualizer`

#### 4. Minimal Coupling
Components should have **minimal dependencies** on each other:
- Prefer props/callbacks over direct imports between components
- Use React Query for server state (decouples data fetching from components)
- Avoid circular dependencies between modules

#### 5. Easy Extension Points
Design with future expansion in mind:
- New node types: Add to `NodeType` enum, create route handler, update `NodeInspector`
- New relationship types: Add to `RelationshipType` enum, update Cytoscape styles
- New visualizations: Graph rendering is isolated in `GraphVisualizer`
- New settings: Add to `AppSettings` model, update `SettingsModal`

### Implementation Guidelines

**Backend:**
```python
# Good: Service handles business logic, route handles HTTP
@router.post("/nodes/observations")
async def create_observation(data: ObservationCreate):
    node_id = await graph_service.create_observation(data)
    return {"id": node_id, "message": "Observation created"}

# Avoid: Business logic in route handler
@router.post("/nodes/observations")
async def create_observation(data: ObservationCreate):
    # Don't put Cypher queries or complex logic here
    ...
```

**Frontend:**
```typescript
// Good: Component receives data via props, emits events via callbacks
function NodeInspector({ nodeId, onClose }: Props) {
  // Component is reusable and testable
}

// Good: Callbacks wrapped in useCallback for stable references
const handleNodeSelect = useCallback((id: string | null) => {
  setSelectedNodeId(id);
  if (id !== null) {
    setSelectedEdgeId(null);
  }
}, []);

// Avoid: Hardcoded dependencies, unstable callbacks
function NodeInspector() {
  const nodeId = useGlobalStore().selectedNode; // Tight coupling
}
```

### Trade-offs

**Pros:**
- New features require changes in isolated areas
- Easier to test individual modules
- Team members can work on different areas without conflicts
- Bugs are typically isolated to specific modules
- Onboarding is simpler (can understand one module at a time)

**Cons:**
- Initial development may feel slower (more structure to set up)
- Requires discipline to maintain boundaries
- Some duplication may occur to avoid coupling
- Need to think ahead about extension points

### Consequences

- All new features should be designed as self-contained modules first
- Code reviews should check for proper separation of concerns
- When a module grows too large, split it before it becomes unwieldy
- Document extension points in code comments for common expansion scenarios
- Prefer composition over inheritance for code reuse
