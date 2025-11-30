# Development Roadmap

This document outlines the development phases and future plans for the Research Connection Graph application.

---

## Development Principles

See [ADR-012: Development Principles](./02-ARCHITECTURE_DECISIONS.md#adr-012-development-principles---modularity-and-extensibility) for detailed guidelines.

### Modularity First

- **Single Responsibility**: Each module does one thing well
- **Clear Interfaces**: Pydantic models (backend), TypeScript types (frontend)
- **Separation of Concerns**: Routes → Services → Database connectors
- **Minimal Coupling**: Props/callbacks over global state
- **Easy Extension**: New node types require isolated changes

### Quality Standards

- Write tests for new features (pytest/vitest)
- Use type hints throughout
- Keep functions small and focused
- Document non-obvious decisions
- Follow existing codebase patterns

---

## Completed Phases

### Phase 0: Environment Setup ✅

- Docker Compose for Neo4j and Redis
- Python virtual environment with uv
- Frontend with Vite + React + TypeScript
- Project structure established

### Phase 1: Backend Foundation ✅

- FastAPI application with async support
- Neo4j and Redis connections
- Pydantic models for all node types
- Graph service with full CRUD operations
- RESTful API routes with auto-documentation
- Health check endpoint

### Phase 2: Frontend Foundation ✅

- React Query for server state
- TypeScript-typed API client
- Component architecture (modals, inspectors, feed)
- Tailwind CSS styling

### Phase 3: Graph Visualization ✅

- Cytoscape.js integration
- Node/edge selection with highlighting
- Color-coded node types
- Interactive zoom/pan
- Fit/reset controls
- Dark mode support

### Phase 4: Core Features ✅

- Node CRUD (Observation, Source, Hypothesis, Entity)
- Relationship CRUD with confidence/notes
- Node and Relationship inspectors
- Settings modal (theme, layout, colors)
- Activity feed placeholder

---

## In Progress

### Phase 5: Embedding & Similarity

**Goal**: Add semantic search for connection discovery.

**Tasks**:
- [ ] Install sentence-transformers
- [ ] Create embedding service
- [ ] Store embeddings in Neo4j vector index
- [ ] Implement similarity search endpoint
- [ ] Generate embeddings on node creation

**Key Endpoints**:
- `POST /api/v1/search/similar` — Find semantically similar nodes

### Phase 6: LLM Integration

**Goal**: AI-powered connection analysis.

**Tasks**:
- [ ] Configure LiteLLM for provider flexibility
- [ ] Create LLM service for connection analysis
- [ ] Implement analysis worker with ARQ
- [ ] Auto-approve/needs-review/reject logic
- [ ] Broadcast results via WebSocket

---

## Future Phases

### Phase 7: Real-Time Updates

- WebSocket connection for live activity feed
- Broadcast events on background task completion
- Connection status indicator
- Review queue for suggested connections

### Phase 8: Feedback Loop & Learning

- Store feedback in PostgreSQL
- Track approval/rejection rates
- Adjust confidence thresholds based on feedback
- Analyze patterns in rejections

### Phase 9: Natural Language Querying

- LLM interprets questions about the graph
- Generates Cypher queries dynamically
- Synthesizes answers from graph data
- Examples: "What supports hypothesis X?", "Show gaps in topic Y"

### Phase 10: User Authentication

- FastAPI-Users integration
- JWT token authentication
- Multi-user support
- Activity attribution per user

### Phase 11: Polish & Performance

- Comprehensive error handling
- Optimized loading states
- Graph performance for large datasets
- Full test coverage

---

## Backlog

Features not yet scheduled:

- **Graph Filtering**: Filter by node type, time range, confidence
- **Graph Search**: Search nodes by text within visualization
- **Layout Switching**: Toggle between force-directed, hierarchical, circular
- **Keyboard Shortcuts**: Common actions accessible via keyboard
- **Undo/Redo**: Revert recent changes
- **Export**: Export graph as JSON or image
- **Batch Operations**: Create multiple nodes/relationships at once
- **Import**: Import data from external sources

---

## Out of Scope

Per [Project Overview](./01-PROJECT_OVERVIEW.md#project-boundaries):

- Mobile application
- Multi-user collaboration (real-time co-editing)
- Advanced citation management
- PDF parsing / automatic source ingestion
- External database integration (PubMed, arXiv)
- Formal ontology/reasoning engines

---

## Contributing

When adding new features:

1. Check if phase is documented here
2. Follow modularity principles from ADR-012
3. Add tests for new code
4. Update relevant documentation
5. Keep PRs focused on single concerns

See [PROJECT_MAP.md](./PROJECT_MAP.md) for code structure.
