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

### Phase 5: Activity Feed & Processing Infrastructure ✅

**Goal**: Interactive activity feed and background processing foundation.

**Completed**:
- [x] Activity data model and Neo4j schema
- [x] ActivityService with CRUD operations
- [x] Activity API routes (list, approve, reject)
- [x] Interactive ActivityFeed component
- [x] ProcessingService for workflow orchestration
- [x] EmbeddingService interface
- [x] RecursiveCharacterSplitter for chunking
- [x] Comprehensive test coverage

### Phase 6: LangChain + LangGraph AI Integration ✅

**Goal**: AI-powered embedding generation and relationship discovery.

> 📄 **Implementation Plan**: See [langchain_implementation.md](./langchain_implementation.md)

**Completed**:
- [x] Add LangChain dependencies (`langchain`, `langchain-openai`, `langchain-neo4j`, `langgraph`)
- [x] Create `backend/app/ai/` module structure
- [x] Implement `AIConfig` with `.env` configuration
- [x] Implement `EmbeddingManager` with OpenAI embeddings
- [x] Implement `SimilaritySearch` using Neo4j vector indexes
- [x] Implement `RelationshipClassifier` with structured LLM output
- [x] Implement `AIWorkflow` orchestrating the full pipeline
- [x] Update `ProcessingService` to call AI workflow
- [x] Enable Neo4j vector indexes

**Note**: Integration tests with mocked OpenAI and end-to-end tests are optional future enhancements.

**Configuration**:
```env
THOUGHTLAB_OPENAI_API_KEY=sk-...
THOUGHTLAB_LLM_MODEL=gpt-4o-mini
THOUGHTLAB_EMBEDDING_MODEL=text-embedding-3-small
```

**Confidence Thresholds**:

| Score | Action |
|-------|--------|
| ≥ 0.8 | Auto-create relationship |
| 0.6-0.8 | Suggest to user |
| < 0.6 | Discard |

---

## In Progress

### Phase 7: LLM-Powered Graph Operations

**Goal**: Unified tool layer enabling LLM agents to intelligently operate on the knowledge graph.

> 📄 **Architecture**: See [TOOL_ARCHITECTURE.md](./TOOL_ARCHITECTURE.md)

**Design Decision**: Rather than adding background job infrastructure (ARQ), we're focusing on making the existing AI capabilities more powerful and user-controlled through a tool-based architecture. This allows LLM agents to perform complex graph operations via tool calls.

**Key Capabilities**:

**Node Operations**:
- [ ] Find and link related nodes (semantic search + auto-linking)
- [ ] Recalculate confidence (context-aware re-analysis)
- [ ] Summarize node (LLM-generated summary)
- [ ] Summarize with context (include relationships)
- [ ] Search web for evidence (supporting/contradicting)
- [ ] Reclassify node type (Observation → Hypothesis, etc.)

**Edge Operations**:
- [ ] Recalculate relationship confidence
- [ ] Reclassify relationship type
- [ ] Summarize relationship (explain connection)
- [ ] Merge related nodes (with confirmation)

**Infrastructure**:
- [ ] Tool layer architecture (`backend/app/tools/`)
- [ ] Tool registration and discovery system
- [ ] User confirmation system for destructive operations
- [ ] Activity feed integration for tool execution
- [ ] LangGraph agent for natural language tool selection

**Safety**:
- All destructive operations (delete, merge) require user confirmation
- LLM receives feedback about user approval/denial decisions
- Comprehensive audit trail in Activity Feed

**Implementation Phases**:
1. Core tool infrastructure & registration
2. Node analysis tools
3. Edge analysis tools
4. Web search integration
5. User confirmation system
6. LangGraph agent interface

---

## Future Phases

### Phase 8: Real-Time Updates

- WebSocket connection for live activity feed
- Broadcast events on background task completion
- Connection status indicator
- Review queue for suggested connections

### Phase 9: Feedback Loop & Learning

- Store feedback in PostgreSQL
- Track approval/rejection rates
- Adjust confidence thresholds based on feedback
- Analyze patterns in rejections (training data for future model improvements)

### Phase 10: Natural Language Querying

- LLM interprets questions about the graph
- Generates Cypher queries dynamically (GraphCypherQAChain)
- Synthesizes answers from graph data
- Examples: "What supports hypothesis X?", "Show gaps in topic Y"

### Phase 11: User Authentication

- FastAPI-Users integration (OAuth 2.1 planned)
- JWT token authentication
- Multi-user support
- Activity attribution per user

### Phase 12: MCP Server

**Goal**: Expose ThoughtLab tools via Model Context Protocol.

> 📄 **Implementation Plan**: See [langchain_implementation.md](./langchain_implementation.md#mcp-server-integration)

- Separate `mcp-server/` package
- Thin wrapper tools calling ThoughtLab API
- Compatible with Claude Desktop, Cursor, etc.
- Publish to PyPI as `thoughtlab-mcp`

### Phase 13: Chrome Extension

**Goal**: Capture web content directly into the knowledge graph.

> 📄 **Implementation Plan**: See [langchain_implementation.md](./langchain_implementation.md#chrome-extension)

- Context menu: "Add as Observation", "Save Page as Source"
- Popup with recent activity and quick search
- Optional sidebar with graph view
- Publish to Chrome Web Store

---

## Optional/Deferred

### Background Job Processing (ARQ/Celery)

**Status**: Deferred until proven necessary by real-world usage data

**Rationale**: Current synchronous AI processing (3-5s per node) is acceptable for single-user research workflows. We're focusing on making AI operations more powerful and user-controlled rather than optimizing for scale prematurely.

**Decision Criteria** - Implement when:
- ❌ Users report API timeouts during node creation
- ❌ Processing time exceeds 10 seconds regularly
- ❌ Multi-user deployments show resource contention
- ❌ Batch operations needed (re-analyze entire graph)
- ❌ Performance monitoring shows clear bottleneck

**Implementation Plan** (when needed):
> 📄 See [langchain_implementation.md](./langchain_implementation.md#arq-background-processing-upgrade)

- ARQ (async Redis queue) or Celery
- Worker service in Docker Compose
- Job status monitoring in Activity Feed
- Retry logic and error handling

### Phase 14: Polish & Performance

- Comprehensive error handling
- Optimized loading states
- Graph performance for large datasets
- Full test coverage

---

## Backlog

Features not yet scheduled:

### Graph Features
- **Graph Filtering**: Filter by node type, time range, confidence
- **Graph Search**: Search nodes by text within visualization
- **Layout Switching**: Toggle between force-directed, hierarchical, circular
- **Keyboard Shortcuts**: Common actions accessible via keyboard
- **Undo/Redo**: Revert recent changes
- **Export**: Export graph as JSON or image
- **Batch Operations**: Create multiple nodes/relationships at once
- **Import**: Import data from external sources

### Additional Integrations
- **CLI Tool**: Command-line interface for ThoughtLab operations
- **Slack Bot**: Add observations and query graph from Slack
- **Obsidian Plugin**: Sync with Obsidian vault
- **VS Code Extension**: Quick capture while coding
- **API Webhooks**: Trigger actions on graph changes

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
