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

**Goal**: Track system events and provide foundation for background processing.

**Completed**: Activity data model, ActivityService with CRUD, interactive ActivityFeed component, ProcessingService orchestration, chunking utilities, comprehensive test coverage

### Phase 6: LangChain + LangGraph AI Integration ✅

**Goal**: AI-powered embedding generation and relationship discovery.

**Completed**: LangChain/LangGraph dependencies, AI module (config, embeddings, similarity search, relationship classifier), AI workflow orchestration, Neo4j vector indexes

**How It Works**:
- New nodes → Generate embeddings → Find similar nodes → LLM classifies relationships
- Confidence ≥0.8: auto-create | 0.6-0.8: suggest to user | <0.6: discard

> 📄 **Details**: [02-ARCHITECTURE_DECISIONS.md](./02-ARCHITECTURE_DECISIONS.md#langgraph--langchain--openai)

### Phase 7: LLM-Powered Graph Operations ✅

**Goal**: Enable LLM agents to intelligently manipulate the knowledge graph through natural language.

> 📄 **Detailed Architecture**: [02-ARCHITECTURE_DECISIONS.md](./02-ARCHITECTURE_DECISIONS.md#llm-powered-graph-operations)
> 📄 **Backend API Spec**: [PHASE_7_API_SPEC.md](./PHASE_7_API_SPEC.md)
> 📄 **LangGraph Integration**: [PHASE_8_LANGGRAPH_INTEGRATION.md](./PHASE_8_LANGGRAPH_INTEGRATION.md)
> 📄 **Implementation Summary**: [COMPLETE_IMPLEMENTATION_SUMMARY.md](./COMPLETE_IMPLEMENTATION_SUMMARY.md)

**Approach**: Built a three-layer modular architecture with complete separation of concerns via HTTP REST API.

**Implementation**:

- **Backend API Layer**: 7 REST endpoints for all LLM operations (find related, summarize, recalculate confidence, etc.)
- **LangGraph Agent Layer**: ReAct agent with 5 tools calling backend via HTTP, intelligent tool selection
- **MCP Server Layer**: 6 MCP tools exposing same operations for Claude Desktop integration

**Core Capabilities**:

- **Node Operations**: Find semantically similar nodes, generate AI summaries with/without context, recalculate confidence based on relationships
- **Relationship Operations**: Summarize connections, explain relationships
- **Health Monitoring**: Backend API health checks
- **Interface Options**: Direct API, LangGraph agents (Python), MCP clients (Claude Desktop)

**Status**: Complete and validated - all three layers implemented with comprehensive documentation

### Phase 12: MCP Server ✅

**Goal**: Expose ThoughtLab tools via Model Context Protocol for Claude Desktop and other MCP-compatible apps.

> 📄 **Complete Guide**: [MCP_SERVER_GUIDE.md](./MCP_SERVER_GUIDE.md)

**Implementation**: FastMCP-based server exposing 6 tools via stdio transport, using same backend API as LangGraph agents.

**Features**:
- Claude Desktop integration ready
- Same backend API ensuring consistency
- Complete separation from agent layer
- Fully validated and documented

**Status**: Complete - ready for use with Claude Desktop

---

## In Progress

*No phases currently in progress*

---

## Future Phases

### Phase 8: Real-Time Updates
WebSocket connection for live activity feed, event broadcasting, connection status indicator, review queue for suggestions

### Phase 9: Feedback Loop & Learning
Store user feedback, track approval/rejection rates, adjust confidence thresholds dynamically, analyze patterns for model improvement

### Phase 10: Natural Language Querying
LLM interprets questions about the graph, generates Cypher queries dynamically, synthesizes answers from graph data

### Phase 11: User Authentication
Multi-user support with FastAPI-Users, JWT tokens, OAuth 2.1, activity attribution per user

### Phase 13: Chrome Extension
Capture web content directly into knowledge graph via context menu, popup dashboard, optional sidebar

> 📄 **Details**: [langchain_implementation.md](./langchain_implementation.md#chrome-extension)

---

## Optional/Deferred

### Background Job Processing (ARQ/Celery)
**Status**: Deferred - synchronous AI processing (3-5s) acceptable for current use case

Implement when: API timeouts, >10s processing, multi-user contention, batch processing needs, or proven performance bottleneck

> 📄 **Details**: [02-ARCHITECTURE_DECISIONS.md](./02-ARCHITECTURE_DECISIONS.md#background-processing-deferred-arq-when-needed)

### Polish & Performance
Comprehensive error handling, optimized loading states, large graph performance, full test coverage

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
