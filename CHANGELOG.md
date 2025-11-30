# Changelog

All notable changes to ThoughtLab will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Architecture decision documentation reorganized by concern (previously ADR log format)
- Vector embedding strategy documented (LangChain + OpenAI + Neo4j vectors)
- Interim decision markers (🔄) for planned future changes
- Consolidated `DEPENDENCIES.md` for package management reference
- Cursor rules for commit and package management workflows

### Changed
- `02-ARCHITECTURE_DECISIONS.md` restructured from ADR log to flat reference format
- AI strategy unified: LangChain + OpenAI for both LLM and embeddings (replacing LiteLLM)
- Documentation consolidated: removed redundant setup/dependency docs

### Removed
- `BACKEND_SETUP.md` - merged into `SETUP.md`
- `FRONTEND_SETUP.md` - merged into `SETUP.md`
- `DEPENDENCY_STRATEGY.md`, `DEPENDENCY_SUMMARY.md`, `DEPENDENCY_UPDATES.md` - consolidated into `DEPENDENCIES.md`
- `UV_SETUP.md` - merged into `SETUP.md`
- `backend/README_TESTS.md`, `frontend/README_TESTS.md` - merged into `TESTING.md`

### Planned (🔄 Interim Decisions)
- **LLM/Embeddings**: Currently OpenAI-only; will add provider selection via settings (Anthropic, Ollama)
- **Authentication**: Currently FastAPI-Users + JWT; will implement OAuth 2.1

---

## [0.3.0] - 2024-XX-XX (Current Development)

### Added
- Link management for nodes (clickable URLs across all node types)
- Entity node type with full CRUD operations
- Concept node type with full CRUD operations
- Source node type with full CRUD operations
- Relationship management endpoints (create, update, delete)
- Relation styles customization in settings
- Node color customization per type
- Dark mode support across all components
- Retry logic for service connections
- Health check mechanisms with proper error handling

### Changed
- Settings management refactored with relation styles and node colors
- GraphVisualizer uses dynamic styles from user preferences
- API service methods improved for better settings handling
- Docker configuration refined

---

## [0.2.0] - Foundation Release

### Added
- **Backend Foundation**
  - FastAPI application with async support
  - Neo4j database connection and driver
  - Redis connection for caching/queues
  - Pydantic models for all node types (Observation, Hypothesis, Source, Concept, Entity)
  - Graph service with full CRUD operations
  - RESTful API routes with OpenAPI auto-documentation
  - Health check endpoint

- **Frontend Foundation**
  - React 18 + TypeScript + Vite setup
  - React Query for server state management
  - TypeScript-typed API client
  - Cytoscape.js graph visualization
  - Activity feed component
  - Node inspector panel
  - Create/edit modals for nodes
  - Settings modal with persistence
  - Tailwind CSS styling

- **Infrastructure**
  - Docker Compose for Neo4j and Redis
  - Python virtual environment with uv
  - Shell scripts for start/stop/restart
  - Neo4j initialization script with constraints and indexes

---

## [0.1.0] - Initial Setup

### Added
- Project structure established
- Basic documentation framework
- Git repository initialized
- Initial dependency selection documented

---

## Version History Summary

| Version | Focus | Status |
|---------|-------|--------|
| 0.1.0 | Project setup | ✅ Complete |
| 0.2.0 | Backend + Frontend foundation | ✅ Complete |
| 0.3.0 | Node types, relationships, settings | 🚧 In Progress |
| 0.4.0 | AI Integration (LangChain + OpenAI) | 📋 Planned |
| 0.5.0 | Vector embeddings + semantic search | 📋 Planned |
| 0.6.0 | Connection discovery workflows | 📋 Planned |
| 1.0.0 | Production-ready release | 📋 Planned |

---

## Upcoming Milestones

### 0.4.0 - AI Integration
- [ ] LangChain integration for LLM calls
- [ ] OpenAI configuration and API setup
- [ ] Connection analysis service
- [ ] Relationship classification

### 0.5.0 - Vector Search
- [ ] Neo4j vector indexes for all node types
- [ ] OpenAI embeddings via LangChain
- [ ] Embedding service implementation
- [ ] Similarity search API endpoints
- [ ] Hybrid graph+vector queries

### 0.6.0 - Discovery Workflows
- [ ] Background job processing (ARQ)
- [ ] Automatic connection suggestions
- [ ] User feedback collection
- [ ] Activity feed real-time updates (WebSocket)

[Unreleased]: https://github.com/study-flamingo/thoughtlab/compare/main...HEAD
[0.3.0]: https://github.com/study-flamingo/thoughtlab/compare/v0.2.0...main
[0.2.0]: https://github.com/study-flamingo/thoughtlab/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/study-flamingo/thoughtlab/releases/tag/v0.1.0

