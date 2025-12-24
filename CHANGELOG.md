# Changelog

All notable changes to ThoughtLab will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

No unreleased changes.

---

## [0.2.0-alpha] - 2025-11-30

This alpha release includes the complete Activity Feed system, background processing infrastructure, and LangChain/LangGraph AI integration (Phases 5-6 from the roadmap).

### Added

#### Activity Feed System
- **Activity model** (`backend/app/models/activity.py`) - Comprehensive activity types for tracking system events
  - Node/relationship lifecycle events
  - Processing status (chunking, embedding, analyzing)
  - LLM suggestions with approve/reject workflow
  - Confidence thresholds: auto-create (>0.8), suggest (0.6-0.8), discard (<0.6)
- **ActivityService** (`backend/app/services/activity_service.py`) - CRUD operations for activities
- **Activity API routes** (`backend/app/api/routes/activities.py`) - REST endpoints for activity management
- **Interactive ActivityFeed** (`frontend/src/components/ActivityFeed.tsx`) - Real-time feed with:
  - Polling for updates
  - Approve/reject buttons for suggestions
  - "View" navigation to relevant nodes
  - Processing progress display
  - Status badges for completed actions

#### Background Processing Infrastructure
- **ProcessingService** (`backend/app/services/processing_service.py`) - Orchestrates node analysis workflow:
  - Chunking → Embedding → Analysis → Suggestions
  - Activity feed status updates at each stage
  - Error handling with activity notifications
- **EmbeddingService stub** (`backend/app/services/embedding_service.py`) - Interface for LangChain integration
- **Chunking utility** (`backend/app/utils/chunking.py`) - RecursiveCharacterSplitter for long content
  - Configurable chunk size and overlap
  - Position tracking for reconstruction
  - Threshold detection (skip chunking for short content)

#### Schema Updates
- `Activity` node in Neo4j with indexes for efficient querying
- Activity TypeScript types (`frontend/src/types/activity.ts`)
- API service methods for activity operations

#### AI Integration (LangChain + LangGraph)
- **AI Module** (`backend/app/ai/`) - Complete LangChain integration:
  - `config.py` - AI configuration with env var support (`THOUGHTLAB_OPENAI_API_KEY`, models, thresholds)
  - `embeddings.py` - OpenAI embeddings via LangChain with Neo4j storage
  - `similarity.py` - Vector similarity search using Neo4j vector indexes
  - `classifier.py` - LLM-based relationship classification with structured Pydantic output
  - `workflow.py` - Main AI processing workflow orchestrator
- **Processing Service Update** - Delegates to AI workflow when configured
- **Neo4j Vector Indexes** - Added to `init.cypher` for all node types
- **Dependencies** - Added `langchain`, `langchain-openai`, `langchain-neo4j`, `langgraph`

#### Documentation
- **LangChain Implementation Plan** (`docs/langchain_implementation.md`) - Comprehensive guide for AI integration:
  - **Unified Tool Architecture** - Shared tool layer for LangGraph, MCP, and frontend
  - **LangGraph vs LangChain decision** - Why LangGraph for intelligent workflows
  - Architecture diagrams for processing workflow
  - Module structure for `backend/app/ai/` and `backend/app/tools/` packages
  - Component implementations (embeddings, similarity, classifier, workflow)
  - Configuration via `.env` file
  - ARQ background processing upgrade plan
  - **MCP Server plan** - Companion server for Claude Desktop integration
  - **Chrome Extension plan** - Quick capture from web pages
  - Testing strategy for AI components
- **SETUP.md** - Added AI configuration section with environment variables and thresholds
- **PROJECT_MAP.md** - Added AI module documentation

#### Test Coverage
- **Test runner script** (`test.sh`) - Unified test runner for both frontend and backend
  - `./test.sh` - Run all tests
  - `./test.sh backend` - Run only backend tests
  - `./test.sh frontend` - Run only frontend tests
  - `./test.sh --quick` - Run unit tests only (no Neo4j required)
- **Backend model tests** (`backend/tests/test_activity_models.py`) - 28 tests for Activity models
  - `ActivityType`, `ActivityStatus`, `SuggestionData`, `ProcessingData`
  - `ActivityCreate`, `ActivityUpdate`, `ActivityResponse`, `ActivityFilter`
  - `SuggestionThresholds` with confidence action logic
- **Backend service tests** (`backend/tests/test_activity_service.py`) - 24 tests for ActivityService
  - CRUD operations, filtering, suggestions workflow
  - Processing status updates, approve/reject logic
- **Backend API tests** (`backend/tests/test_api_activities.py`) - 18 tests for activity routes
- **Frontend activity tests** (`frontend/src/types/__tests__/activity.test.ts`) - 32 tests for type helpers
- **Frontend API tests** (`frontend/src/services/__tests__/api.test.ts`) - 28 tests including activity endpoints
- **Frontend ActivityFeed tests** (`frontend/src/components/__tests__/ActivityFeed.test.tsx`) - 28 tests
  - Loading/error states, activity display, approve/reject interactions
  - Polling behavior, navigation callbacks, timestamp formatting

### Changed
- Relationship types changed from enum to open strings (allows LLM-created relationship types)
- Existing tests updated for new model signatures (`HypothesisCreate.name`, relationship type strings)
- Added missing `ChunkUpdate` model to `nodes.py`
- Architecture decision documentation reorganized by concern (previously ADR log format)
- Vector embedding strategy documented (LangChain + OpenAI + Neo4j vectors)
- Interim decision markers (🔄) for planned future changes
- Consolidated `DEPENDENCIES.md` for package management reference
- Cursor rules for commit and package management workflows
- `02-ARCHITECTURE_DECISIONS.md` restructured from ADR log to flat reference format
- AI strategy unified: LangChain + OpenAI for both LLM and embeddings (replacing LiteLLM)
- Documentation consolidated: removed redundant setup/dependency docs
- ActivityFeed now receives `onSelectNode` callback for navigation

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

## [0.2.0] - 2024-XX-XX - Foundation Release

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
| 0.2.0-alpha | Activity Feed + AI Integration (Phases 5-6) | ✅ Complete |
| 0.3.0 | ARQ Background Processing (Phase 7) | 📋 Planned |
| 0.4.0 | Real-time Updates (Phase 8) | 📋 Planned |
| 1.0.0 | Production-ready release | 📋 Planned |

---

## Upcoming Milestones

### 0.3.0 - ARQ Background Processing (Phase 7)
- [ ] Add ARQ worker configuration
- [ ] Create job definitions for node processing
- [ ] Add worker service to Docker Compose
- [ ] Update API routes to enqueue jobs
- [ ] Monitor job status in Activity Feed

### 0.4.0 - Real-time Updates (Phase 8)
- [ ] WebSocket connection for live activity feed
- [ ] Broadcast events on background task completion
- [ ] Connection status indicator
- [ ] Review queue for suggested connections

### Future Enhancements
- [ ] Integration tests with mocked OpenAI
- [ ] End-to-end tests with real API (optional)
- [ ] Natural language querying (Phase 10)
- [ ] User authentication (Phase 11)
- [ ] MCP Server (Phase 12)
- [ ] Chrome Extension (Phase 13)

[Unreleased]: https://github.com/study-flamingo/thoughtlab/compare/main...HEAD
[0.2.0-alpha]: https://github.com/study-flamingo/thoughtlab/compare/v0.2.0...main
[0.2.0]: https://github.com/study-flamingo/thoughtlab/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/study-flamingo/thoughtlab/releases/tag/v0.1.0

