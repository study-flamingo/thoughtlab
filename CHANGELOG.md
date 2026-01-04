# Changelog

All notable changes to ThoughtLab will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

#### Tool Layer Architecture Refactor
- **Modular Tool Service** (`backend/app/services/tools/`)
  - `service.py` - ToolService facade class
  - `base.py` - Shared infrastructure (config, LLM, Neo4j helpers)
  - `operations/node_analysis.py` - find_related, summarize, confidence ops
  - `operations/node_modification.py` - reclassify, merge, web_evidence ops
  - `operations/relationship_analysis.py` - edge summarize, confidence, reclassify ops

- **Shared Tool Definitions** (`backend/app/tools/`)
  - `tool_definitions.py` - 12 tool definitions with metadata, parameters, execution modes
  - `tool_registry.py` - Registry for MCP + LangGraph tool discovery
  - Defines MCPExecutionMode (SYNC/ASYNC) and dangerous tool flags

- **Job Queue & Report Storage**
  - `backend/app/services/job_service.py` - Redis-based job queue for async operations
  - `backend/app/services/report_service.py` - Redis storage for LangGraph results
  - `backend/app/models/job_models.py` - Job and Report Pydantic models

- **MCP Server Refactor**
  - MCP server now mounted at `/mcp` in FastAPI app
  - `backend/app/mcp/mcp_tools.py` - Tool wrappers calling ToolService directly (no HTTP)
  - Dangerous tools gated by `THOUGHTLAB_MCP_ADMIN_MODE` env var

- **LangGraph Agent Refactor**
  - `backend/app/agents/agent_tools.py` - Tools calling ToolService directly (no HTTP)
  - Results saved to ReportService for later viewing
  - All 10 LangGraph tools now available

- **Tool Models** (`backend/app/models/tool_models.py`)
  - Centralized Pydantic request/response models for all AI tools

#### LLM-Powered AI Tools (Phase 7)
- **Toast Notification System** (`frontend/src/components/Toast.tsx`)
  - ToastProvider context with `showToast(message, type, duration)`
  - Four toast types: success (green), error (red), warning (amber), info (blue)
  - Slide-in animation with auto-dismiss

- **AI Tools Section Component** (`frontend/src/components/AIToolsSection.tsx`)
  - Collapsible section with purple "AI Tools" header
  - AIToolButton component with loading states

- **Tool TypeScript Types** (`frontend/src/types/tools.ts`)
  - Request/response types for all 10 tools

- **Node Inspector AI Tools** (6 tools)
  - Find Related Nodes - semantic similarity search
  - Summarize - LLM-generated summary
  - Summarize with Context - includes relationship context
  - Recalculate Confidence - re-evaluates based on graph context
  - Reclassify Node - change node type with dropdown selector
  - Search Web for Evidence - placeholder (requires TAVILY_API_KEY)

- **Relation Inspector AI Tools** (3 tools)
  - Summarize Relationship - explains connection in plain language
  - Recalculate Confidence - re-evaluates relationship strength
  - Reclassify Relationship - change type with dropdown or AI suggestion

- **New Backend Tool Endpoints** (`backend/app/api/routes/tools.py`)
  - `POST /api/v1/tools/nodes/{id}/reclassify` - Change node type
  - `POST /api/v1/tools/nodes/{id}/search-web-evidence` - Web search placeholder
  - `POST /api/v1/tools/nodes/merge` - Merge two nodes
  - `POST /api/v1/tools/relationships/{id}/recalculate-confidence` - Recalculate edge confidence
  - `POST /api/v1/tools/relationships/{id}/reclassify` - Change relationship type

- **Tool Service Methods** (`backend/app/services/tool_service.py`)
  - `reclassify_node()` - Changes node label in Neo4j
  - `search_web_evidence()` - Placeholder for Tavily integration
  - `merge_nodes()` - Combines nodes and transfers relationships
  - `recalculate_edge_confidence()` - LLM-based relationship strength evaluation
  - `reclassify_relationship()` - Changes relationship type with AI suggestion option

### Changed
- **Architecture**: MCP and LangGraph now call ToolService directly (in-process, no HTTP)
- **Architecture**: Tool definitions shared between MCP and LangGraph via registry
- API client extended with 10 new tool methods
- NodeInspector now includes AI Tools section with 6 tools
- RelationInspector now includes AI Tools section with 3 tools
- Tool capabilities endpoint updated to reflect all implemented tools
- MCP server now accessible at `/mcp` endpoint with Streamable HTTP transport

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
| 0.3.0 | LLM-Powered Graph Operations (Phase 7) | 📋 Planned |
| 0.4.0 | Real-time Updates (Phase 8) | 📋 Planned |
| 1.0.0 | Production-ready release | 📋 Planned |

---

## Upcoming Milestones

### 0.3.0 - LLM-Powered Graph Operations (Phase 7)

**Focus**: Unified tool layer enabling LLM agents to intelligently operate on the knowledge graph via tool calls.

**Design Decision**: Deferring ARQ/background processing in favor of powerful, user-controlled LLM operations. Current synchronous AI processing (3-5s) is acceptable; we're optimizing for capability over scale.

**Node Operations**:
- [ ] Find and link related nodes (semantic search + auto-linking)
- [ ] Recalculate node confidence (context-aware re-analysis)
- [ ] Summarize node with/without relationship context
- [ ] Search web for supporting/contradicting evidence
- [ ] Reclassify node type (Observation → Hypothesis, etc.)

**Edge Operations**:
- [ ] Recalculate relationship confidence
- [ ] Reclassify relationship type
- [ ] Summarize relationship (explain connection)
- [ ] Merge related nodes (with user confirmation)

**Infrastructure**:
- [ ] Tool layer architecture (`backend/app/tools/`)
- [ ] Tool registration and discovery system
- [ ] User confirmation system for destructive operations
- [ ] Activity feed integration for tool execution
- [ ] LangGraph agent for natural language interface

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

