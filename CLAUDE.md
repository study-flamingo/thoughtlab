# Claude Code Guidelines for ThoughtLab

This document provides guidelines for using Claude Code effectively with the ThoughtLab project.

---

## Quick Context Reference

When working on ThoughtLab, refer to these documentation files for context:

### Core Documentation
- **[DEVELOPMENT_GUIDE.md](./DEVELOPMENT_GUIDE.md)** - Comprehensive developer guide (architecture, setup, workflows, extending)
- **[PROJECT_MAP.md](./docs/PROJECT_MAP.md)** - Code structure and file locations
- **[README.md](./README.md)** - Project overview and quickstart

### Detailed References
- **[SETUP.md](./docs/SETUP.md)** - Detailed installation instructions
- **[TESTING.md](./docs/TESTING.md)** - Testing guide
- **[DEPENDENCIES.md](./docs/DEPENDENCIES.md)** - Package management with uv
- **[MCP_SERVER_GUIDE.md](./docs/MCP_SERVER_GUIDE.md)** - MCP server setup and usage
- **[PHASE_7_API_SPEC.md](./docs/PHASE_7_API_SPEC.md)** - Backend API reference
- **[PHASE_8_LANGGRAPH_INTEGRATION.md](./docs/PHASE_8_LANGGRAPH_INTEGRATION.md)** - LangGraph integration guide

---

## General Principles

### 1. Context-Aware Development

- **Always read before writing**: Never propose changes to code you haven't read
- **Understand existing patterns**: Follow the patterns already established in the codebase
- **Check documentation first**: Refer to PROJECT_MAP.md and DEVELOPMENT_GUIDE.md to understand where code lives

### 2. Collaborative Decision Making

- **Ask before breaking changes**: Always confirm with the user before making potentially breaking changes
- **Design discussions**: For architectural decisions or large refactors, ask for user input
- **Explain trade-offs**: When suggesting changes, explain the benefits and potential downsides

### 3. Documentation-Driven Development

Keep documentation synchronized with code changes. When making changes, update:

1. **[CHANGELOG.md](./CHANGELOG.md)** - User-facing changes
2. **[DEVELOPMENT_GUIDE.md](./DEVELOPMENT_GUIDE.md)** - If architecture or workflows change
3. **[PROJECT_MAP.md](./docs/PROJECT_MAP.md)** - If new files/modules are added
4. **[SETUP.md](./docs/SETUP.md)** - If setup process changes
5. **[TESTING.md](./docs/TESTING.md)** - If testing approach changes
6. **[README.md](./README.md)** - If features or quickstart instructions change
7. **Context7** - When writing code that relies on 3rd party libraries, ALWAYS use context7 to see the latest docs.
---

## Development Principles

### Modularity First

Design features with modularity in mind for easy expansion with minimal refactoring:

- **Single Responsibility**: Each module/component does one thing well
- **Clear Interfaces**: Pydantic models (backend), TypeScript types (frontend)
- **Separation of Concerns**: Routes → Services → Database connectors
- **Minimal Coupling**: Props/callbacks over global state
- **Easy Extension Points**: New features require isolated changes

See [DEVELOPMENT_GUIDE.md: Development Principles](./DEVELOPMENT_GUIDE.md#development-principles) for detailed guidelines.

### Avoid Over-Engineering

- Don't add features beyond what was requested
- Don't refactor unrelated code during bug fixes
- Don't add error handling for impossible scenarios
- Don't create abstractions for single-use code
- Three similar lines > premature abstraction

### Code Quality Standards

- Write tests for new features (pytest/vitest)
- Use type hints throughout (Python 3.11+, TypeScript)
- Keep functions small and focused (<50 lines ideal)
- Document non-obvious decisions in comments
- Follow existing codebase patterns

---

## Technology-Specific Guidelines

### Backend (Python + FastAPI)

**Package Management**: Always use **uv** for Python package management
```bash
cd backend
uv pip install package-name
uv sync --all-extras
```

**Architecture**:
- Routes handle HTTP (validate, delegate, respond)
- Services handle business logic
- Database connectors handle data access
- Use async/await throughout

**Testing**:
```bash
cd backend
pytest                    # Run tests
pytest --cov=app          # With coverage
```

### Frontend (React + TypeScript)

**Package Management**: Use npm
```bash
cd frontend
npm install package-name
```

**Architecture**:
- Components are focused and reusable
- React Query for server state
- Props down, callbacks up
- TypeScript for type safety

**Testing**:
```bash
cd frontend
npm test                  # Run tests
npm run test:coverage     # With coverage
```

### AI Integration (LangChain/LangGraph)

ThoughtLab has a **three-layer AI architecture**:

1. **Backend API Layer**: REST endpoints in `backend/app/api/routes/tools.py`
2. **LangGraph Agent Layer**: Intelligent agents in `backend/app/agents/`
3. **MCP Server Layer**: Model Context Protocol server in `backend/app/mcp/`

All layers share the same backend API. See [DEVELOPMENT_GUIDE.md: AI Integration](./DEVELOPMENT_GUIDE.md#ai-integration) for details.

---

## Common Workflows

### Adding a New Feature

1. **Read documentation**: Check PROJECT_MAP.md for similar features
2. **Plan approach**: Discuss architecture with user if significant
3. **Implement**: Follow modularity principles
4. **Test**: Write tests (backend and/or frontend)
5. **Document**: Update relevant docs
6. **Commit**: Clear commit message following project conventions

### Adding a New Node Type

See [DEVELOPMENT_GUIDE.md: Adding a New Node Type](./DEVELOPMENT_GUIDE.md#adding-a-new-node-type) for step-by-step guide covering:
- Models (Pydantic)
- Service layer (graph_service.py)
- API routes
- Frontend types and UI
- Neo4j constraints

### Adding a New API Endpoint

See [DEVELOPMENT_GUIDE.md: Adding a New API Endpoint](./DEVELOPMENT_GUIDE.md#adding-a-new-api-endpoint) for step-by-step guide.

### Modifying the Database Schema

See [DEVELOPMENT_GUIDE.md: Modifying the Graph Schema](./DEVELOPMENT_GUIDE.md#modifying-the-graph-schema) for guidance on:
- Updating Pydantic models
- Modifying Cypher queries
- Updating frontend types
- Migration considerations

---

## Git & GitHub

### Branch Strategy

- Develop on feature branches
- Use descriptive branch names (e.g., `feature/add-node-type`, `fix/api-error`)
- Keep commits focused and atomic

### Commit Messages

Follow the project's commit message conventions:
- Use conventional commits format: `type(scope): description`
- Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`
- Examples:
  - `feat(api): add endpoint for bulk node creation`
  - `fix(frontend): resolve graph rendering issue`
  - `docs: update setup guide with AI configuration`

### Before Pushing

1. **Run tests**: Ensure all tests pass
   ```bash
   cd backend && pytest
   cd frontend && npm test -- --run
   ```

2. **Update documentation**: As listed above

3. **Verify no secrets**: Check for API keys or credentials

---

## Security Considerations

- **Never commit secrets**: Use environment variables
- **Validate user input**: Use Pydantic models for validation
- **Avoid SQL injection**: Use parameterized queries (Neo4j driver handles this)
- **CORS properly configured**: Only allow necessary origins
- **API authentication**: When implemented, protect sensitive endpoints

See [SECURITY.md](./SECURITY.md) for full security policy.

---

## Troubleshooting Tips

### Backend Issues

```bash
# Check services are running
docker-compose ps

# Check backend logs
cd backend
uvicorn app.main:app --reload

# Verify environment
cat .env
```

### Frontend Issues

```bash
# Check API connection
curl http://localhost:8000/health

# Verify environment
cat frontend/.env
```

### AI Integration Issues

```bash
# Validate backend API
cd backend
python validate_tools.py

# Validate LangGraph agent
python validate_agent.py

# Validate MCP server
python validate_mcp.py
```

See [DEVELOPMENT_GUIDE.md: Troubleshooting](./DEVELOPMENT_GUIDE.md#troubleshooting) for comprehensive troubleshooting.

---

## Best Practices for Claude Code

### Reading Files

- **Use Read tool**: Don't use `cat` via Bash for reading files
- **Read before editing**: Always read files before proposing changes
- **Read related files**: Understand context by reading related components

### Making Changes

- **Use Edit tool**: For targeted changes to existing files
- **Use Write tool**: Only for new files (prefer editing existing)
- **Test changes**: Run tests after significant changes

### Searching

- **Use Grep**: For searching code content
- **Use Glob**: For finding files by pattern
- **Use PROJECT_MAP.md**: As first reference for finding code locations

### Task Management

- **Use TodoWrite**: For multi-step tasks to track progress
- **Mark completed**: Mark todos completed immediately after finishing
- **Keep focused**: Break large tasks into smaller, trackable steps

---

## Quick Reference

### Project Structure

```
thoughtlab/
├── backend/              # Python FastAPI backend
│   ├── app/
│   │   ├── api/routes/  # API endpoints
│   │   ├── services/    # Business logic
│   │   ├── ai/          # LangChain integration
│   │   ├── agents/      # LangGraph agents
│   │   └── mcp/         # MCP server
│   └── tests/           # Backend tests
├── frontend/            # React TypeScript frontend
│   └── src/
│       ├── components/  # React components
│       ├── services/    # API client
│       └── types/       # TypeScript types
└── docs/                # Documentation

```

### Key Commands

```bash
# Setup
./setup.sh                          # One-command setup

# Development
./start.sh                          # Start all services
./stop.sh                           # Stop all services

# Backend
cd backend
uvicorn app.main:app --reload       # Dev server
pytest                              # Run tests
python validate_tools.py            # Validate AI tools

# Frontend
cd frontend
npm run dev                         # Dev server
npm test                            # Run tests

# Infrastructure
docker-compose up -d                # Start Neo4j + Redis
docker-compose ps                   # Check status
```

---

## Development Environment

### CLI Tools Container

For consistent development with all necessary CLI tools (git, GitHub CLI, Railway CLI), use the development container:

```bash
# Enter dev environment with all tools
./scripts/dev.sh

# Inside the container - all tools available
git status
gh repo sync
railway logs
```

**Why this exists:** Running `git` or `gh` through automated tools (like OpenClaw's exec) often fails because those tools don't have access to the host's SSH keys or GitHub credentials. The dev container mounts your host's credentials, so authentication works seamlessly.

**What's included:**
- `git` - Version control
- `gh` - GitHub CLI (with your host's authentication)
- `railway` - Railway deployment CLI
- `node`/`npm` - Frontend tooling
- `python`/`uv` - Backend tooling

**Authentication:** The dev container automatically mounts:
- `~/.gitconfig` - Git configuration
- `~/.ssh` - SSH keys
- `~/.config/gh` - GitHub CLI credentials
- `~/.railway` - Railway credentials

### Git Workflow

1. **Enter dev environment:** `./scripts/dev.sh`
2. **Make changes** to code
3. **Test locally** before committing
4. **Commit and push:**
   ```bash
   git add .
   git commit -m "feat: description"
   git push origin main
   # or: gh repo sync
   ```
5. **Check Railway deployment:** `railway logs`

---

## Getting Help

- **Documentation**: Check DEVELOPMENT_GUIDE.md first
- **Code locations**: Use PROJECT_MAP.md to find relevant files
- **API reference**: See API docs at http://localhost:8000/docs (when running)
- **Ask user**: When uncertain about architecture or approach

---

**Last Updated**: 2026-02-14
