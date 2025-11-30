# Dependency Management

## Strategy

We use a **minimal constraint** approach:

1. **Core packages only**: Only packages directly imported are constrained
2. **Minimum versions**: Use `>=` constraints to specify minimum required versions
3. **Let uv resolve**: Transitive dependencies are resolved automatically

### Benefits

- **Faster resolution**: uv finds compatible versions quickly
- **Fewer conflicts**: Less chance of over-constraining
- **Security updates**: Transitive dependencies update automatically
- **Simpler maintenance**: Only update core packages

---

## Package Manager: uv

This project uses [uv](https://github.com/astral-sh/uv) for Python package management—a fast Rust-based installer that's 10-100x faster than pip.

### Installation

**Linux/macOS/WSL:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell):**
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Usage

```bash
cd backend

# Create virtual environment
uv venv

# Activate
source .venv/bin/activate  # Linux/macOS
.\.venv\Scripts\Activate.ps1  # Windows

# Install dependencies
uv pip install -r requirements.txt

# Update all dependencies
uv pip install --upgrade -r requirements.txt

# Add a new package
uv pip install package-name
```

### Lock File (Production)

For reproducible deployments:
```bash
uv pip compile requirements.txt -o requirements.lock
```

---

## Current Versions

### Backend (Python)

| Package | Version | Purpose |
|---------|---------|---------|
| fastapi | 0.121.2 | Web framework |
| uvicorn | 0.32.1 | ASGI server |
| neo4j | 5.26.0 | Graph database driver |
| asyncpg | 0.30.0 | PostgreSQL async driver |
| sqlalchemy | 2.0.36 | ORM |
| redis | 5.2.1 | Redis client |
| sentence-transformers | >=2.3.1,<6.0.0 | ML embeddings |
| litellm | 1.62.3 | LLM integration |
| arq | 0.29.0 | Background task queue |
| pytest | 8.3.4 | Testing |
| httpx | 0.28.1 | HTTP client (tests) |

### Frontend (Node.js)

| Package | Version | Purpose |
|---------|---------|---------|
| react | 18.3.1 | UI framework |
| @tanstack/react-query | 5.62.11 | Server state |
| axios | 1.7.9 | HTTP client |
| cytoscape | 3.31.0 | Graph visualization |
| zustand | 4.5.7 | Client state |
| vite | 5.4.21 | Build tool |
| vitest | 2.1.8 | Testing |
| typescript | 5.7.2 | Type safety |
| tailwindcss | 3.4.17 | Styling |

---

## Updating Dependencies

### Backend

```bash
cd backend
source .venv/bin/activate
uv pip install --upgrade -r requirements.txt
pytest  # Verify tests pass
```

### Frontend

```bash
cd frontend
npm update
npm test -- --run  # Verify tests pass
```

### Breaking Changes to Watch

Major versions available but not updated (require migration work):

- **React 19.x**: Breaking changes from 18.x
- **Vite 7.x**: Breaking changes from 5.x
- **ESLint 9.x**: New config format (eslint.config.js)
- **Tailwind CSS 4.x**: Breaking changes from 3.x
- **zustand 5.x**: Breaking changes from 4.x

---

*Last updated: 2025-01-16*

