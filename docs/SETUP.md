# Setup Guide

## Prerequisites

- **Python 3.11+** (pyenv recommended)
- **Node.js 18+** (nvm recommended)
- **Docker & Docker Compose**
- **uv** (Python package manager) — see [DEPENDENCIES.md](./DEPENDENCIES.md)

### Verify Prerequisites

```bash
python --version   # 3.11+
node --version     # 18+
docker --version
uv --version       # Install: curl -LsSf https://astral.sh/uv/install.sh | sh
```

---

## Quick Start

### 1. Start Docker Services

```bash
docker-compose up -d
docker-compose ps  # Wait for services to be healthy
```

Services:
- **Neo4j Browser**: http://localhost:7474 (neo4j / research_graph_password)
- **Redis**: localhost:6379

### 2. Backend Setup

```bash
cd backend

# Create virtual environment
uv venv
source .venv/bin/activate  # Windows: .\.venv\Scripts\Activate.ps1

# Install dependencies
uv pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env: add SECRET_KEY (generate with: python -c "import secrets; print(secrets.token_urlsafe(32))")
# Edit .env: add THOUGHTLAB_OPENAI_API_KEY for AI features (see AI Configuration below)
```

### 3. Frontend Setup

```bash
cd frontend
npm install
cp .env.example .env  # Optional: configure API URL if not localhost
```

### 4. Initialize Neo4j

Run in Neo4j Browser (http://localhost:7474):

```cypher
// Unique constraints
CREATE CONSTRAINT observation_id IF NOT EXISTS FOR (o:Observation) REQUIRE o.id IS UNIQUE;
CREATE CONSTRAINT hypothesis_id IF NOT EXISTS FOR (h:Hypothesis) REQUIRE h.id IS UNIQUE;
CREATE CONSTRAINT source_id IF NOT EXISTS FOR (s:Source) REQUIRE s.id IS UNIQUE;
CREATE CONSTRAINT concept_id IF NOT EXISTS FOR (c:Concept) REQUIRE c.id IS UNIQUE;
CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE;

// Text indexes
CREATE TEXT INDEX observation_text IF NOT EXISTS FOR (o:Observation) ON o.text;
CREATE TEXT INDEX hypothesis_claim IF NOT EXISTS FOR (h:Hypothesis) ON h.claim;

// Temporal indexes
CREATE INDEX observation_created IF NOT EXISTS FOR (o:Observation) ON o.created_at;
```

---

## Running the Application

### Backend

```bash
cd backend
source .venv/bin/activate
uvicorn app.main:app --reload
```

- API: http://localhost:8000
- Docs: http://localhost:8000/docs
- Health: http://localhost:8000/health

### Frontend

```bash
cd frontend
npm run dev
```

- App: http://localhost:5173

---

## Convenience Scripts

From project root:

```bash
./start.sh    # Start all services
./stop.sh     # Stop all services
./restart.sh  # Restart all services
```

---

## Troubleshooting

**Port conflicts**: Change ports in `docker-compose.yml`

**Connection errors**:
- Verify Docker services: `docker-compose ps`
- Check `.env` credentials match `docker-compose.yml`
- Neo4j takes ~30 seconds to fully start

**Python not found**: Use `python3` instead of `python`

**uv not found**: Restart terminal after installation, or add `~/.cargo/bin` to PATH

---

## AI Configuration

ThoughtLab uses LangChain and OpenAI for AI-powered features:
- **Embedding Generation**: Converts node content to vector embeddings
- **Similarity Search**: Finds related content using Neo4j vector indexes
- **Relationship Classification**: Uses LLM to identify relationship types
- **Automated Suggestions**: Creates relationship suggestions based on confidence

### Required Environment Variables

```bash
# OpenAI API Key (required for AI features)
THOUGHTLAB_OPENAI_API_KEY=sk-your-openai-api-key
```

Get your API key at: https://platform.openai.com/api-keys

### Optional Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `THOUGHTLAB_LLM_MODEL` | `gpt-4o-mini` | LLM model for classification |
| `THOUGHTLAB_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `THOUGHTLAB_EMBEDDING_DIMENSIONS` | `1536` | Embedding vector size |
| `THOUGHTLAB_AUTO_CREATE_THRESHOLD` | `0.8` | Auto-create relationships above this confidence |
| `THOUGHTLAB_SUGGEST_THRESHOLD` | `0.6` | Create suggestions above this confidence |
| `THOUGHTLAB_SIMILARITY_MIN_SCORE` | `0.5` | Minimum similarity for candidates |
| `THOUGHTLAB_MAX_SIMILAR_NODES` | `20` | Max candidates to analyze |

### Confidence Thresholds

| Confidence | Action |
|------------|--------|
| ≥ 0.8 | Auto-create relationship (marked as `created_by: system-llm`) |
| 0.6 - 0.8 | Create suggestion in Activity Feed for user review |
| < 0.6 | Discard silently |

### Without AI Configuration

If `THOUGHTLAB_OPENAI_API_KEY` is not set, ThoughtLab works normally but:
- No automatic embedding generation
- No AI-powered relationship suggestions
- Manual relationship creation only

---

## Next Steps

- [PROJECT_MAP.md](./PROJECT_MAP.md) — Code structure reference
- [TESTING.md](./TESTING.md) — Run tests
- [DEPENDENCIES.md](./DEPENDENCIES.md) — Package management
