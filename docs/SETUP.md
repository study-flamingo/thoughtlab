# Setup Instructions

## Prerequisites Check

✅ Python 3.12.3 detected  
✅ Docker and Docker Compose available  
⚠️ Node.js not detected - you'll need to install it

### Install Node.js (if needed)

**Option 1: Using nvm (recommended)**
```bash
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
source ~/.bashrc
nvm install 18
nvm use 18
```

**Option 2: Direct install**
- Visit https://nodejs.org/ and download Node.js 18+ LTS

## Quick Setup

### 1. Start Docker Services

```bash
docker-compose up -d
```

Wait for services to be healthy (check with `docker-compose ps`).

### 2. Set Up Backend

```bash
cd backend

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create .env file from example
cp .env.example .env

# Generate a secure SECRET_KEY
python3 -c "import secrets; print(secrets.token_urlsafe(32))"
# Copy the output and paste it as SECRET_KEY in .env

# Edit .env and add your OPENAI_API_KEY (if using OpenAI)
# Or configure for local models (Ollama) if preferred
```

### 3. Set Up Frontend

```bash
cd frontend

# Install dependencies
npm install

# Create .env file from example
cp .env.example .env
```

### 4. Initialize Databases

**Neo4j:**
1. Open http://localhost:7474 in your browser
2. Login with `neo4j` / `research_graph_password`
3. Run the initialization script (see below)

**PostgreSQL:**
The schema will be created automatically when you run migrations (Phase 1).

## Next Steps

Once setup is complete, you can:

1. **Start the backend:**
   ```bash
   cd backend
   source venv/bin/activate
   uvicorn app.main:app --reload
   ```

2. **Start the frontend:**
   ```bash
   cd frontend
   npm run dev
   ```

3. **Verify:**
   - Backend: http://localhost:8000/docs
   - Frontend: http://localhost:5173
   - Neo4j Browser: http://localhost:7474

## Neo4j Initialization

After Neo4j is running, connect to the browser and run:

```cypher
// Unique constraints on IDs
CREATE CONSTRAINT observation_id IF NOT EXISTS
FOR (o:Observation) REQUIRE o.id IS UNIQUE;

CREATE CONSTRAINT hypothesis_id IF NOT EXISTS
FOR (h:Hypothesis) REQUIRE h.id IS UNIQUE;

CREATE CONSTRAINT source_id IF NOT EXISTS
FOR (s:Source) REQUIRE s.id IS UNIQUE;

CREATE CONSTRAINT concept_id IF NOT EXISTS
FOR (c:Concept) REQUIRE c.id IS UNIQUE;

CREATE CONSTRAINT entity_id IF NOT EXISTS
FOR (e:Entity) REQUIRE e.id IS UNIQUE;

// Index for text search
CREATE TEXT INDEX observation_text IF NOT EXISTS
FOR (o:Observation) ON o.text;

CREATE TEXT INDEX hypothesis_claim IF NOT EXISTS
FOR (h:Hypothesis) ON h.claim;

// Index for temporal queries
CREATE INDEX observation_created IF NOT EXISTS
FOR (o:Observation) ON o.created_at;
```

### UUID ID Policy and Backfill

All node and relationship identifiers use UUIDv4 stored in property `id`. If you are upgrading an existing database that previously relied on internal relationship IDs, run the following once:

```cypher
// Assign UUIDs to existing relationships
MATCH ()-[r]->()
WHERE r.id IS NULL
SET r.id = randomUUID();
```

Optional relationship ID indexes (Neo4j 5+):

```cypher
CREATE INDEX rel_id_supports IF NOT EXISTS FOR ()-[r:SUPPORTS]-() ON (r.id);
CREATE INDEX rel_id_contradicts IF NOT EXISTS FOR ()-[r:CONTRADICTS]-() ON (r.id);
CREATE INDEX rel_id_relates IF NOT EXISTS FOR ()-[r:RELATES_TO]-() ON (r.id);
```

References:
- Cypher functions (randomUUID): https://neo4j.com/docs/cypher-manual/4.3/functions/
- Relationship index hint format (Neo4j 5): https://neo4j.com/docs/status-codes/current/notifications/all-notifications/

## Troubleshooting

- **Port conflicts:** Change ports in docker-compose.yml if needed
- **Python not found:** Use `python3` instead of `python`
- **Docker issues:** Ensure Docker Desktop is running (if on Windows/Mac)
