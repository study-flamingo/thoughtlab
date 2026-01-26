# ThoughtLab Development Scripts

This document describes the enhanced development scripts that support multiple development modes: fully containerized, mixed (local frontend + containerized backend), and fully local.

## Script Structure

### Root Scripts (Wrappers)
- `setup.sh` → `scripts/setup.sh`
- `start.sh` → `scripts/start.sh`
- `stop.sh` → `scripts/stop.sh`
- `restart.sh` → `scripts/restart.sh`

### Enhanced Scripts in `scripts/` Directory

#### `scripts/setup.sh`
**Purpose**: Complete project setup with interactive mode selection

**Features**:
- Interactive mode selection (Docker, Mixed, Local)
- Automatic dependency installation (uv, npm packages)
- Environment file creation with secure secret key generation
- Docker image building (when needed)
- Comprehensive dependency checking

**Usage**:
```bash
./scripts/setup.sh
```

**Modes**:
1. **Containerized** (Docker) - All services in containers (Easiest)
2. **Mixed** - Frontend local, backend in containers (Recommended)
3. **Fully Local** - Everything runs locally (Most control)

---

#### `scripts/start.sh`
**Purpose**: Start development servers with automatic mode detection

**Features**:
- Automatic mode detection based on project state
- Multi-process management with PID files
- Port availability checking and cleanup
- Health checks for services
- Real-time log display
- Graceful cleanup on Ctrl+C

**Usage**:
```bash
./scripts/start.sh
```

**Supported Modes**:
- **Mixed Mode**: Local frontend + Containerized backend
- **Docker Mode**: All services in Docker
- **Local Mode**: Fully local development

**Key Features**:
- Creates `.run/logs/` directory for logs
- Saves PIDs in `.run/` directory
- Tail logs automatically
- Handles port conflicts gracefully

---

#### `scripts/start-docker.sh`
**Purpose**: Start all services in Docker containers

**Features**:
- Full containerization (frontend, backend, Neo4j, Redis)
- Build and start in one command
- Service health checks
- Interactive log viewing option
- Creates `.containerized-frontend` marker

**Usage**:
```bash
./scripts/start-docker.sh
```

**What it starts**:
- Frontend: http://localhost:5173
- Backend: http://localhost:8000
- Neo4j: http://localhost:7474
- Redis: localhost:6379

---

#### `scripts/stop.sh`
**Purpose**: Stop all running services (Docker + local processes)

**Features**:
- Stop by service type (`all`, `frontend-only`, `backend-only`)
- Multi-platform support (Windows, Linux, macOS)
- PID file cleanup
- Port-based process killing
- Force kill for stubborn processes

**Usage**:
```bash
./scripts/stop.sh                    # Stop everything
./scripts/stop.sh frontend-only      # Stop frontend only
./scripts/stop.sh backend-only       # Stop backend only
```

---

#### `scripts/restart.sh`
**Purpose**: Restart all services with proper cleanup

**Features**:
- Stops all services first
- Clears Python and Vite caches
- Verifies ports are released
- Restarts in detected or specified mode
- Cache clearing for reliable restarts

**Usage**:
```bash
./scripts/restart.sh                 # Restart in detected mode
./scripts/restart.sh docker          # Restart in Docker mode
./scripts/restart.sh local           # Restart in local mode
```

---

## Development Modes

### 1. Containerized Development (Easiest)
**Best for**: Beginners, consistent environment, quick setup

**How to use**:
```bash
./scripts/setup.sh           # Select option 1
./scripts/start-docker.sh    # Start all services
```

**Pros**:
- Zero local dependencies (except Docker)
- Consistent environment across all developers
- All services isolated in containers
- Easy cleanup with `docker-compose down`

**Cons**:
- Slightly slower frontend hot reload
- More Docker resource usage

### 2. Mixed Development (Recommended)
**Best for**: Frontend developers who want fast hot reload

**How to use**:
```bash
./scripts/setup.sh           # Select option 2
./scripts/start.sh           # Start mixed mode
```

**What happens**:
- Backend: Runs in Docker container
- Frontend: Runs locally with Vite hot reload
- Neo4j/Redis: Run in Docker containers

**Pros**:
- Fast frontend development (instant hot reload)
- Backend isolated and consistent
- Lower resource usage than full Docker

**Cons**:
- Requires Node.js locally
- Frontend port conflicts possible

### 3. Fully Local Development
**Best for**: Experienced developers who want full control

**How to use**:
```bash
./scripts/setup.sh           # Select option 3
./scripts/start.sh           # Start local mode
```

**What happens**:
- Everything runs locally (Python backend, Node.js frontend)
- Neo4j runs in Docker (required)
- Redis runs locally

**Pros**:
- Full control over all processes
- Best performance
- Easier debugging of individual components

**Cons**:
- Requires Python, Node.js, Docker
- More complex setup
- Potential version conflicts

---

## Common Workflows

### Daily Development
```bash
# First time setup
./scripts/setup.sh

# Start development
./scripts/start.sh

# Make changes... (hot reload works automatically)

# Stop when done (Ctrl+C in start.sh terminal, or:)
./scripts/stop.sh
```

### Switching Modes
```bash
# Switch from local to Docker
rm -f .containerized-frontend
./scripts/stop.sh
./scripts/start-docker.sh

# Switch from Docker to local
./scripts/stop.sh
rm -f .containerized-frontend
./scripts/start.sh
```

### Debugging Issues
```bash
# Check what's running
docker-compose ps

# View all logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f frontend
docker-compose logs -f backend

# Stop everything and start fresh
./scripts/restart.sh
```

### Development Iteration
```bash
# Clear everything and rebuild
./scripts/stop.sh
docker-compose down -v          # Remove volumes too
./scripts/setup.sh
./scripts/start.sh
```

---

## File Locations

### Runtime Files
- `.run/` - PID files and runtime data
- `.run/logs/` - Application logs
  - `backend.log` - Backend server logs
  - `frontend.log` - Frontend server logs

### Configuration Files
- `.containerized-frontend` - Marker file (exists when using Docker frontend)
- `docker-compose.yml` - Docker services definition
- `frontend/Dockerfile.dev` - Frontend development container

### Log Directories
- Local development: `.run/logs/`
- Docker development: Use `docker-compose logs -f`

---

## Troubleshooting

### Port Conflicts
```bash
# Check what's using ports
lsof -i :5173    # Frontend port
lsof -i :8000    # Backend port

# Force stop everything
./scripts/stop.sh
```

### Docker Issues
```bash
# Check Docker status
docker info

# Restart Docker if needed
docker-compose down
docker-compose up -d
```

### Permission Issues (Linux/macOS)
```bash
# Fix script permissions
chmod +x scripts/*.sh

# Fix log directory permissions
sudo chown -R $USER:$USER .run/
```

### Node Module Issues
```bash
# Reinstall node modules
rm -rf frontend/node_modules
cd frontend && npm install
```

### Python Virtual Environment Issues
```bash
# Recreate venv
cd backend
rm -rf .venv
uv venv
uv pip install -r requirements.txt
```

---

## Environment Variables

### Frontend (.env)
```bash
# For local development
VITE_API_URL=http://localhost:8000/api/v1

# For Docker development
VITE_API_URL=http://backend:8000/api/v1  # Set automatically by docker-compose
```

### Backend (.env)
```bash
# Database connections (Docker)
NEO4J_URI=bolt://neo4j:7687
REDIS_URL=redis://redis:6379

# Security
SECRET_KEY=your_generated_secret_key
```

---

## Advanced Usage

### Custom Port Configuration
Edit `docker-compose.yml` for Docker services or use environment variables for local development.

### Scaling Services
```bash
# Scale frontend (Docker mode only)
docker-compose up --scale frontend=2 -d
```

### Running Specific Services
```bash
# Start only backend and Neo4j
docker-compose up -d backend neo4j

# Start only frontend locally
cd frontend && npm run dev
```

### Production Build Testing
```bash
# Build and run production frontend
docker-compose build frontend
docker-compose run --rm frontend
```

---

## Migration from Old Scripts

The old scripts in the root directory now act as wrappers that call the enhanced versions in `scripts/`. All functionality is preserved, but you now have access to:

- **Better error handling** with clear messages
- **Multiple development modes** with automatic detection
- **Improved logging** with centralized log directory
- **Port conflict resolution** with automatic cleanup
- **Health checks** for services
- **Cross-platform support** (Windows, Linux, macOS)

### Backward Compatibility
All existing commands still work:
```bash
./setup.sh      # Now uses scripts/setup.sh
./start.sh      # Now uses scripts/start.sh
./stop.sh       # Now uses scripts/stop.sh
./restart.sh    # Now uses scripts/restart.sh
```

### New Commands
```bash
./scripts/start-docker.sh    # Containerized development
```

---

## Best Practices

1. **Use Mixed Mode for Development**: Best balance of performance and isolation
2. **Use Docker Mode for Testing**: Ensures consistent environment
3. **Always Check Ports**: Scripts handle this automatically, but good to know
4. **Read Logs**: Check `.run/logs/` or `docker-compose logs` for issues
5. **Clean Restart**: Use `./scripts/restart.sh` when in doubt
6. **Version Control**: Don't commit `.env` files or `.run/` directory

---

## Performance Tips

### Frontend Hot Reload
- **Mixed/Local**: Instant reload (Vite)
- **Docker**: Slightly slower due to volume mounting

### Backend Reloading
- **Local**: Automatic reload on file changes (uvicorn --reload)
- **Docker**: Requires container restart (`docker-compose restart backend`)

### Memory Usage
- **Docker**: ~2-3GB (all services)
- **Mixed**: ~1.5GB (frontend local, backend containerized)
- **Local**: ~1GB (excluding Neo4j)

---

## Getting Help

- **Documentation**: See `DEVELOPMENT_GUIDE.md` for detailed development workflows
- **Containerized**: See `frontend/DOCKER_DEV.md` for Docker-specific info
- **API Docs**: http://localhost:8000/docs (when backend is running)
- **Neo4j Browser**: http://localhost:7474 (browser interface)