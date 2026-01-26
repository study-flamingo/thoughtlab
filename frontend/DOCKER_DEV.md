# Frontend Containerized Development

This document describes how to run the frontend in a Docker container as part of the ThoughtLab development environment.

## Overview

The frontend can now be run in a Docker container alongside the backend, Neo4j, and Redis services. This provides:

- **Isolated development environment**
- **Automatic network configuration** between frontend and backend
- **Hot reloading** for frontend changes
- **Consistent dependencies** across all developers

## Quick Start

### Option 1: Using the Development Script (Recommended)

```bash
# From the project root directory
./start-dev.sh
```

This will:
1. Stop any existing containers
2. Build and start all services (backend, frontend, Neo4j, Redis)
3. Show service status and URLs

### Option 2: Manual Docker Compose

```bash
# From the project root directory
docker-compose up --build -d
```

### Option 3: Run Frontend Locally (Alternative)

If you prefer to run the frontend locally while keeping backend services in Docker:

```bash
# Start only backend services
docker-compose up -d neo4j redis backend

# Run frontend locally
cd frontend
npm run dev
```

## Service URLs

Once running, you can access:

- **Frontend**: http://localhost:5173
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Neo4j Browser**: http://localhost:7474
- **Redis CLI**: `redis-cli -p 6379`

## Environment Configuration

### API Proxy Configuration

The frontend uses a **Vite proxy** to route API requests to the backend, avoiding CORS issues entirely:

- **Containerized mode**: `VITE_PROXY_TARGET=http://backend:8000` (Docker service name)
- **Local development**: Defaults to `http://localhost:8000`

The React app makes requests to `/api/v1/...` (relative paths), and Vite proxies them to the backend. This setup:
- Avoids CORS issues completely
- Works for both containerized and mixed development modes
- Uses clean relative URLs in the codebase

The backend also includes CORS configuration for direct access if needed:
- `http://localhost:5173`, `http://localhost:3000`, `http://127.0.0.1:5173`, `http://frontend:5173`, `http://frontend:80`

## Development Workflow

### 1. Make Changes
- Edit files in the `frontend/src` directory
- Changes trigger automatic hot reload in the container
- The container maps your local `frontend` directory to `/app` inside the container

### 2. View Logs
```bash
# View all logs
docker-compose logs -f

# View specific service
docker-compose logs -f frontend
docker-compose logs -f backend
```

### 3. Stop Services
```bash
# Stop all services
docker-compose down

# Stop but keep volumes (preserves data)
docker-compose stop

# Stop and remove volumes (clean slate)
docker-compose down -v
```

### 4. Restart Specific Service
```bash
docker-compose restart frontend
```

## Troubleshooting

### Frontend Container Won't Start

1. **Check logs**: `docker-compose logs frontend`
2. **Port conflict**: Ensure port 5173 isn't in use locally
3. **Node modules**: Delete `node_modules` locally if there are permission issues

### API Calls Fail

1. **Check backend**: `docker-compose logs backend`
2. **Check proxy**: API calls use relative paths (`/api/v1/...`) - Vite proxies to backend
3. **Verify proxy target**: Check `VITE_PROXY_TARGET` in docker-compose.yml
4. **Test directly**: `curl http://localhost:5173/api/v1/graph/full` should return data

### Hot Reload Not Working

1. **Check volume mapping**: `docker-compose exec frontend ls /app`
2. **Restart container**: `docker-compose restart frontend`
3. **Check Vite logs**: `docker-compose logs frontend`

### TypeScript Errors

If you see TypeScript errors that weren't there before:

1. **Check extensions**: Ensure `cytoscape-grid-guide` and `cytoscape-navigator` are installed
2. **Type definitions**: Verify `src/types/*.d.ts` files exist
3. **TS config**: Ensure `tsconfig.json` includes `"src/types"`

## Development Tips

### Using Docker Compose Profiles

You can use Docker Compose profiles to selectively start services:

```bash
# Start only database services
docker-compose --profile db up

# Start backend services (no frontend)
docker-compose up neo4j redis backend
```

### Debugging

Access container shell for debugging:

```bash
# Frontend container
docker-compose exec frontend sh

# Backend container
docker-compose exec backend sh

# Install additional tools in containers
docker-compose exec frontend apk add curl
```

### Testing API Integration

Test that frontend can reach backend:

```bash
# From inside frontend container
docker-compose exec frontend sh
curl http://backend:8000/health

# Or from host
curl http://localhost:8000/health
```

## Production Build

For production deployment, use the production Dockerfile:

```bash
# Build production image
docker build -f frontend/Dockerfile -t thoughtlab-frontend:latest ./frontend

# Run production container
docker run -p 80:80 thoughtlab-frontend:latest
```

## Migration Notes

### From Local Development

If you were previously running frontend locally:

1. **Stop local frontend**: Kill any processes using port 5173
2. **API proxy**: The frontend now uses relative URLs (`/api/v1/...`) with Vite proxy
3. **Clear cache**: Remove `.vite` cache directory if needed
4. **Rebuild**: Run `docker-compose up --build -d`

### File Permissions

If you encounter permission issues on Linux/Mac:

```bash
# Fix frontend directory permissions
sudo chown -R $USER:$USER frontend/
chmod -R 755 frontend/
```

## Additional Commands

```bash
# Rebuild frontend only
docker-compose build frontend

# Scale frontend instances (for testing)
docker-compose up --scale frontend=2 -d

# Run tests in container
docker-compose exec frontend npm test

# Check container health
docker-compose ps
```

## Architecture

```
┌─────────────────────────────────────────────────┐
│  Host Machine                                   │
│                                                 │
│  ┌───────────────────────────────────────────┐  │
│  │  Docker Network: thoughtlab-network       │  │
│  │                                           │  │
│  │  ┌────────────┐   ┌──────────────┐       │  │
│  │  │ Frontend   │   │ Backend      │       │  │
│  │  │ Port:5173  │   │ Port:8000    │       │  │
│  │  │            │◄──┤              │       │  │
│  │  └────────────┘   └──────────────┘       │  │
│  │       │                   │               │  │
│  │       │                   │               │  │
│  │  ┌─────▼─────┐     ┌─────▼──────┐        │  │
│  │  │  Neo4j    │     │   Redis    │        │  │
│  │  │ Port:7474 │     │ Port:6379  │        │  │
│  │  └───────────┘     └────────────┘        │  │
│  └───────────────────────────────────────────┘  │
│                                                 │
│  Browser Access: http://localhost:5173          │
└─────────────────────────────────────────────────┘
```

The frontend container communicates with the backend over the Docker network, while being accessible from your host browser at `http://localhost:5173`.