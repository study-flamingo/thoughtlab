# Development Scripts

**All development scripts have moved to the `scripts/` directory.**

## Quick Start

```bash
# 1. Setup (choose your development mode)
./scripts/setup.sh

# 2. Start development servers
./scripts/start.sh

# 3. Stop when done
./scripts/stop.sh
```

## Available Scripts

### Setup & Development
- **`./scripts/setup.sh`** - Complete setup with interactive mode selection
- **`./scripts/start.sh`** - Start development servers (auto-detects best mode)
- **`./scripts/start-docker.sh`** - Start all services in Docker containers
- **`./scripts/stop.sh`** - Stop all running services
- **`./scripts/restart.sh`** - Restart all services with clean cache

## Development Modes

1. **Containerized (Docker)** - All services in containers (Easiest)
2. **Mixed** - Local frontend + Containerized backend (Recommended)
3. **Fully Local** - Everything runs locally (Most control)

## Documentation

For detailed documentation, see:
- **`docs/SCRIPTS.md`** - Complete script documentation and usage guide
- **`docs/DEVELOPMENT_GUIDE.md`** - Comprehensive development guide
- **`frontend/DOCKER_DEV.md`** - Containerized development guide

## File Structure

```
scripts/
├── setup.sh              # Enhanced setup
├── start.sh              # Start development
├── start-docker.sh       # Start containerized
├── stop.sh               # Stop services
└── restart.sh            # Restart services

docs/
├── SCRIPTS.md           # Script documentation
├── DEVELOPMENT_GUIDE.md # Development guide
└── DOCKER_DEV.md        # Docker development guide
```

---

**Note**: All scripts are executable and support Windows (Git Bash), Linux, and macOS.