# ThoughtLab MCP Server

This directory contains the Model Context Protocol (MCP) server implementation for ThoughtLab.

## Overview

The MCP server exposes ThoughtLab's knowledge graph operations as MCP tools, enabling Claude Desktop and other MCP clients to interact with your research graph through a standardized protocol.

## Quick Start

1. **Install dependencies:**
   ```bash
   cd backend
   uv sync --all-extras
   ```

2. **Set environment variables:**
   ```bash
   export THOUGHTLAB_OPENAI_API_KEY="sk-..."
   export THOUGHTLAB_API_BASE_URL="http://localhost:8000/api/v1"
   ```

3. **Start backend server:**
   ```bash
   # From project root
   ./start.sh
   ```

4. **Validate MCP server:**
   ```bash
   python validate_mcp.py
   ```

5. **Run MCP server:**
   ```bash
   python mcp_server.py
   ```

## Architecture

```
MCP Server (FastMCP) → HTTP API → Backend Services
```

- **Pure HTTP client** - No direct database access
- **6 tools** exposed via MCP protocol
- **Same backend** as LangGraph agents
- **Completely modular** and independently deployable

## Available Tools

1. **find_related_nodes** - Find semantically similar nodes
2. **summarize_node** - Generate AI summaries
3. **summarize_node_with_context** - Context-aware summaries
4. **recalculate_node_confidence** - Assess node reliability
5. **summarize_relationship** - Explain connections
6. **check_api_health** - Backend health check

## Files

- `__init__.py` - Module exports
- `server.py` - Main MCP server implementation (400+ lines)
- `README.md` - This file

## Configuration

Configure via environment variables:

```bash
# Required
THOUGHTLAB_OPENAI_API_KEY="sk-..."

# Optional
THOUGHTLAB_API_BASE_URL="http://localhost:8000/api/v1"
THOUGHTLAB_API_TIMEOUT="30.0"
```

## Claude Desktop Integration

See the complete guide: [docs/MCP_SERVER_GUIDE.md](../../../docs/MCP_SERVER_GUIDE.md)

### Quick Setup

```bash
# Install for Claude Desktop
fastmcp install claude-desktop mcp_server.py \
  --server-name "ThoughtLab" \
  --env THOUGHTLAB_OPENAI_API_KEY="sk-..." \
  --env THOUGHTLAB_API_BASE_URL="http://localhost:8000/api/v1"
```

## Usage Example

Once configured in Claude Desktop:

**You:** "Find nodes related to observation obs-123"

**Claude:** (Uses `find_related_nodes` tool and explains results)

**You:** "Give me a comprehensive analysis of hypothesis hyp-456"

**Claude:** (Uses `summarize_node_with_context` to show full picture)

## Development

### Testing

```bash
# Validate implementation
python validate_mcp.py

# Run server
python mcp_server.py
```

### Adding New Tools

1. Add tool function in `server.py`
2. Decorate with `@mcp.tool()`
3. Make HTTP API call to backend
4. Return formatted string
5. Update validation script

Example:
```python
@mcp.tool()
async def my_new_tool(param: str) -> str:
    """Tool description."""
    result = await _api_request("POST", "/endpoint", {"param": param})
    return f"Result: {result}"
```

## See Also

- [MCP Server Guide](../../../docs/MCP_SERVER_GUIDE.md) - Complete documentation
- [Backend API Spec](../../../docs/PHASE_7_API_SPEC.md) - API reference
- [FastMCP Docs](https://gofastmcp.com/) - FastMCP framework
- [MCP Protocol](https://modelcontextprotocol.io/) - Protocol specification
