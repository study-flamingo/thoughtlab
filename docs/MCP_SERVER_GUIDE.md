# ThoughtLab MCP Server Guide

This guide explains how to use the ThoughtLab MCP (Model Context Protocol) server with Claude Desktop and other MCP clients.

---

## Overview

The ThoughtLab MCP server exposes knowledge graph operations as MCP tools, allowing Claude and other AI assistants to interact with your research graph through a standardized protocol.

### Architecture

```
┌─────────────────────────────────────────────┐
│        Claude Desktop (MCP Client)          │
│                                             │
│  Uses MCP tools to interact with graph      │
└───────────────────┬─────────────────────────┘
                    │
                    ▼ Streamable HTTP (MCP Protocol)
┌─────────────────────────────────────────────┐
│  ThoughtLab MCP Server (FastMCP at /mcp)    │
│                                             │
│  10+ Tools (registered from tool_definitions):
│  • find_related_nodes                       │
│  • summarize_node                           │
│  • summarize_node_with_context              │
│  • recalculate_node_confidence              │
│  • reclassify_node                          │
│  • search_web_evidence                      │
│  • merge_nodes (dangerous, gated)           │
│  • summarize_relationship                   │
│  • recalculate_edge_confidence              │
│  • reclassify_relationship                  │
│  • check_api_health                         │
└───────────────────┬─────────────────────────┘
                    │
                    ▼ In-Process Calls (no HTTP)
┌─────────────────────────────────────────────┐
│            ToolService                      │
│                                             │
│  Shared service with LangGraph agents       │
│  Calls Neo4j, OpenAI, and other services   │
└─────────────────────────────────────────────┘
```

**Key Architecture Change**: The MCP server now calls ToolService directly (in-process),
sharing the same code path as LangGraph agents. This eliminates HTTP overhead and ensures
consistent behavior across all interfaces.

---

## Installation

### Prerequisites

1. **Python 3.13+** with uv package manager
2. **Backend running** (all services: Neo4j, Redis, FastAPI)
3. **OpenAI API key** set in environment

### Install Dependencies

```bash
cd backend
uv sync --all-extras
```

This installs:
- `fastmcp>=0.2.0` - FastMCP framework
- All backend dependencies
- All LangChain/LangGraph dependencies

---

## Configuration

### Environment Variables

The MCP server uses these environment variables:

```bash
# Required: OpenAI API key for AI operations
export THOUGHTLAB_OPENAI_API_KEY="sk-..."

# Optional: Enable dangerous tools (merge, delete, etc.)
export THOUGHTLAB_MCP_ADMIN_MODE="false"  # Set to "true" to enable

# Optional: LLM configuration
export THOUGHTLAB_LLM_MODEL="gpt-4o-mini"
export THOUGHTLAB_EMBEDDING_MODEL="text-embedding-3-small"
```

**Note on Admin Mode**: By default, dangerous tools like `merge_nodes` are disabled
in MCP for safety. Set `THOUGHTLAB_MCP_ADMIN_MODE=true` to enable them.

### Verify Configuration

```bash
cd backend
python validate_mcp.py
```

Expected output:
```
[+] All validations passed!
```

---

## Running the Server

### Standalone Mode (Testing)

```bash
cd backend
python mcp_server.py
```

The server runs on stdio transport and waits for MCP client connections.

### With Claude Desktop

See [Claude Desktop Configuration](#claude-desktop-configuration) below.

---

## Available Tools

The MCP server exposes 10+ tools (registered from shared tool definitions):

### 1. find_related_nodes

Find semantically similar nodes using vector embeddings.

**Parameters:**
- `node_id` (string, required): Node to find similar nodes for
- `limit` (integer, optional): Max results (default: 10)
- `min_similarity` (float, optional): Min score 0-1 (default: 0.5)
- `node_types` (array, optional): Filter by types
- `auto_link` (boolean, optional): Auto-create relationships (default: false)

**Example:**
```
Find nodes related to observation obs-123
```

### 2. summarize_node

Generate AI summary of node content.

**Parameters:**
- `node_id` (string, required): Node to summarize
- `max_length` (integer, optional): Max characters (default: 200)
- `style` (string, optional): "concise", "detailed", or "bullet_points"

**Example:**
```
Summarize hypothesis hyp-456 in bullet points
```

### 3. summarize_node_with_context

Context-aware summary including relationships.

**Parameters:**
- `node_id` (string, required): Node to summarize
- `depth` (integer, optional): Relationship hops (default: 1)
- `relationship_types` (array, optional): Filter types
- `max_length` (integer, optional): Max characters (default: 300)

**Example:**
```
Give me a comprehensive view of observation obs-789 with all context
```

### 4. recalculate_node_confidence

Re-analyze confidence based on graph context.

**Parameters:**
- `node_id` (string, required): Node to recalculate
- `factor_in_relationships` (boolean, optional): Consider connections (default: true)

**Example:**
```
How confident should we be in hypothesis hyp-123?
```

### 5. summarize_relationship

Explain connection between nodes in plain language.

**Parameters:**
- `edge_id` (string, required): Relationship to summarize
- `include_evidence` (boolean, optional): Show evidence (default: true)

**Example:**
```
Why are observation obs-123 and hypothesis hyp-456 connected?
```

### 6. check_api_health

Check backend API health and configuration.

**Parameters:** None

**Example:**
```
Check if the ThoughtLab backend is working
```

---

## Claude Desktop Configuration

### Method 1: Using fastmcp install (Recommended)

```bash
# Install for Claude Desktop
cd backend
fastmcp install claude-desktop mcp_server.py \
  --server-name "ThoughtLab" \
  --env THOUGHTLAB_OPENAI_API_KEY="sk-..." \
  --env THOUGHTLAB_API_BASE_URL="http://localhost:8000/api/v1"
```

This automatically:
- Adds server to Claude Desktop config
- Configures environment variables
- Sets up stdio transport

### Method 2: Manual Configuration

Edit Claude Desktop's config file:

**macOS/Linux:**
```bash
code ~/Library/Application\ Support/Claude/claude_desktop_config.json
```

**Windows:**
```bash
code %APPDATA%\Claude\claude_desktop_config.json
```

Add this configuration:

```json
{
  "mcpServers": {
    "thoughtlab": {
      "command": "python",
      "args": [
        "C:\\Users\\YourName\\Code\\Repos\\thoughtlab\\backend\\mcp_server.py"
      ],
      "env": {
        "THOUGHTLAB_OPENAI_API_KEY": "sk-...",
        "THOUGHTLAB_API_BASE_URL": "http://localhost:8000/api/v1"
      }
    }
  }
}
```

**Note:** Update the path to match your installation location.

### Restart Claude Desktop

After configuration, restart Claude Desktop to load the MCP server.

---

## Usage Examples

Once configured, you can interact with your knowledge graph through Claude:

### Example 1: Explore Related Content

**You:** "Find nodes related to observation obs-123 and tell me what you found"

**Claude:** Uses `find_related_nodes` tool, then explains the connections it discovered.

### Example 2: Comprehensive Analysis

**You:** "Give me a complete analysis of hypothesis hyp-456"

**Claude:** Uses `summarize_node_with_context` to show the hypothesis with all supporting/contradicting evidence.

### Example 3: Confidence Assessment

**You:** "Should I trust observation obs-789? Recalculate its confidence"

**Claude:** Uses `recalculate_node_confidence` and explains the factors affecting reliability.

### Example 4: Relationship Explanation

**You:** "Why are these two nodes connected? Explain relationship rel-abc"

**Claude:** Uses `summarize_relationship` to explain the connection in plain language.

### Example 5: Health Check

**You:** "Is the ThoughtLab backend working properly?"

**Claude:** Uses `check_api_health` to verify the backend status.

---

## Troubleshooting

### Server Not Appearing in Claude Desktop

**Check:**
1. Configuration file path is correct
2. JSON is valid (no trailing commas)
3. Python path in config is correct
4. Claude Desktop was restarted after config change

**Debug:**
```bash
# Test server directly
cd backend
python mcp_server.py

# Should wait for input (stdio mode)
```

### Tool Calls Failing

**Check:**
1. Backend server is running:
   ```bash
   curl http://localhost:8000/api/v1/tools/health
   ```

2. Environment variables are set in config

3. Check Claude Desktop logs (Help → View Logs)

### API Errors

**Common Issues:**

1. **Backend not running:**
   ```bash
   # Start backend
   ./start.sh
   ```

2. **Wrong API URL:**
   ```bash
   # Verify in config
   "THOUGHTLAB_API_BASE_URL": "http://localhost:8000/api/v1"
   ```

3. **OpenAI API key missing:**
   ```bash
   # Add to config env section
   "THOUGHTLAB_OPENAI_API_KEY": "sk-..."
   ```

---

## Testing Without Claude Desktop

You can test the MCP server using the FastMCP client:

```python
import asyncio
from fastmcp import Client

async def test_mcp_server():
    # Connect to server via HTTP (requires server running separately)
    client = Client("http://localhost:8000/mcp")

    async with client:
        # List available tools
        tools = await client.list_tools()
        print(f"Available tools: {[t.name for t in tools]}")

        # Call a tool
        result = await client.call_tool(
            "check_api_health",
            {}
        )
        print(result)

asyncio.run(test_mcp_server())
```

---

## Architecture Benefits

### 1. Unified Tool Layer

All interfaces share the same ToolService:
- MCP server (Claude Desktop)
- LangGraph agents (programmatic)
- REST API (frontend UI)

### 2. In-Process Efficiency

- MCP calls ToolService directly (no HTTP overhead)
- Same code path as LangGraph agents
- Shared tool definitions from `app/tools/tool_definitions.py`
- Consistent behavior across all interfaces

### 3. Standardized Protocol

- Works with any MCP client
- Not tied to Claude Desktop
- Future-proof architecture
- Industry standard protocol

### 4. Safety Controls

- Dangerous tools gated by `THOUGHTLAB_MCP_ADMIN_MODE`
- Job queue for async operations
- Report storage for LangGraph results

---

## Advanced Configuration

### Custom Server Name

```python
from app.mcp import create_mcp_server

mcp = create_mcp_server("My Custom Name")
```

### Custom API URL

```bash
export THOUGHTLAB_API_BASE_URL="https://production.example.com/api/v1"
```

### Logging

```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # More verbose
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
```

---

## Performance

### Typical Response Times

| Operation | Time | Notes |
|-----------|------|-------|
| Health check | ~100ms | Simple API call |
| Find related | ~500ms | Vector search |
| Summarize | ~2-3s | LLM generation |
| With context | ~3-5s | LLM + graph traversal |
| Recalculate | ~3-4s | LLM analysis |

### Optimization Tips

1. **Reduce limits:** Use lower `limit` values for faster searches
2. **Cache results:** Backend caches LLM responses
3. **Batch operations:** Process multiple nodes together
4. **Tune timeout:** Adjust `THOUGHTLAB_API_TIMEOUT` if needed

---

## Security Considerations

### API Key Protection

- Never commit API keys to version control
- Use environment variables in config
- Rotate keys regularly
- Use separate keys for dev/prod

### Network Security

- MCP server runs locally by default
- Backend API should use HTTPS in production
- Consider firewall rules for API access
- Use VPN for remote access

---

## Future Enhancements

### Planned Features

- [ ] Streaming responses for long operations
- [ ] Batch tool operations
- [ ] Graph visualization generation
- [ ] Export/report tools
- [ ] Authentication/authorization
- [ ] Rate limiting
- [ ] Metrics and monitoring
- [ ] Caching layer

### Recently Implemented

- Streamable HTTP transport (mounted at `/mcp`)
- Web search integration (via `search_web_evidence` tool)
- Node merging with confirmation (via `merge_nodes` tool, gated by admin mode)
- 10+ tools registered from shared tool definitions
- In-process ToolService calls (no HTTP overhead)

---

## See Also

- [Phase 7: Backend API Specification](./PHASE_7_API_SPEC.md)
- [Phase 8: LangGraph Integration](./PHASE_8_LANGGRAPH_INTEGRATION.md)
- [Implementation Summary](./IMPLEMENTATION_SUMMARY.md)
- [FastMCP Documentation](https://gofastmcp.com/)
- [MCP Protocol Specification](https://modelcontextprotocol.io/)

---

## Support

### Documentation
- `docs/` directory for all guides
- `backend/examples/` for code examples
- API documentation at `/docs` endpoint

### Issues
- GitHub Issues for bug reports
- Include MCP server logs
- Provide configuration (redact API keys)

### Community
- Model Context Protocol community
- FastMCP discussions
- ThoughtLab project discussions
