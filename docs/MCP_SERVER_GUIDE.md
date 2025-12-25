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
│  Uses MCP tools to interact with graph     │
└───────────────────┬─────────────────────────┘
                    │
                    ▼ stdio/HTTP (MCP Protocol)
┌─────────────────────────────────────────────┐
│      ThoughtLab MCP Server (FastMCP)        │
│                                             │
│  6 Tools:                                   │
│  • find_related_nodes                       │
│  • summarize_node                           │
│  • summarize_node_with_context              │
│  • recalculate_node_confidence              │
│  • summarize_relationship                   │
│  • check_api_health                         │
└───────────────────┬─────────────────────────┘
                    │
                    ▼ HTTP API Calls
┌─────────────────────────────────────────────┐
│         ThoughtLab Backend API              │
│                                             │
│  REST endpoints for all operations          │
└─────────────────────────────────────────────┘
```

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
# Required: OpenAI API key for backend operations
export THOUGHTLAB_OPENAI_API_KEY="sk-..."

# Optional: Custom backend API URL
export THOUGHTLAB_API_BASE_URL="http://localhost:8000/api/v1"

# Optional: API request timeout in seconds
export THOUGHTLAB_API_TIMEOUT="30.0"
```

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

The MCP server exposes 6 tools:

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

### 1. Same Backend for All Interfaces

The MCP server uses the exact same backend API as:
- LangGraph agents
- Future Chrome extension
- Future CLI tools
- Direct API access

### 2. Complete Separation

- MCP server is a pure HTTP client
- No direct database access
- No shared services
- Can deploy independently

### 3. Standardized Protocol

- Works with any MCP client
- Not tied to Claude Desktop
- Future-proof architecture
- Industry standard protocol

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
- [ ] Web search integration
- [ ] Node merging with confirmation
- [ ] Graph visualization generation
- [ ] Export/report tools

### Coming Soon

- HTTP transport option (not just stdio)
- Authentication/authorization
- Rate limiting
- Metrics and monitoring
- Caching layer

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
