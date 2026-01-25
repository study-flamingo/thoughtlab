---
name: mcp-test
description: Test an MCP server locally. Use to verify tools, resources, and prompts work correctly.
---

# Test MCP Server

When invoked, test the MCP server to verify it works correctly.

## Testing Methods

1. **Direct invocation** - Run server and call methods
2. **MCP Inspector** - Interactive debugging tool
3. **Claude Desktop** - Test in real environment
4. **Unit tests** - Automated testing

## Python (FastMCP) Testing

### Test Client
```python
import pytest
from fastmcp.testing import TestClient
from my_server.server import mcp

@pytest.fixture
def client():
    return TestClient(mcp)

async def test_hello_tool(client):
    """Test the hello tool returns correct greeting."""
    result = await client.call_tool("hello", {"name": "World"})
    assert "Hello, World" in result.content[0].text

async def test_search_with_filters(client):
    """Test search tool with filter parameters."""
    result = await client.call_tool("search", {
        "query": "test",
        "limit": 5
    })
    data = json.loads(result.content[0].text)
    assert len(data) <= 5

async def test_resource(client):
    """Test resource returns expected data."""
    result = await client.read_resource("config://app")
    config = json.loads(result.contents[0].text)
    assert "version" in config
```

### Run Tests
```bash
# Run all tests
pytest -v

# Run specific test
pytest tests/test_server.py::test_hello_tool -v

# Run with coverage
pytest --cov=my_server tests/
```

## TypeScript Testing

### Test Setup
```typescript
import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { InMemoryTransport } from '@modelcontextprotocol/sdk/inMemory.js';
import { Client } from '@modelcontextprotocol/sdk/client/index.js';

describe('MCP Server', () => {
  let server: McpServer;
  let client: Client;

  beforeEach(async () => {
    server = createServer();  // Your server factory
    client = new Client({ name: 'test-client', version: '1.0.0' });

    const [clientTransport, serverTransport] = InMemoryTransport.createLinkedPair();
    await server.connect(serverTransport);
    await client.connect(clientTransport);
  });

  afterEach(async () => {
    await client.close();
  });

  it('should list tools', async () => {
    const tools = await client.listTools();
    expect(tools.tools.length).toBeGreaterThan(0);
  });

  it('should call hello tool', async () => {
    const result = await client.callTool({
      name: 'hello',
      arguments: { name: 'World' }
    });
    expect(result.content[0].text).toContain('Hello, World');
  });
});
```

## MCP Inspector

The MCP Inspector provides interactive debugging:

```bash
# Install
npx @anthropic-ai/mcp-inspector

# Run with your server
npx @anthropic-ai/mcp-inspector python -m my_server.server
npx @anthropic-ai/mcp-inspector node dist/index.js
```

Features:
- List all tools, resources, prompts
- Call tools interactively
- Read resources
- View request/response logs

## Claude Desktop Testing

### Add to Claude Desktop config

**macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
**Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "my-server": {
      "command": "python",
      "args": ["-m", "my_server.server"]
    }
  }
}
```

For TypeScript:
```json
{
  "mcpServers": {
    "my-server": {
      "command": "node",
      "args": ["/path/to/dist/index.js"]
    }
  }
}
```

### Verify in Claude
1. Restart Claude Desktop
2. Check MCP icon shows your server
3. Ask Claude to use a tool from your server
4. Verify output is correct

## Manual Testing

### stdio transport
```bash
# Start server
python -m my_server.server

# Send JSON-RPC request via stdin
{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}

# Call a tool
{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"hello","arguments":{"name":"World"}}}
```

### HTTP transport
```bash
# Start server
python -m my_server.server --transport http --port 8000

# List tools
curl -X POST http://localhost:8000/mcp \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}'

# Call tool
curl -X POST http://localhost:8000/mcp \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"hello","arguments":{"name":"World"}}}'
```

## Test Checklist

- [ ] All tools return expected output
- [ ] Error cases handled gracefully
- [ ] Resources return valid data
- [ ] Authentication works (if configured)
- [ ] Works in Claude Desktop
- [ ] No hanging processes or memory leaks
