---
name: mcp-tool
description: Add a new tool to an MCP server. Use when you need to add functionality to an existing server.
---

# Add MCP Tool

When invoked, add a properly structured tool to the MCP server.

## Tool Design Principles

1. **Clear name**: Action-oriented verb (search, create, get)
2. **Good description**: Explain what it does and when to use it
3. **Typed inputs**: All parameters with types and descriptions
4. **Structured output**: Return useful, parseable content

## Python (FastMCP)

### Basic Tool
```python
@mcp.tool
def search_documents(query: str, limit: int = 10) -> list[dict]:
    """Search for documents matching the query.

    Args:
        query: Search terms to find in documents
        limit: Maximum number of results to return

    Returns:
        List of matching documents with id and title
    """
    results = db.search(query, limit=limit)
    return [{"id": r.id, "title": r.title} for r in results]
```

### Tool with Pydantic Model
```python
from pydantic import BaseModel, Field

class SearchParams(BaseModel):
    query: str = Field(description="Search terms")
    filters: dict[str, str] | None = Field(default=None, description="Optional filters")
    limit: int = Field(default=10, ge=1, le=100, description="Max results")

@mcp.tool
def advanced_search(params: SearchParams) -> list[dict]:
    """Search with advanced filtering options."""
    return perform_search(params)
```

### Tool with Context
```python
from fastmcp import Context

@mcp.tool
def get_user_data(user_id: str, ctx: Context) -> dict:
    """Get data for a specific user.

    Requires authentication.
    """
    # Access auth info from context
    current_user = ctx.request_context.get("user")

    # Access server state
    db = ctx.server.state["db"]

    return db.get_user(user_id)
```

## TypeScript (@modelcontextprotocol/sdk)

### Basic Tool
```typescript
import * as z from 'zod';

server.registerTool(
  'search_documents',
  {
    title: 'Search Documents',
    description: 'Search for documents matching the query',
    inputSchema: {
      query: z.string().describe('Search terms'),
      limit: z.number().default(10).describe('Max results')
    }
  },
  async ({ query, limit }) => {
    const results = await db.search(query, limit);
    return {
      content: [{
        type: 'text',
        text: JSON.stringify(results)
      }]
    };
  }
);
```

### Tool with Structured Output
```typescript
server.registerTool(
  'calculate',
  {
    title: 'Calculate',
    description: 'Perform a calculation',
    inputSchema: {
      a: z.number().describe('First number'),
      b: z.number().describe('Second number'),
      operation: z.enum(['add', 'subtract', 'multiply', 'divide'])
    },
    outputSchema: {
      result: z.number(),
      operation: z.string()
    }
  },
  async ({ a, b, operation }) => {
    const ops = {
      add: a + b,
      subtract: a - b,
      multiply: a * b,
      divide: a / b
    };
    const result = ops[operation];

    return {
      content: [{ type: 'text', text: `${a} ${operation} ${b} = ${result}` }],
      structuredContent: { result, operation }
    };
  }
);
```

### Tool with Error Handling
```typescript
server.registerTool(
  'fetch_url',
  {
    title: 'Fetch URL',
    description: 'Fetch content from a URL',
    inputSchema: {
      url: z.string().url().describe('URL to fetch')
    }
  },
  async ({ url }) => {
    try {
      const response = await fetch(url);
      if (!response.ok) {
        return {
          content: [{ type: 'text', text: `HTTP error: ${response.status}` }],
          isError: true
        };
      }
      const text = await response.text();
      return {
        content: [{ type: 'text', text }]
      };
    } catch (error) {
      return {
        content: [{ type: 'text', text: `Fetch failed: ${error.message}` }],
        isError: true
      };
    }
  }
);
```

## Tool Checklist

- [ ] Name is action-oriented and clear
- [ ] Description explains purpose and usage
- [ ] All parameters have types and descriptions
- [ ] Default values where sensible
- [ ] Error cases handled gracefully
- [ ] Output is useful and parseable
