---
name: mcp-resource
description: Add a new resource to an MCP server. Use when you need to expose data that can be read by the client.
---

# Add MCP Resource

When invoked, add a properly structured resource to the MCP server.

## Resource vs Tool

- **Resource**: Data to be READ (like a file or API response)
- **Tool**: Action to be EXECUTED (like a function call)

Use resources for: configuration, documents, API data, database records

## Python (FastMCP)

### Static Resource
```python
@mcp.resource("config://app")
def get_app_config() -> str:
    """Application configuration settings."""
    return json.dumps({
        "version": "1.0.0",
        "environment": "production",
        "features": ["auth", "api"]
    })
```

### Resource with MIME Type
```python
@mcp.resource("data://report.csv", mime_type="text/csv")
def get_report() -> str:
    """Generate CSV report of recent activity."""
    return "date,user,action\n2024-01-15,alice,login\n2024-01-15,bob,purchase"
```

### Dynamic Resource (Template)
```python
@mcp.resource("users://{user_id}/profile")
def get_user_profile(user_id: str) -> str:
    """Fetch user profile by ID.

    Args:
        user_id: The unique identifier of the user
    """
    user = db.get_user(user_id)
    return json.dumps({
        "id": user.id,
        "name": user.name,
        "email": user.email
    })
```

### Resource from Database
```python
@mcp.resource("db://orders/{order_id}")
def get_order(order_id: str) -> str:
    """Retrieve order details."""
    order = db.orders.find_one({"id": order_id})
    if not order:
        return json.dumps({"error": "Order not found"})
    return json.dumps(order)
```

## TypeScript (@modelcontextprotocol/sdk)

### Static Resource
```typescript
server.registerResource(
  'config',
  'app://configuration',
  {
    title: 'App Configuration',
    description: 'Application settings and feature flags',
    mimeType: 'application/json'
  },
  async (uri) => ({
    contents: [{
      uri: uri.href,
      text: JSON.stringify({
        version: '1.0.0',
        features: ['auth', 'api']
      })
    }]
  })
);
```

### Dynamic Resource (Template)
```typescript
import { ResourceTemplate } from '@modelcontextprotocol/sdk/server/mcp.js';

server.registerResource(
  'user-profile',
  new ResourceTemplate('users://{userId}/profile', { list: undefined }),
  {
    title: 'User Profile',
    description: 'User profile data by ID'
  },
  async (uri, { userId }) => ({
    contents: [{
      uri: uri.href,
      mimeType: 'application/json',
      text: JSON.stringify({
        id: userId,
        name: `User ${userId}`,
        role: 'developer'
      })
    }]
  })
);
```

### Resource with Completion
```typescript
server.registerResource(
  'repository',
  new ResourceTemplate('github://repos/{owner}/{repo}', {
    list: undefined,
    complete: {
      owner: (value) =>
        ['microsoft', 'google', 'anthropic']
          .filter(o => o.startsWith(value)),
      repo: (value, context) => {
        const owner = context?.arguments?.['owner'];
        const repos: Record<string, string[]> = {
          'microsoft': ['vscode', 'typescript'],
          'anthropic': ['claude-code', 'anthropic-sdk']
        };
        return (repos[owner] || []).filter(r => r.startsWith(value));
      }
    }
  }),
  { title: 'GitHub Repository' },
  async (uri, { owner, repo }) => ({
    contents: [{
      uri: uri.href,
      text: `Repository: ${owner}/${repo}`
    }]
  })
);
```

## URI Scheme Guidelines

Use meaningful schemes:
- `config://` - Configuration data
- `db://` - Database records
- `api://` - External API data
- `file://` - File system content
- `users://` - User-related data

## Resource Checklist

- [ ] URI scheme is meaningful
- [ ] Description explains the data
- [ ] MIME type set if not plain text
- [ ] Templates have clear parameter names
- [ ] Completion hints if limited valid values
- [ ] Error cases return useful messages
