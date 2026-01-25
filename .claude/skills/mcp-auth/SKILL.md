---
name: mcp-auth
description: Add authentication to an MCP server. Use when you need to secure your server with OAuth or JWT.
---

# Add MCP Authentication

When invoked, configure authentication for the MCP server.

## Authentication Options

1. **JWT Verification** - Validate tokens from an existing auth provider
2. **OAuth Provider** - Full OAuth flow (GitHub, Google, etc.)
3. **Custom Verification** - Your own token validation logic

## Python (FastMCP)

### JWT Verification
```python
from fastmcp import FastMCP
from fastmcp.server.auth import JWTVerifier

# For tokens from Auth0, Clerk, or similar
auth = JWTVerifier(
    public_key=public_key_pem,  # Get from provider's JWKS
    audience="my-mcp-server",
    issuer="https://your-auth-provider.com/"
)

mcp = FastMCP("secure-server", auth=auth)

@mcp.tool
def protected_action(data: str) -> str:
    """This tool requires authentication."""
    return f"Processed: {data}"

# Run with HTTP transport (required for auth)
if __name__ == "__main__":
    mcp.run(transport="http", port=8000)
```

### GitHub OAuth
```python
from fastmcp import FastMCP
from fastmcp.server.auth.providers.github import GitHubProvider

auth = GitHubProvider(
    client_id="your-github-client-id",
    client_secret="your-github-client-secret",
    base_url="http://localhost:8000"
)

mcp = FastMCP("github-auth-server", auth=auth)

@mcp.tool
def user_action(data: str, ctx: Context) -> str:
    """Action for authenticated GitHub users."""
    # Access user info from context
    user = ctx.request_context.get("user")
    return f"User {user['login']} processed: {data}"

if __name__ == "__main__":
    mcp.run(transport="http", port=8000)
```

### Google OAuth
```python
from fastmcp.server.auth.providers.google import GoogleProvider

auth = GoogleProvider(
    client_id="your-google-client-id",
    client_secret="your-google-client-secret",
    base_url="http://localhost:8000"
)

mcp = FastMCP("google-auth-server", auth=auth)
```

### Self-Issued JWT (for testing)
```python
from fastmcp.server.auth import JWTVerifier
from fastmcp.server.auth.providers.jwt import RSAKeyPair

# Generate key pair (do this once, save keys)
key_pair = RSAKeyPair.generate()

# Create a token for testing
access_token = key_pair.create_token(audience="my-server")
print(f"Test token: {access_token}")

# Verify tokens
auth = JWTVerifier(
    public_key=key_pair.public_key,
    audience="my-server"
)

mcp = FastMCP("jwt-server", auth=auth)
```

## TypeScript (@modelcontextprotocol/sdk)

### OAuth Proxy Provider
```typescript
import { ProxyOAuthServerProvider } from '@modelcontextprotocol/sdk/server/auth/providers/proxyProvider.js';
import { mcpAuthRouter } from '@modelcontextprotocol/sdk/server/auth/router.js';
import express from 'express';

const app = express();

// Configure OAuth with your provider (Auth0, Okta, etc.)
const authProvider = new ProxyOAuthServerProvider({
  endpoints: {
    authorizationUrl: 'https://your-provider.com/authorize',
    tokenUrl: 'https://your-provider.com/oauth/token',
    revocationUrl: 'https://your-provider.com/oauth/revoke'
  },
  verifyAccessToken: async (token) => {
    // Validate token with your provider
    const response = await fetch('https://your-provider.com/userinfo', {
      headers: { Authorization: `Bearer ${token}` }
    });

    if (!response.ok) {
      throw new Error('Invalid token');
    }

    return {
      token,
      clientId: 'my-client',
      scopes: ['read', 'write']
    };
  },
  getClient: async (clientId) => ({
    client_id: clientId,
    redirect_uris: ['http://localhost:3000/callback'],
    grant_types: ['authorization_code', 'refresh_token']
  })
});

// Mount auth routes
app.use(mcpAuthRouter({
  provider: authProvider,
  issuerUrl: new URL('https://your-provider.com'),
  baseUrl: new URL('http://localhost:3000'),
  serviceDocumentationUrl: new URL('https://docs.example.com/')
}));

app.listen(3000);
```

## Environment Variables

```bash
# .env.example
AUTH_CLIENT_ID=your-client-id
AUTH_CLIENT_SECRET=your-client-secret
AUTH_ISSUER=https://your-provider.com/
AUTH_AUDIENCE=your-api-identifier
JWT_PUBLIC_KEY=-----BEGIN PUBLIC KEY-----...
```

## Testing Authentication

```bash
# Get token from provider (example with Auth0)
curl -X POST https://your-tenant.auth0.com/oauth/token \
  -H "Content-Type: application/json" \
  -d '{
    "client_id": "your-client-id",
    "client_secret": "your-client-secret",
    "audience": "your-api",
    "grant_type": "client_credentials"
  }'

# Call authenticated MCP endpoint
curl -X POST http://localhost:8000/mcp \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"protected_action","arguments":{"data":"test"}}}'
```

## Auth Checklist

- [ ] Transport is HTTP (not stdio)
- [ ] Client ID and secret are in environment variables
- [ ] Audience/issuer configured correctly
- [ ] HTTPS in production
- [ ] Token validation is complete (signature, expiry, audience)
- [ ] Error messages don't leak sensitive info
