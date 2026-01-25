---
name: auth
description: Guide for integrating authentication providers. Use when adding auth to an application.
---

# Authentication Integration Guide

When invoked, help integrate an authentication provider into the application.

## Process

1. **Choose provider**: Auth0, Clerk, Supabase, Firebase, etc.
2. **Configure provider**: Create app, get credentials
3. **Implement flow**: Add login, callback, session management
4. **Protect routes**: Add middleware for protected endpoints
5. **Test**: Verify flow works end-to-end

## Provider Quick Reference

### Auth0
```
Dashboard: https://manage.auth0.com
Docs: https://auth0.com/docs

Credentials needed:
- Domain (your-tenant.auth0.com)
- Client ID
- Client Secret
- Audience (API identifier)
```

### Clerk
```
Dashboard: https://dashboard.clerk.com
Docs: https://clerk.com/docs

Credentials needed:
- Publishable Key (pk_...)
- Secret Key (sk_...)
```

### Supabase
```
Dashboard: https://app.supabase.com
Docs: https://supabase.com/docs/guides/auth

Credentials needed:
- Project URL
- Anon Key (public)
- Service Role Key (server-side only)
```

### Firebase
```
Console: https://console.firebase.google.com
Docs: https://firebase.google.com/docs/auth

Credentials needed:
- Project ID
- Client Email
- Private Key
```

## Implementation Patterns

### Backend (Express + Auth0)
```typescript
import { auth } from 'express-oauth2-jwt-bearer';

const jwtCheck = auth({
  audience: process.env.AUTH0_AUDIENCE,
  issuerBaseURL: `https://${process.env.AUTH0_DOMAIN}/`,
  tokenSigningAlg: 'RS256'
});

// Protect routes
app.use('/api', jwtCheck);

// Access user info
app.get('/api/me', (req, res) => {
  res.json({ userId: req.auth.sub });
});
```

### Frontend (React + Clerk)
```typescript
import { ClerkProvider, SignedIn, SignedOut } from '@clerk/clerk-react';

function App() {
  return (
    <ClerkProvider publishableKey={import.meta.env.VITE_CLERK_KEY}>
      <SignedIn>
        <Dashboard />
      </SignedIn>
      <SignedOut>
        <LoginPage />
      </SignedOut>
    </ClerkProvider>
  );
}
```

### Next.js (Middleware)
```typescript
// middleware.ts
import { clerkMiddleware, createRouteMatcher } from '@clerk/nextjs/server';

const isProtected = createRouteMatcher(['/dashboard(.*)', '/api(.*)']);

export default clerkMiddleware((auth, req) => {
  if (isProtected(req)) auth().protect();
});
```

## Environment Variables

```bash
# .env.example
# Auth0
AUTH0_DOMAIN=your-tenant.auth0.com
AUTH0_CLIENT_ID=your-client-id
AUTH0_CLIENT_SECRET=your-client-secret
AUTH0_AUDIENCE=https://your-api.example.com

# Clerk
CLERK_PUBLISHABLE_KEY=pk_test_...
CLERK_SECRET_KEY=sk_test_...

# Supabase
SUPABASE_URL=https://xxx.supabase.co
SUPABASE_ANON_KEY=eyJ...
SUPABASE_SERVICE_ROLE_KEY=eyJ...  # Server only!
```

## Common Flows

### Login Redirect
```
1. User clicks login
2. Redirect to provider's login page
3. User authenticates
4. Provider redirects to callback URL
5. Exchange code for tokens
6. Store tokens, redirect to app
```

### Token Refresh
```
1. Access token expires
2. Use refresh token to get new access token
3. If refresh fails, redirect to login
```

### Logout
```
1. Clear local tokens/session
2. Redirect to provider's logout endpoint
3. Provider clears their session
4. Redirect back to app
```

## Auth Checklist

- [ ] Provider account created
- [ ] Credentials in environment variables
- [ ] Callback URL configured in provider
- [ ] Login flow works
- [ ] Protected routes require auth
- [ ] Token refresh implemented
- [ ] Logout clears all sessions
- [ ] HTTPS in production
