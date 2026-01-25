---
name: summarize
description: Summarize code, documentation, or findings into a concise format. Use after exploration to distill key information.
---

# Summarization

When invoked, create a clear, concise summary of the provided content.

## Summary Types

### Code Summary
Focus on:
- Purpose and responsibility
- Key functions/classes
- Dependencies and integrations
- Notable patterns or concerns

### Documentation Summary
Focus on:
- Main concepts
- Key API/features
- Important caveats
- Getting started essentials

### Research Summary
Focus on:
- Key findings
- Consensus vs conflicting opinions
- Actionable recommendations
- Source reliability

## Format Guidelines

### Short Summary (default)
- 3-5 bullet points
- Key facts only
- Actionable insights

### Detailed Summary
- Section headings
- Supporting details
- Code examples if relevant
- References to sources

## Output Template

```
## Summary: [Topic]

### Key Points
- Point 1
- Point 2
- Point 3

### Details
[Expanded information if needed]

### Recommendations
[What to do with this information]

### Sources/References
[Where this came from]
```

## Example Usage

User: "Summarize how this authentication system works"

After reading the relevant files:
```
## Summary: Authentication System

### Key Points
- Uses JWT tokens with 1-hour expiry
- Auth0 as identity provider
- Refresh tokens stored in httpOnly cookies
- Middleware validates on every API request

### Flow
1. User logs in via Auth0 redirect
2. Callback exchanges code for tokens
3. Access token used for API calls
4. Refresh happens automatically when expired

### Key Files
- `src/auth/middleware.ts:15` - Token validation
- `src/auth/callback.ts:42` - Token exchange
- `src/auth/refresh.ts:8` - Refresh logic
```
