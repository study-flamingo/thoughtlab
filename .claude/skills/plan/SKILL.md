---
name: plan
description: Create a structured implementation plan before writing code. Use when starting a new feature or complex task.
---

# Implementation Planning

When invoked, create a clear, actionable plan before writing any code.

## Process

1. **Understand the goal** - What needs to be accomplished?
2. **Explore context** - What exists already? What patterns to follow?
3. **Break down tasks** - Decompose into small, verifiable steps
4. **Identify risks** - What might block progress?
5. **Get approval** - Confirm plan before executing

## Task Breakdown Guidelines

Good tasks are:
- **Atomic**: One thing to do
- **Verifiable**: Know when it's done
- **Ordered**: Dependencies clear
- **Sized right**: 15-60 minutes each

## Plan Template

```
## Plan: [Feature/Task Name]

### Goal
[One sentence describing success]

### Context
[Relevant existing code, patterns to follow]

### Tasks
1. [ ] First task
   - Acceptance: How to verify
   - Files: What will change

2. [ ] Second task
   - Acceptance: How to verify
   - Files: What will change

### Risks/Questions
- [ ] Risk 1: Mitigation
- [ ] Question needing answer

### Out of Scope
- Things explicitly not included
```

## After Planning

Once plan is approved:
1. Create todos using TodoWrite
2. Mark each task in_progress before starting
3. Mark complete immediately when done
4. Update plan if scope changes

## Example Usage

User: "Add OAuth login to the app"

Plan:
```
## Plan: OAuth Login Integration

### Goal
Users can log in via GitHub OAuth

### Context
- Using Express backend at src/api/
- Auth0 not currently set up
- Session stored in cookies

### Tasks
1. [ ] Set up Auth0 application
   - Acceptance: Client ID/secret obtained
   - Files: .env.example

2. [ ] Install Auth0 SDK
   - Acceptance: Package in dependencies
   - Files: package.json

3. [ ] Create /auth/login endpoint
   - Acceptance: Redirects to Auth0
   - Files: src/api/auth.ts

4. [ ] Create /auth/callback endpoint
   - Acceptance: Exchanges code for token
   - Files: src/api/auth.ts

5. [ ] Add auth middleware
   - Acceptance: Protected routes require login
   - Files: src/middleware/auth.ts

6. [ ] Add logout endpoint
   - Acceptance: Clears session
   - Files: src/api/auth.ts

### Risks/Questions
- [ ] Which OAuth provider? (GitHub, Google, both?)
- [ ] Session storage strategy (JWT vs server session)
```
