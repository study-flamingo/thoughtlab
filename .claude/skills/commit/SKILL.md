---
name: commit
description: Create well-formatted git commits following conventional commit standards. Use when ready to commit changes.
---

# Git Commit

When invoked, create a properly formatted commit.

## Process

1. Check `git status` for changes
2. Check `git diff --staged` for what will be committed
3. Stage relevant files if needed
4. Write conventional commit message
5. Commit

## Conventional Commit Format

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

### Types
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Formatting, no code change
- `refactor`: Code restructuring, no feature/fix
- `test`: Adding or updating tests
- `chore`: Maintenance, dependencies, tooling

### Scope (optional)
Area of the codebase: `auth`, `api`, `ui`, `db`, etc.

### Description
- Imperative mood ("add" not "added")
- Lowercase
- No period at end
- Under 72 characters

## Examples

```bash
# Feature
git commit -m "feat(auth): add OAuth2 login flow"

# Bug fix
git commit -m "fix(api): handle timeout errors gracefully"

# Documentation
git commit -m "docs: update API documentation"

# Refactor
git commit -m "refactor(db): extract query builder"

# With body
git commit -m "feat(auth): add refresh token rotation

Implements automatic token refresh when access token expires.
Tokens are rotated on each refresh for security.

Closes #123"
```

## Pre-Commit Checklist

Before committing:
- [ ] Changes compile/lint cleanly
- [ ] Tests pass
- [ ] No debug code or console.logs
- [ ] Only related changes included
- [ ] Sensitive data not included

## Commands

```bash
# Check status
git status

# Stage specific files
git add src/auth.ts src/middleware.ts

# Stage all changes
git add .

# Interactive staging
git add -p

# Commit
git commit -m "type(scope): description"

# Amend last commit (not yet pushed)
git commit --amend
```

## Breaking Changes

For breaking changes, add `!` after type or use footer:

```bash
git commit -m "feat(api)!: change response format"

# Or with footer
git commit -m "refactor(api): update endpoint structure

BREAKING CHANGE: /users endpoint now returns paginated response"
```
