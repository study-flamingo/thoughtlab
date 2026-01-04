# Planning Guide for ThoughtLab

This document provides a template and guidelines for creating architectural plans for significant refactoring or feature work in ThoughtLab.

---

## When to Create a Plan

Create a formal plan when:
- Refactoring spans multiple modules or layers
- Adding new infrastructure (job queues, external services, etc.)
- Changing core architectural patterns
- Work spans 3+ phases or affects 5+ files

Skip formal planning for:
- Bug fixes with clear scope
- Single-file changes
- Documentation updates
- Dependency bumps

---

## Plan Structure Template

### 1. Goal

A brief statement (2-4 bullet points) of what the refactor achieves:

```markdown
## Goal
Refactor [system] into [architecture] with:
- [Key outcome 1]
- [Key outcome 2]
- [Key outcome 3]
```

### 2. Architecture Diagram

ASCII diagram showing the high-level structure:

```markdown
## Proposed Architecture

```text
┌─────────────────────────────────┐
│       External Consumers         │
└────────────┬────────────────────┘
             │
┌────────────▼────────────────────┐
│       Application Layer          │
│  ┌──────────┐    ┌──────────┐   │
│  │ Layer A  │    │ Layer B  │   │
│  └────┬─────┘    └────┬─────┘   │
│       │               │          │
│       └───────┬───────┘          │
│               ▼                  │
│  ┌─────────────────────────┐    │
│  │    Core Service Layer    │    │
│  └─────────────────────────┘    │
└─────────────────────────────────┘
`` `
```

Include **Key Flow** descriptions explaining request paths through the system.

### 3. Directory Structure

Show the final state of the file structure:

```markdown
## Directory Structure (Final State)

```text
backend/app/
├── models/
│   └── new_models.py       # Description
├── services/
│   └── new_service/        # Description
│       ├── __init__.py
│       └── ...
└── ...
`` `
```

### 4. Key Design Decisions

Numbered sections explaining important architectural choices:

```markdown
## Key Design Decisions

### 1. [Decision Name]
[Explanation of the decision, rationale, and implications]

### 2. [Another Decision]
...
```

Include:
- **Why** the decision was made
- **Trade-offs** considered
- **Code examples** where helpful
- **Configuration** requirements

### 5. Dependencies & Configuration

Document required packages and environment variables:

```markdown
## Dependencies & Configuration

**Required Packages** (in `pyproject.toml`):
- `package>=version` - Purpose

**Environment Variables:**
```bash
VARIABLE_NAME=default  # Description
`` `

**Existing Infrastructure:**
- What existing systems are reused
```

### 6. Migration Phases

Break the work into atomic phases:

```markdown
## Migration Phases

### Phase 1: [Phase Name]
1. Step one
2. Step two
3. Verify tests pass

**Files:**
- CREATE: `path/to/new/file.py`
- MODIFY: `path/to/existing/file.py`
- RENAME: `old/path.py` → `new/path.py`
- DELETE: `path/to/remove.py`

### Phase 2: [Next Phase]
...
```

Each phase should:
- Be independently testable
- List specific file changes
- End with verification step

### 7. Backwards Compatibility

Document what changes and what stays the same:

```markdown
## Backwards Compatibility
- [Unchanged aspect]
- [Changed aspect with migration path]
- [New capability]
```

### 8. Files Summary

Quick reference table of all file operations:

```markdown
## Files Summary

| Action | File |
|--------|------|
| CREATE | `path/to/file.py` |
| MODIFY | `path/to/file.py` |
| RENAME | `old.py` → `new.py` |
| DELETE | `path/to/file.py` |
```

---

## Best Practices

### Naming Conventions

Use descriptive file names that indicate purpose:
- `tool_models.py` not just `models.py`
- `tool_routes.py` not just `routes.py`
- `mcp_tools.py` for MCP-specific wrappers

### Phase Sizing

- Each phase should be completable in one session
- Phases should be independently testable
- Group related changes together
- Keep the total number of phases manageable (5-8 ideal)

### Testing Strategy

Always include a testing strategy:
- Which existing tests verify the refactor?
- When to update/add tests (during or after)?
- What manual testing is needed?

### Documentation Updates

Include documentation in the final phase:
- `PROJECT_MAP.md` - File structure changes
- `DEVELOPMENT_GUIDE.md` - Architecture changes
- `CHANGELOG.md` - User-facing changes

---

## Example Reference

See [Tool Layer Architecture Refactor Plan](../../../.claude/plans/fluttering-singing-thompson.md) for a complete example of this template in use.
