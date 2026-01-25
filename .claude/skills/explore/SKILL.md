---
name: explore
description: Quickly explore and understand a codebase. Use when you need to find files, understand structure, or locate specific functionality.
---

# Codebase Exploration

When invoked, systematically explore the codebase to answer the user's question.

## Process

1. **Clarify the goal** - What are we looking for?
2. **Start broad** - Check project structure, README, package files
3. **Narrow down** - Use Glob and Grep to find relevant files
4. **Read and synthesize** - Understand what you find

## Exploration Commands

### Project structure
```
Glob: *
Glob: src/**/*
```

### Find by file type
```
Glob: **/*.ts
Glob: **/*.py
Glob: **/*.md
```

### Find by name pattern
```
Glob: **/*config*
Glob: **/*auth*
Glob: **/test*
```

### Find by content
```
Grep: "function functionName"
Grep: "class ClassName"
Grep: "import.*from"
```

## Output Format

Always provide:
- File paths with line numbers
- Brief explanation of what each finding means
- Suggested next steps if deeper investigation needed

## Example Usage

User: "Where is authentication handled?"

Response approach:
1. Grep for auth-related terms: `auth`, `login`, `session`, `jwt`
2. Check common locations: `src/auth/`, `lib/auth/`, `middleware/`
3. Read key files to understand the flow
4. Summarize with file paths and brief descriptions
