---
name: ticket
description: Work on a ticket/issue from start to finish. Use when implementing a feature or fix from an issue tracker.
---

# Ticket Workflow

When invoked with a ticket reference, work through it end-to-end.

## Process

### 1. Understand
- Get ticket details (title, description, acceptance criteria)
- Review linked issues, comments, designs
- Ask clarifying questions if needed

### 2. Prepare
- Explore codebase for relevant files
- Understand current implementation
- Create feature branch

### 3. Implement
- Follow TDD where appropriate
- Make incremental commits
- Keep ticket updated with progress

### 4. Complete
- Create pull request
- Link PR to ticket
- Move ticket to review

## Branch Naming

```
{type}/{ticket-id}-{description}

Examples:
feature/PROJ-123-oauth-login
fix/PROJ-456-login-redirect
```

## Commit References

Include ticket ID in commits:
```bash
git commit -m "feat(auth): add OAuth flow

Implements PROJ-123"
```

## PR Title Format

```
feat(PROJ-123): add OAuth login flow
fix(PROJ-456): resolve login redirect loop
```

## Commands

```bash
# GitHub Issues
gh issue view 123
gh issue list --label "bug"

# Create branch
git checkout -b feature/PROJ-123-description

# Create PR linked to issue
gh pr create --title "feat(PROJ-123): description"
```

## Progress Updates

Keep ticket updated:
- When starting: "Starting implementation"
- When blocked: "Blocked on X, waiting for Y"
- When PR ready: "PR #456 ready for review"

## Checklist

### Before Starting
- [ ] Requirements are clear
- [ ] Acceptance criteria defined
- [ ] Dependencies identified

### During Implementation
- [ ] Branch created with ticket ID
- [ ] Commits reference ticket
- [ ] Tests written
- [ ] Documentation updated

### Before Completing
- [ ] All acceptance criteria met
- [ ] PR created and linked
- [ ] Ticket moved to review
- [ ] Reviewers assigned

## Ticket Statuses

Typical flow:
```
Backlog → In Progress → In Review → Done
```

Update status as you work:
1. Move to "In Progress" when starting
2. Move to "In Review" when PR created
3. Move to "Done" when merged
