---
name: pr-review
description: Review a pull request for quality, security, and correctness. Use when evaluating PRs.
---

# Pull Request Review

When invoked with a PR number, review it thoroughly.

## Process

1. Get PR details: `gh pr view <number>`
2. Get diff: `gh pr diff <number>`
3. Review against checklist
4. Categorize feedback by severity
5. Submit review

## Review Checklist

### Correctness
- [ ] Logic is correct
- [ ] Edge cases handled
- [ ] Error handling appropriate
- [ ] Types are correct

### Security
- [ ] No hardcoded secrets
- [ ] Input validation present
- [ ] No injection vulnerabilities
- [ ] Auth/authz correct

### Quality
- [ ] Code is readable
- [ ] No duplication
- [ ] Tests included
- [ ] Documentation updated

### Performance
- [ ] No obvious inefficiencies
- [ ] Appropriate data structures
- [ ] Async used correctly

## Feedback Categories

**Critical** - Must fix before merge
- Security issues
- Data loss potential
- Breaking bugs

**Warning** - Should fix
- Performance issues
- Missing error handling
- Poor test coverage

**Suggestion** - Nice to have
- Style improvements
- Minor refactoring
- Documentation enhancements

## Commands

```bash
# View PR
gh pr view 123

# Get diff
gh pr diff 123

# Checkout PR locally
gh pr checkout 123

# Submit review
gh pr review 123 --approve
gh pr review 123 --request-changes --body "Please fix..."
gh pr review 123 --comment --body "Looks good, minor suggestions"
```

## Review Comment Format

```markdown
### [Severity]: Brief title

**File**: `path/to/file.ts:42`

**Issue**: What's wrong and why it matters.

**Suggestion**: How to fix it.

```typescript
// Instead of
problematicCode();

// Consider
betterCode();
```
```

## Review Summary Template

```markdown
## PR Review: #123

### Summary
[Overall assessment in 1-2 sentences]

### Critical Issues
- Issue 1

### Suggestions
- Suggestion 1

### Questions
- Question about implementation choice

### Verdict
[APPROVED / REQUEST CHANGES]
```
