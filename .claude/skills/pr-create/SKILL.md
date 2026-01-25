---
name: pr-create
description: Create a pull request with a well-structured description. Use when ready to submit changes for review.
---

# Create Pull Request

When invoked, create a PR with a complete description.

## Process

1. Ensure branch is pushed: `git push -u origin <branch>`
2. Check what will be included: `git log main..HEAD`
3. Create PR with structured description

## PR Title Format

Follow conventional commit style:
```
feat(scope): brief description
fix(scope): brief description
refactor(scope): brief description
```

## PR Description Template

```markdown
## Summary
<!-- 1-3 bullet points describing what this PR does -->

- Added X feature
- Fixed Y bug
- Updated Z documentation

## Changes
<!-- Detailed list of changes -->

- `src/auth.ts`: Added OAuth flow
- `src/middleware.ts`: Token validation
- `tests/auth.test.ts`: Test coverage

## Test Plan
<!-- How to verify this works -->

- [ ] Unit tests pass
- [ ] Manual testing of login flow
- [ ] Verified in staging

## Screenshots
<!-- If applicable, add screenshots -->

## Related Issues
<!-- Link to issues this addresses -->

Closes #123
Related to #456
```

## Commands

```bash
# Push branch
git push -u origin feature/my-feature

# Create PR
gh pr create \
  --title "feat(auth): add OAuth login" \
  --body "## Summary
- Added OAuth2 authentication
- Integrated with Auth0

## Test Plan
- [ ] Unit tests pass
- [ ] Manual login testing"

# Create draft PR
gh pr create --draft --title "WIP: feature"

# Create with reviewers
gh pr create --reviewer @username --title "feat: feature"
```

## Pre-PR Checklist

- [ ] Branch is up to date with main
- [ ] All tests pass
- [ ] Linting passes
- [ ] No merge conflicts
- [ ] Changes are focused (single concern)
- [ ] Documentation updated if needed
- [ ] Screenshots for UI changes

## After Creating

1. Add reviewers if not done
2. Link to related issues
3. Add labels if needed
4. Monitor CI checks
