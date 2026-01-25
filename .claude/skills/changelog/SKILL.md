---
name: changelog
description: Generate a changelog from git commits. Use when preparing release notes or documenting changes.
---

# Changelog Generation

When invoked, generate a user-friendly changelog from git history.

## Process

1. Get commits since last release: `git log <tag>..HEAD --oneline`
2. Categorize by type (feat, fix, etc.)
3. Transform technical commits into user-friendly descriptions
4. Format as changelog

## Commands

```bash
# Commits since last tag
git log v1.0.0..HEAD --oneline

# Commits in date range
git log --since="2024-01-01" --oneline

# Commits with details
git log v1.0.0..HEAD --pretty=format:"%h %s"
```

## Changelog Format

```markdown
# Changelog

## [1.1.0] - 2024-01-15

### Added
- OAuth2 login with GitHub and Google providers
- Dark mode theme option

### Changed
- Improved loading performance on dashboard
- Updated API response format for pagination

### Fixed
- Login redirect loop on expired sessions
- Memory leak in real-time notifications

### Security
- Fixed XSS vulnerability in comment rendering
```

## Category Mapping

| Commit Type | Changelog Section |
|-------------|------------------|
| feat | Added |
| fix | Fixed |
| perf | Changed (performance) |
| refactor | Changed |
| security | Security |
| deprecate | Deprecated |
| remove | Removed |

## Transformation Examples

**Technical commit:**
```
feat(auth): implement OAuth2 PKCE flow with Auth0 provider
```

**User-friendly changelog:**
```
- Added social login with GitHub and Google accounts
```

**Technical commit:**
```
fix(api): handle race condition in concurrent requests
```

**User-friendly changelog:**
```
- Fixed occasional errors when submitting forms quickly
```

## Output Template

```markdown
## [Version] - Date

### Added
- New features in user-friendly language

### Changed
- Improvements and updates

### Fixed
- Bug fixes users will notice

### Security
- Security-related fixes (without exploit details)

### Deprecated
- Features that will be removed

### Removed
- Features that were removed
```

## Tips

- Focus on user impact, not implementation
- Group related changes
- Filter out internal changes (refactor, chore, test)
- Include breaking changes prominently
- Link to issues/PRs for details
