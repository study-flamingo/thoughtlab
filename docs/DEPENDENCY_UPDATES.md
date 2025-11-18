# Dependency Updates

This document tracks the dependency versions and updates made.

## Backend Dependencies (Python)

Updated to latest stable versions as of 2025:

### Major Updates

- **FastAPI**: `0.109.0` → `0.121.2` (Latest stable)
- **Uvicorn**: `0.27.0` → `0.32.1` (Latest stable)
- **Neo4j**: `5.17.0` → `5.26.0` (Latest stable)
- **SQLAlchemy**: `2.0.25` → `2.0.36` (Latest 2.0.x)
- **Redis**: `5.0.1` → `5.2.1` (Latest stable)
- **sentence-transformers**: `2.3.1` → `5.1.2` (Major version update - significant API changes)
- **litellm**: `1.17.9` → `1.62.3` (Latest stable)
- **numpy**: `1.26.3` → `>=1.24.0,<3.0.0` (Flexible version - supports both 1.x and 2.x)
- **pytest**: `8.0.0` → `8.3.4` (Latest 8.x)
- **httpx**: `0.26.0` → `0.28.1` (Latest stable)

### Minor Updates

- **python-multipart**: `0.0.6` → `0.0.12`
- **asyncpg**: `0.29.0` → `0.30.0`
- **pydantic-settings**: `2.1.0` → `2.6.1`
- **arq**: `0.25.0` → `0.29.0`
- **pytest-asyncio**: `0.23.3` → `0.24.0`

### Unchanged

- **python-dotenv**: `1.0.1` (Already latest)
- **python-jose**: `3.3.0` (Already latest)
- **passlib**: `1.7.4` (Already latest)

## Frontend Dependencies (Node.js)

Updated to latest stable versions within same major versions (conservative approach):

### Updates (Same Major Version)

- **React**: `18.2.0` → `18.3.1` (Latest 18.x - React 19 available but has breaking changes)
- **@tanstack/react-query**: `5.17.0` → `5.62.11` (Latest 5.x)
- **axios**: `1.6.5` → `1.7.9` (Latest stable)
- **cytoscape**: `3.27.0` → `3.31.0` (Latest stable)
- **zustand**: `4.4.7` → `4.5.7` (Latest 4.x - 5.x available but has breaking changes)
- **vite**: `5.0.11` → `5.4.21` (Latest 5.x - 7.x available but has breaking changes)
- **vitest**: `1.1.0` → `2.1.8` (Latest 2.x - 4.x available but has breaking changes)
- **TypeScript**: `5.3.3` → `5.7.2` (Latest 5.x)
- **ESLint**: `8.56.0` → `8.57.1` (Latest 8.x - 9.x available but has breaking changes)

### Minor Updates

- **lucide-react**: `0.309.0` → `0.468.0`
- **@testing-library/react**: `14.1.2` → `14.3.1` (Latest 14.x - 16.x available)
- **@testing-library/user-event**: `14.5.1` → `14.5.2`
- **@types/react**: `18.2.48` → `18.3.18`
- **@typescript-eslint/eslint-plugin**: `6.18.1` → `7.18.0` (Latest 7.x - 8.x available)
- **@typescript-eslint/parser**: `6.18.1` → `7.18.0` (Latest 7.x - 8.x available)
- **@vitejs/plugin-react**: `4.2.1` → `4.7.0`
- **tailwindcss**: `3.4.1` → `3.4.17` (Latest 3.x - 4.x available but has breaking changes)
- **postcss**: `8.4.33` → `8.4.49`
- **jsdom**: `23.0.1` → `26.0.1` (Major version update - should be compatible)

## Compatibility Notes

### Breaking Changes to Watch For

1. **sentence-transformers 5.x**: Major version jump from 2.x - significant API changes. Test thoroughly.
2. **numpy 2.x**: May have breaking changes from 1.x. Using flexible version range to support both.
3. **vitest 2.x**: May have breaking changes from 1.x (updated but test thoroughly)

### Major Versions Available But Not Updated (Breaking Changes)

- **React 19.x**: Available but has breaking changes from 18.x
- **Vite 7.x**: Available but has breaking changes from 5.x
- **Vitest 4.x**: Available but has breaking changes from 2.x
- **ESLint 9.x**: Configuration format changed, requires eslint.config.js
- **Tailwind CSS 4.x**: Available but has breaking changes from 3.x
- **zustand 5.x**: Available but has breaking changes from 4.x
- **@testing-library/react 16.x**: Available but may have breaking changes

These can be updated in a future major version update after thorough testing.

### Testing Required

After updating, test:
- [ ] Backend API endpoints work
- [ ] Frontend builds successfully
- [ ] Graph visualization works
- [ ] Tests pass
- [ ] No runtime errors

## Update Date

Updated: 2025-01-16
