# Testing Strategy - Improved Approach

## Current Issue
After attempting to refactor tests from heavy mocking to MSW-based integration testing, we encountered significant challenges with axios + MSW + jsdom compatibility in the test environment.

## The Problem
- **axios in jsdom**: Requires proper base URL configuration to work with relative paths
- **MSW interception**: Not intercepting axios requests properly in test environment
- **URL parsing errors**: `Invalid URL '/api/v1/graph/full'` - axios can't create valid URL

## Root Cause
The axios instance in `api.ts` uses `baseURL: '/api/v1'` which works fine in browser/Vite dev server environment, but in jsdom test environment, axios needs an absolute base URL to create valid requests.

## Pragmatic Solution

### 1. Keep Current Test Structure, But Improve It
Instead of pure MSW integration tests, use a **hybrid approach**:

```typescript
// Use MSW for API simulation but with proper configuration
import { setupServer } from 'msw/node'
import { http, HttpResponse } from 'msw'

// Configure axios for test environment
axios.defaults.baseURL = 'http://localhost:3000'

const server = setupServer(
  http.get('http://localhost:3000/api/v1/graph/full', () => {
    return HttpResponse.json(mockData)
  })
)
```

### 2. Test Pyramid Strategy
```
┌─────────────────────────────────┐
│  E2E Tests (Playwright)         │ ← Test real user flows
│  (Few, critical paths)          │
├─────────────────────────────────┤
│  Integration Tests (React)      │ ← Test components with MSW
│  (More, component groups)       │
├─────────────────────────────────┤
│  Unit Tests (Vitest)            │ ← Test pure functions
│  (Most, utilities)              │
└─────────────────────────────────┘
```

### 3. Recommended Test Types

**Integration Tests (Current Focus):**
- ✅ Test component behavior with mocked API responses
- ✅ Test React Query integration
- ✅ Test user interactions and state changes
- ✅ Test error boundaries and loading states

**Not:**
- ❌ Don't test library internals (axios, Cytoscape)
- ❌ Don't test CSS styling (that's what visual regression tests are for)

### 4. File Structure for Tests
```
frontend/src/
├── components/
│   └── __tests__/
│       ├── __integration__/     ← Integration tests with MSW
│       ├── __unit__/            ← Unit tests for utilities
│       └── __e2e__/             ← E2E tests (Playwright)
├── test/
│   ├── mocks/                   ← MSW handlers
│   ├── utils.tsx               ← Test utilities
│   └── setup.ts                ← Global test setup
```

## Implementation Plan

### Phase 1: Fix Current Integration Tests (✅ In Progress)
1. Configure axios for test environment properly
2. Use MSW with absolute URLs
3. Test core user flows

### Phase 2: Add Unit Tests
1. Test utility functions (no mocks needed)
2. Test data transformation functions
3. Test validation logic

### Phase 3: Add E2E Tests
1. Set up Playwright for critical user flows
2. Test complete user journeys
3. Visual regression testing

### Phase 4: Add Visual Regression Tests
1. Use tools like Storybook + Chromatic
2. Test component rendering across browsers
3. Catch CSS regressions

## Alternative Approaches Considered

### Option A: MSW + Axios (Current)
**Pros:**
- Realistic API simulation
- No axios mock maintenance
- Tests actual network layer

**Cons:**
- Complex setup in jsdom
- URL configuration issues
- Environment differences

### Option B: Mock Service Layer (Recommended)
**Pros:**
- Simple, reliable
- Focus on component behavior
- Fast execution
- Easy to maintain

**Cons:**
- Requires mock maintenance
- Less realistic network layer

### Option C: E2E Only
**Pros:**
- Most realistic
- Tests complete flows

**Cons:**
- Slow
- Resource intensive
- Hard to test edge cases

## Recommendation

Given the complexity we've encountered and the need for rock-solid tests, I recommend:

1. **Use Option B** (mock service layer) for unit/integration tests
2. **Keep MSW** for E2E tests and critical integration scenarios
3. **Focus on testing behavior** rather than implementation
4. **Add comprehensive test coverage** for critical paths

## Next Steps

1. Clean up current test setup
2. Implement service layer mocking
3. Write comprehensive integration tests
4. Add E2E tests for critical flows
5. Establish test quality standards

This approach gives us reliable, maintainable tests while avoiding the complexity issues we've encountered with MSW in jsdom.