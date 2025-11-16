# Frontend Tests

## Running Tests

```bash
cd frontend
npm install  # If not already done
npm test
```

## Test UI

```bash
npm run test:ui
```

## Test Coverage

```bash
npm run test:coverage
```

## Test Structure

- `src/components/__tests__/` - Component tests
- `src/services/__tests__/` - API service tests
- `src/test/setup.ts` - Test setup and configuration
- `src/test/utils.tsx` - Test utilities and helpers

## Testing Utilities

- `renderWithProviders` - Renders components with React Query provider
- Uses `@testing-library/react` for component testing
- Uses `vitest` as test runner

## Writing New Tests

1. Create test file: `*.test.tsx` or `*.test.ts`
2. Import from test utils: `import { render, screen } from '../test/utils'`
3. Use React Testing Library queries

Example:
```typescript
import { describe, it, expect } from 'vitest';
import { render, screen } from '../test/utils';
import MyComponent from '../MyComponent';

describe('MyComponent', () => {
  it('renders correctly', () => {
    render(<MyComponent />);
    expect(screen.getByText('Hello')).toBeInTheDocument();
  });
});
```
