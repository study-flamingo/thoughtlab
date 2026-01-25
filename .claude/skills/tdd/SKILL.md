---
name: tdd
description: Follow test-driven development. Write a failing test first, then implement, then refactor. Use when implementing features or fixing bugs.
---

# Test-Driven Development

When invoked, follow the Red-Green-Refactor cycle strictly.

## The TDD Cycle

### 1. RED - Write a Failing Test
```
- Write the smallest test that demonstrates desired behavior
- Run it and confirm it FAILS
- If it passes, the test isn't testing the right thing
```

### 2. GREEN - Make it Pass
```
- Write the minimum code to make the test pass
- Don't optimize or generalize yet
- Just make it work
```

### 3. REFACTOR - Clean Up
```
- Improve code quality while keeping tests green
- Remove duplication
- Improve naming
- Extract functions if needed
```

## Rules

1. **Never write implementation before a failing test**
2. **Watch the test fail** - If you didn't see it fail, you don't know it tests the right thing
3. **Minimal implementation** - Only write code required to pass the current test
4. **One test at a time** - Don't write multiple tests before implementing

## Test Structure

```typescript
describe('Feature', () => {
  it('should do expected behavior', () => {
    // Arrange - set up test data
    const input = { name: 'test' };

    // Act - call the function
    const result = processInput(input);

    // Assert - verify the outcome
    expect(result.processed).toBe(true);
  });
});
```

## Common Test Cases

1. **Happy path** - Normal expected input
2. **Edge cases** - Empty, null, boundary values
3. **Error cases** - Invalid input, failure conditions
4. **Integration** - Components working together

## Example TDD Session

Task: Implement a function that validates email addresses

### Step 1: RED
```typescript
// Write the test first
describe('validateEmail', () => {
  it('should return true for valid email', () => {
    expect(validateEmail('user@example.com')).toBe(true);
  });
});
```
Run test → FAILS (function doesn't exist)

### Step 2: GREEN
```typescript
// Minimal implementation
function validateEmail(email: string): boolean {
  return email.includes('@');
}
```
Run test → PASSES

### Step 3: Add more tests (RED again)
```typescript
it('should return false for email without domain', () => {
  expect(validateEmail('user@')).toBe(false);
});
```
Run test → FAILS

### Step 4: Improve implementation (GREEN)
```typescript
function validateEmail(email: string): boolean {
  const parts = email.split('@');
  return parts.length === 2 && parts[1].includes('.');
}
```
Run test → PASSES

### Step 5: REFACTOR
```typescript
function validateEmail(email: string): boolean {
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  return emailRegex.test(email);
}
```
All tests still pass.

## Anti-Patterns to Avoid

- Writing tests after implementation
- Writing multiple tests before implementing
- Testing implementation details, not behavior
- Skipping the "watch it fail" step
- Over-engineering in the GREEN phase
