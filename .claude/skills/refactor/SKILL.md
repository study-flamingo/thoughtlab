---
name: refactor
description: Safely refactor code without changing behavior. Use when improving code quality, extracting functions, or restructuring.
---

# Safe Refactoring

When invoked, improve code structure while preserving behavior.

## Golden Rule

**Tests must pass before AND after every change.**

If there are no tests, write them first.

## Process

1. **Ensure test coverage** - Tests for current behavior
2. **Make one small change** - Single refactoring step
3. **Run tests** - Verify nothing broke
4. **Repeat** - Next small change

## Common Refactorings

### Extract Function
```typescript
// Before
function processOrder(order) {
  // validate
  if (!order.items || order.items.length === 0) {
    throw new Error('Empty order');
  }
  if (!order.customer) {
    throw new Error('No customer');
  }
  // process...
}

// After
function validateOrder(order) {
  if (!order.items || order.items.length === 0) {
    throw new Error('Empty order');
  }
  if (!order.customer) {
    throw new Error('No customer');
  }
}

function processOrder(order) {
  validateOrder(order);
  // process...
}
```

### Extract Variable
```typescript
// Before
if (user.age >= 18 && user.country === 'US' && user.verified) {
  // ...
}

// After
const isEligible = user.age >= 18 && user.country === 'US' && user.verified;
if (isEligible) {
  // ...
}
```

### Rename
```typescript
// Before
const d = new Date();
function calc(x, y) { return x + y; }

// After
const currentDate = new Date();
function calculateSum(first, second) { return first + second; }
```

### Remove Duplication
```typescript
// Before
function createUser(name) {
  const id = crypto.randomUUID();
  const createdAt = new Date().toISOString();
  return { id, name, createdAt };
}

function createPost(title) {
  const id = crypto.randomUUID();
  const createdAt = new Date().toISOString();
  return { id, title, createdAt };
}

// After
function withMetadata<T>(data: T) {
  return {
    id: crypto.randomUUID(),
    createdAt: new Date().toISOString(),
    ...data
  };
}

function createUser(name) {
  return withMetadata({ name });
}

function createPost(title) {
  return withMetadata({ title });
}
```

### Simplify Conditional
```typescript
// Before
function getDiscount(user) {
  if (user.isPremium) {
    if (user.yearsActive > 5) {
      return 0.2;
    } else {
      return 0.1;
    }
  } else {
    return 0;
  }
}

// After
function getDiscount(user) {
  if (!user.isPremium) return 0;
  if (user.yearsActive > 5) return 0.2;
  return 0.1;
}
```

## Refactoring Checklist

Before:
- [ ] Tests exist for current behavior
- [ ] All tests pass
- [ ] Code is committed (can revert)

During:
- [ ] One change at a time
- [ ] Tests after each change
- [ ] No behavior changes

After:
- [ ] All tests still pass
- [ ] Code is cleaner/clearer
- [ ] Commit with descriptive message

## When NOT to Refactor

- No test coverage (write tests first)
- Under time pressure (do it properly later)
- While fixing a bug (fix first, refactor separately)
- Major architecture change (needs planning)

## Commit Messages

```
refactor: extract validation into separate function
refactor: rename variables for clarity
refactor: simplify conditional logic in getDiscount
refactor: remove duplicate code in entity creation
```
