---
name: docstring
description: Add documentation to functions and classes. Use when code needs inline documentation.
---

# Code Documentation

When invoked, add or improve docstrings and comments.

## Docstring Formats

### Python (Google Style)
```python
def fetch_user(user_id: str, include_posts: bool = False) -> User | None:
    """Fetch a user by their unique identifier.

    Retrieves user data from the database, optionally including
    their associated posts.

    Args:
        user_id: The unique identifier of the user to fetch.
        include_posts: Whether to include the user's posts.
            Defaults to False for performance.

    Returns:
        The User object if found, None otherwise.

    Raises:
        DatabaseError: If the database connection fails.
        ValidationError: If user_id is not a valid UUID.

    Example:
        >>> user = fetch_user("123e4567-e89b-12d3-a456-426614174000")
        >>> user.name
        'John Doe'
    """
```

### Python (Class)
```python
class UserService:
    """Service for managing user operations.

    Provides methods for creating, reading, updating, and deleting
    users in the system. Handles authentication and authorization
    checks internally.

    Attributes:
        db: Database connection instance.
        cache: Optional cache for frequently accessed users.

    Example:
        >>> service = UserService(db=database)
        >>> user = service.create(email="user@example.com")
    """

    def __init__(self, db: Database, cache: Cache | None = None):
        """Initialize the UserService.

        Args:
            db: Database connection to use for operations.
            cache: Optional cache instance for performance.
        """
```

### TypeScript (JSDoc)
```typescript
/**
 * Fetch a user by their unique identifier.
 *
 * @param userId - The unique identifier of the user
 * @param options - Optional configuration
 * @param options.includePosts - Whether to include user's posts
 * @returns The user if found, null otherwise
 * @throws {DatabaseError} If database connection fails
 *
 * @example
 * ```typescript
 * const user = await fetchUser('123', { includePosts: true });
 * console.log(user?.name);
 * ```
 */
async function fetchUser(
  userId: string,
  options?: { includePosts?: boolean }
): Promise<User | null> {
```

### TypeScript (Interface)
```typescript
/**
 * Configuration options for the API client.
 */
interface ApiClientOptions {
  /** Base URL for API requests */
  baseUrl: string;

  /** Request timeout in milliseconds (default: 30000) */
  timeout?: number;

  /** Custom headers to include in all requests */
  headers?: Record<string, string>;

  /**
   * Retry configuration for failed requests.
   * Set to false to disable retries.
   */
  retry?: RetryOptions | false;
}
```

## When to Document

**Always document:**
- Public APIs (exported functions, classes)
- Complex logic that isn't self-evident
- Non-obvious parameters or return values
- Side effects
- Exceptions/errors that can be thrown

**Skip documentation for:**
- Private implementation details (unless complex)
- Self-explanatory code (`getName()` returning `name`)
- Simple getters/setters

## Documentation Quality

### Good
```python
def calculate_shipping(weight: float, distance: float) -> Decimal:
    """Calculate shipping cost based on package weight and distance.

    Uses the standard rate of $0.50 per pound plus $0.10 per mile.
    Minimum charge is $5.00.

    Args:
        weight: Package weight in pounds (must be positive)
        distance: Shipping distance in miles (must be positive)

    Returns:
        Shipping cost as a Decimal, minimum $5.00

    Raises:
        ValueError: If weight or distance is not positive
    """
```

### Bad
```python
def calculate_shipping(weight: float, distance: float) -> Decimal:
    """Calculate shipping."""  # Too vague, missing details
```

## Comment Guidelines

### Good comments
```python
# Use binary search for O(log n) lookup in sorted list
index = bisect.bisect_left(sorted_items, target)

# Auth0 requires audience to match exactly, including trailing slash
audience = f"{base_url}/"
```

### Bad comments
```python
# Increment i
i += 1

# Get user
user = get_user(id)
```

## Checklist

- [ ] All public functions have docstrings
- [ ] Parameters and returns documented
- [ ] Exceptions documented
- [ ] Examples for complex usage
- [ ] Comments explain "why" not "what"
