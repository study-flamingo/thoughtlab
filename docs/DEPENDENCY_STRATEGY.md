# Dependency Management Strategy

## Approach

We use a **minimal constraint** approach for dependency management:

1. **Core packages only**: Only packages directly imported/used in our code are constrained
2. **Minimum versions**: Use `>=` constraints to specify minimum required versions
3. **Let uv resolve**: Transitive dependencies are resolved automatically by `uv`

## Core Packages (Constrained)

These are packages we directly import and use:

- **fastapi** - Web framework
- **uvicorn** - ASGI server
- **neo4j** - Graph database driver
- **asyncpg** - PostgreSQL async driver
- **sqlalchemy** - ORM
- **redis** - Redis client
- **python-dotenv** - Environment variable loading
- **pydantic-settings** - Settings management
- **sentence-transformers** - ML embeddings
- **litellm** - LLM integration
- **arq** - Background task queue
- **python-jose** - JWT handling
- **passlib** - Password hashing
- **pytest** - Testing framework
- **pytest-asyncio** - Async test support
- **httpx** - HTTP client for tests

## Transitive Dependencies (Unconstrained)

These are automatically resolved by `uv`:

- **numpy** - Required by sentence-transformers and other ML packages
- **python-multipart** - Required by FastAPI for file uploads
- **pydantic** - Required by pydantic-settings and FastAPI
- **torch** - Required by sentence-transformers
- And many others...

## Benefits

1. **Faster resolution**: uv can find compatible versions more quickly
2. **Fewer conflicts**: Less chance of over-constraining dependencies
3. **Security updates**: Transitive dependencies get updated automatically
4. **Simpler maintenance**: Only need to update core packages

## Updating Dependencies

To update all dependencies to latest compatible versions:

```bash
cd backend
uv pip install --upgrade -r requirements.txt
```

To update a specific core package:

```bash
uv pip install --upgrade fastapi
# Then update requirements.txt if needed
```

## Lock File (Future Consideration)

For production deployments, consider generating a lock file:

```bash
uv pip compile requirements.txt -o requirements.lock
```

This creates a fully resolved, reproducible set of all dependencies.
