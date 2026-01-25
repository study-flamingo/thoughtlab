---
name: api-spec
description: Generate OpenAPI specifications for REST APIs. Use when documenting or designing APIs.
---

# OpenAPI Specification Generation

When invoked, create or update OpenAPI spec for the API.

## Process

1. **Identify endpoints**: List all API routes
2. **Document schemas**: Define request/response types
3. **Add metadata**: Descriptions, examples, errors
4. **Validate**: Ensure spec is complete and valid

## OpenAPI 3.1 Template

```yaml
openapi: 3.1.0
info:
  title: My API
  version: 1.0.0
  description: API description here

servers:
  - url: https://api.example.com/v1
    description: Production
  - url: http://localhost:3000/v1
    description: Development

paths:
  /users:
    get:
      summary: List users
      operationId: listUsers
      tags: [Users]
      parameters:
        - $ref: '#/components/parameters/Limit'
        - $ref: '#/components/parameters/Offset'
      responses:
        '200':
          description: List of users
          content:
            application/json:
              schema:
                type: object
                properties:
                  data:
                    type: array
                    items:
                      $ref: '#/components/schemas/User'
                  pagination:
                    $ref: '#/components/schemas/Pagination'
        '401':
          $ref: '#/components/responses/Unauthorized'

    post:
      summary: Create user
      operationId: createUser
      tags: [Users]
      security:
        - bearerAuth: []
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/CreateUser'
      responses:
        '201':
          description: User created
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/User'
        '400':
          $ref: '#/components/responses/BadRequest'
        '401':
          $ref: '#/components/responses/Unauthorized'

  /users/{userId}:
    get:
      summary: Get user by ID
      operationId: getUser
      tags: [Users]
      parameters:
        - name: userId
          in: path
          required: true
          schema:
            type: string
            format: uuid
      responses:
        '200':
          description: User found
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/User'
        '404':
          $ref: '#/components/responses/NotFound'

components:
  securitySchemes:
    bearerAuth:
      type: http
      scheme: bearer
      bearerFormat: JWT

  schemas:
    User:
      type: object
      required: [id, email, createdAt]
      properties:
        id:
          type: string
          format: uuid
          example: "123e4567-e89b-12d3-a456-426614174000"
        email:
          type: string
          format: email
          example: "user@example.com"
        name:
          type: string
          example: "John Doe"
        createdAt:
          type: string
          format: date-time

    CreateUser:
      type: object
      required: [email]
      properties:
        email:
          type: string
          format: email
        name:
          type: string

    Pagination:
      type: object
      properties:
        total:
          type: integer
        offset:
          type: integer
        limit:
          type: integer

    Error:
      type: object
      required: [code, message]
      properties:
        code:
          type: string
        message:
          type: string
        details:
          type: array
          items:
            type: object
            properties:
              field:
                type: string
              message:
                type: string

  parameters:
    Limit:
      name: limit
      in: query
      schema:
        type: integer
        default: 20
        maximum: 100

    Offset:
      name: offset
      in: query
      schema:
        type: integer
        default: 0

  responses:
    BadRequest:
      description: Invalid request
      content:
        application/json:
          schema:
            $ref: '#/components/schemas/Error'

    Unauthorized:
      description: Authentication required
      content:
        application/json:
          schema:
            $ref: '#/components/schemas/Error'

    NotFound:
      description: Resource not found
      content:
        application/json:
          schema:
            $ref: '#/components/schemas/Error'
```

## Schema Patterns

### Pagination
```yaml
# Offset-based
Pagination:
  type: object
  properties:
    total: { type: integer }
    offset: { type: integer }
    limit: { type: integer }

# Cursor-based
CursorPagination:
  type: object
  properties:
    nextCursor: { type: string }
    hasMore: { type: boolean }
```

### Timestamps
```yaml
Timestamps:
  type: object
  properties:
    createdAt:
      type: string
      format: date-time
    updatedAt:
      type: string
      format: date-time
```

### Enum
```yaml
Status:
  type: string
  enum: [pending, active, inactive]
```

## Validation

```bash
# Validate with spectral
npx @stoplight/spectral-cli lint openapi.yaml

# Generate docs
npx @redocly/cli build-docs openapi.yaml
```

## Output Location

Save as:
- `openapi.yaml` or `openapi.json` at project root
- `docs/api/openapi.yaml` for documentation
- Link in README
