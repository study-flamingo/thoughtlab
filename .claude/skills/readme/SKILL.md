---
name: readme
description: Generate or update a README file. Use when a project needs documentation.
---

# README Generation

When invoked, create or update a README.md for the project.

## Process

1. **Analyze project**: Check package files, structure, existing docs
2. **Identify key info**: Purpose, installation, usage, API
3. **Generate sections**: Following standard structure
4. **Add examples**: Real, working code examples

## README Template

```markdown
# Project Name

Brief description of what this project does and why it exists.

## Features

- Feature 1
- Feature 2
- Feature 3

## Installation

```bash
npm install project-name
# or
pip install project-name
```

## Quick Start

```typescript
import { something } from 'project-name';

const result = something.do();
```

## Usage

### Basic Usage

Explain the most common use case with code example.

### Advanced Usage

More complex scenarios.

## API Reference

### `functionName(param)`

Description of what it does.

**Parameters:**
- `param` (string): Description

**Returns:** Description of return value

**Example:**
```typescript
const result = functionName('value');
```

## Configuration

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `option1` | string | `'default'` | What it does |

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `API_KEY` | Yes | API key for service |

## Development

```bash
# Install dependencies
npm install

# Run tests
npm test

# Build
npm run build
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT - see [LICENSE](LICENSE)
```

## Section Guidelines

### Title & Description
- Clear, descriptive project name
- One sentence explaining what it does
- Badges (optional): build status, version, license

### Installation
- All package managers supported
- Prerequisites if any
- Version requirements

### Quick Start
- Minimal working example
- Should work copy-paste
- Most common use case

### API Reference
- All public functions/methods
- Parameters with types
- Return values
- Examples for each

### Configuration
- All options in table format
- Defaults clearly marked
- Environment variables separate

## Project-Type Specific

### Library
Focus on: Installation, API, examples

### CLI Tool
Focus on: Installation, commands, options

### Web App
Focus on: Setup, deployment, environment variables

### MCP Server
Focus on: Installation, Claude Desktop config, available tools

## README Checklist

- [ ] Clear project description
- [ ] Installation instructions work
- [ ] Quick start example runs
- [ ] All public API documented
- [ ] Configuration options listed
- [ ] License specified
- [ ] Contact/contribution info
