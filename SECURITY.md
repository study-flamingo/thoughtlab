# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in this project, please report it responsibly:

1. **Do NOT** create a public GitHub issue for security vulnerabilities
2. Email the maintainers directly (add your email here)
3. Include as much detail as possible about the vulnerability

We will respond within 48 hours and work with you to understand and address the issue.

## Security Measures

This project implements several security measures:

### Automated Scanning
- **Gitleaks**: Scans for secrets and sensitive data on every push/PR
- **Dependency Audits**: npm and pip dependencies are scanned for known vulnerabilities

### Development Practices
- Environment variables are used for all sensitive configuration
- `.env` files are gitignored and never committed
- Pre-commit hooks can be installed to catch secrets locally

### For Contributors

Before contributing, please:

1. Install git hooks: `./scripts/install-hooks.sh`
2. Never commit `.env` files
3. Never hardcode API keys, tokens, or passwords
4. Use `.env.example` files to document required environment variables

## Known Development Credentials

The following credentials are **intentionally** in the codebase for local development only:

| Credential | Value | Purpose |
|------------|-------|---------|
| Neo4j Password | `research_graph_password` | Local Docker development |
| SECRET_KEY placeholder | `change-me-in-production` | Example only - auto-generated in setup |

⚠️ **These are for local development only. Production deployments must use unique, secure credentials.**

