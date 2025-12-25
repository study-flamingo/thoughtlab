#!/usr/bin/env python
"""Validation script for tool service and API routes."""

import sys
import traceback
from pathlib import Path

# Add app to path
sys.path.insert(0, str(Path(__file__).parent))

def validate_imports():
    """Validate all imports work."""
    print("[*] Validating imports...")

    try:
        print("  - Importing tool_service...")
        from app.services.tool_service import (
            get_tool_service,
            FindRelatedNodesResponse,
            SummarizeNodeResponse,
            SummarizeNodeWithContextResponse,
            RecalculateConfidenceResponse,
            SummarizeRelationshipResponse,
        )

        print("  - Importing tools routes...")
        from app.api.routes import tools

        print("  - Importing main app...")
        from app.main import app

        print("[+] All imports successful!")
        return True

    except Exception as e:
        print(f"[-] Import failed: {e}")
        traceback.print_exc()
        return False


def validate_service():
    """Validate service initialization."""
    print("\n[*] Validating service...")

    try:
        from app.services.tool_service import get_tool_service

        service = get_tool_service()
        print(f"  - Service initialized: {service.__class__.__name__}")
        print(f"  - AI configured: {service.config.is_configured}")
        print(f"  - LLM model: {service.config.llm_model}")

        print("[+] Service validation successful!")
        return True

    except Exception as e:
        print(f"[-] Service validation failed: {e}")
        traceback.print_exc()
        return False


def validate_routes():
    """Validate API routes are registered."""
    print("\n[*] Validating routes...")

    try:
        from app.main import app

        # Get all routes
        routes = []
        for route in app.routes:
            if hasattr(route, 'path') and hasattr(route, 'methods'):
                routes.append((route.path, list(route.methods)))

        # Check for our new tool routes
        tool_routes = [r for r in routes if '/tools/' in r[0]]

        print(f"  - Found {len(tool_routes)} tool routes:")
        for path, methods in sorted(tool_routes):
            print(f"    {', '.join(methods):8s} {path}")

        # Verify expected routes exist
        expected_paths = [
            '/api/v1/tools/nodes/{node_id}/find-related',
            '/api/v1/tools/nodes/{node_id}/summarize',
            '/api/v1/tools/nodes/{node_id}/summarize-with-context',
            '/api/v1/tools/nodes/{node_id}/recalculate-confidence',
            '/api/v1/tools/relationships/{edge_id}/summarize',
            '/api/v1/tools/health',
            '/api/v1/tools/capabilities',
        ]

        route_paths = [r[0] for r in routes]
        missing = []
        for expected in expected_paths:
            if expected not in route_paths:
                missing.append(expected)

        if missing:
            print(f"\n[!] Missing routes:")
            for path in missing:
                print(f"    {path}")
        else:
            print(f"\n[+] All expected routes present!")

        return len(missing) == 0

    except Exception as e:
        print(f"[-] Route validation failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all validations."""
    print("=" * 60)
    print("Tool Service Validation")
    print("=" * 60)
    print()

    results = []

    results.append(("Imports", validate_imports()))
    results.append(("Service", validate_service()))
    results.append(("Routes", validate_routes()))

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    for name, passed in results:
        status = "[+] PASS" if passed else "[-] FAIL"
        print(f"  {name:20s} {status}")

    all_passed = all(r[1] for r in results)

    print()
    if all_passed:
        print("[+] All validations passed!")
        return 0
    else:
        print("[-] Some validations failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
