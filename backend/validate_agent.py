#!/usr/bin/env python
"""Validation script for LangGraph agent layer.

This script validates:
1. All agent modules import correctly
2. Tools are properly registered
3. Agent can be created
4. Configuration works
"""

import sys
import traceback
from pathlib import Path

# Add app to path
sys.path.insert(0, str(Path(__file__).parent))


def validate_imports():
    """Validate all imports work."""
    print("[*] Validating agent layer imports...")

    try:
        print("  - Importing agent config...")
        from app.agents.config import AgentConfig

        print("  - Importing agent state...")
        from app.agents.state import AgentState

        print("  - Importing agent tools...")
        from app.agents.tools import get_thoughtlab_tools

        print("  - Importing agent...")
        from app.agents.agent import create_thoughtlab_agent, run_agent

        print("  - Importing main agent module...")
        from app.agents import (
            get_thoughtlab_tools,
            create_thoughtlab_agent,
            AgentConfig,
        )

        print("[+] All imports successful!")
        return True

    except Exception as e:
        print(f"[-] Import failed: {e}")
        traceback.print_exc()
        return False


def validate_tools():
    """Validate tools are registered."""
    print("\n[*] Validating tools...")

    try:
        from app.agents.tools import get_thoughtlab_tools

        tools = get_thoughtlab_tools()
        print(f"  - Found {len(tools)} tools:")

        for tool in tools:
            print(f"    - {tool.name}: {tool.description[:60]}...")

        expected_tools = [
            "find_related_nodes",
            "summarize_node",
            "summarize_node_with_context",
            "recalculate_node_confidence",
            "summarize_relationship",
        ]

        tool_names = [t.name for t in tools]
        missing = [name for name in expected_tools if name not in tool_names]

        if missing:
            print(f"\n[!] Missing tools: {missing}")
            return False

        print("\n[+] All expected tools present!")
        return True

    except Exception as e:
        print(f"[-] Tool validation failed: {e}")
        traceback.print_exc()
        return False


def validate_config():
    """Validate configuration."""
    print("\n[*] Validating configuration...")

    try:
        from app.agents.config import AgentConfig

        # Test default config
        config = AgentConfig()
        print(f"  - Default config created")
        print(f"    Model: {config.model_name}")
        print(f"    API URL: {config.api_base_url}")
        print(f"    Temperature: {config.temperature}")
        print(f"    Configured: {config.is_configured}")

        # Test custom config
        custom = AgentConfig(
            model_name="gpt-4o",
            temperature=0.5,
            max_iterations=20,
        )
        print(f"  - Custom config created")
        print(f"    Model: {custom.model_name}")
        print(f"    Temperature: {custom.temperature}")
        print(f"    Max iterations: {custom.max_iterations}")

        # Test safe dump
        safe_dump = config.model_dump_safe()
        print(f"  - Safe dump: {safe_dump}")

        print("\n[+] Configuration validation successful!")
        return True

    except Exception as e:
        print(f"[-] Configuration validation failed: {e}")
        traceback.print_exc()
        return False


def validate_agent_creation():
    """Validate agent can be created."""
    print("\n[*] Validating agent creation...")

    try:
        from app.agents import create_thoughtlab_agent, AgentConfig

        config = AgentConfig()
        agent = create_thoughtlab_agent(config)

        print(f"  - Agent created successfully")
        print(f"  - Agent type: {type(agent).__name__}")

        # Check if agent has required methods
        if not hasattr(agent, 'ainvoke'):
            print(f"[!] Warning: Agent missing 'ainvoke' method")
            return False

        print("\n[+] Agent creation successful!")
        return True

    except Exception as e:
        error_msg = str(e).encode('ascii', errors='replace').decode('ascii')
        print(f"[-] Agent creation failed: {error_msg}")
        traceback.print_exc()
        return False


def main():
    """Run all validations."""
    print("=" * 60)
    print("LangGraph Agent Layer Validation")
    print("=" * 60)
    print()

    results = []

    results.append(("Imports", validate_imports()))
    results.append(("Tools", validate_tools()))
    results.append(("Configuration", validate_config()))
    results.append(("Agent Creation", validate_agent_creation()))

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
        print()
        print("Next steps:")
        print("  1. Set THOUGHTLAB_OPENAI_API_KEY environment variable")
        print("  2. Start the backend server")
        print("  3. Run: python examples/agent_demo.py")
        return 0
    else:
        print("[-] Some validations failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
