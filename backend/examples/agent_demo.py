#!/usr/bin/env python
"""Demo script for ThoughtLab LangGraph agent.

This script demonstrates how to use the ThoughtLab agent to interact
with the knowledge graph via natural language.

Prerequisites:
1. Backend server running (./start.sh or docker-compose up)
2. THOUGHTLAB_OPENAI_API_KEY environment variable set
3. Some nodes in the knowledge graph to query

Usage:
    cd backend
    python examples/agent_demo.py
"""

import asyncio
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.agents import create_thoughtlab_agent, run_agent, AgentConfig


async def demo_basic_usage():
    """Demonstrate basic agent usage."""
    print("=" * 60)
    print("ThoughtLab Agent - Basic Usage Demo")
    print("=" * 60)
    print()

    # Create agent with default configuration
    print("Creating agent...")
    config = AgentConfig(verbose=True)

    if not config.is_configured:
        print("ERROR: THOUGHTLAB_OPENAI_API_KEY environment variable not set!")
        print("Please set it to your OpenAI API key.")
        return

    agent = create_thoughtlab_agent(config)
    print(f"Agent created successfully!")
    print(f"Model: {config.model_name}")
    print(f"API URL: {config.api_base_url}")
    print()

    # Example 1: Find related nodes
    print("-" * 60)
    print("Example 1: Finding Related Nodes")
    print("-" * 60)
    print()

    # Note: Replace 'obs-123' with an actual node ID from your graph
    response = await run_agent(
        agent,
        "Find nodes related to observation obs-123 and tell me what you found"
    )
    print("Agent Response:")
    print(response)
    print()

    # Example 2: Summarize a node
    print("-" * 60)
    print("Example 2: Summarizing a Node")
    print("-" * 60)
    print()

    response = await run_agent(
        agent,
        "Give me a detailed summary of hypothesis hyp-456"
    )
    print("Agent Response:")
    print(response)
    print()

    # Example 3: Context-aware summary
    print("-" * 60)
    print("Example 3: Context-Aware Summary")
    print("-" * 60)
    print()

    response = await run_agent(
        agent,
        "Summarize observation obs-789 including all its relationships. "
        "Show me what supports it and what contradicts it."
    )
    print("Agent Response:")
    print(response)
    print()


async def demo_interactive():
    """Interactive demo - chat with the agent."""
    print("=" * 60)
    print("ThoughtLab Agent - Interactive Mode")
    print("=" * 60)
    print()
    print("Type your questions and the agent will use tools to answer.")
    print("Type 'quit' or 'exit' to stop.")
    print()

    config = AgentConfig()

    if not config.is_configured:
        print("ERROR: THOUGHTLAB_OPENAI_API_KEY environment variable not set!")
        return

    agent = create_thoughtlab_agent(config)
    print("Agent ready! Ask me anything about your knowledge graph.")
    print()

    while True:
        try:
            user_input = input("You: ").strip()

            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

            if not user_input:
                continue

            print()
            print("Agent: ", end="", flush=True)

            response = await run_agent(agent, user_input)
            print(response)
            print()

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            print()


async def demo_multi_step():
    """Demonstrate multi-step reasoning."""
    print("=" * 60)
    print("ThoughtLab Agent - Multi-Step Reasoning Demo")
    print("=" * 60)
    print()

    config = AgentConfig(temperature=0.0)  # Deterministic

    if not config.is_configured:
        print("ERROR: THOUGHTLAB_OPENAI_API_KEY environment variable not set!")
        return

    agent = create_thoughtlab_agent(config)

    # Complex query that requires multiple tool calls
    print("Complex Query: Analyze the state of a hypothesis")
    print("-" * 60)
    print()

    response = await run_agent(
        agent,
        """Analyze hypothesis hyp-123:
        1. First, give me a summary with full context
        2. Then, find related nodes to see what else connects to this topic
        3. Finally, assess the confidence we should have in this hypothesis

        Give me a comprehensive analysis."""
    )

    print("Agent Response:")
    print(response)
    print()


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="ThoughtLab Agent Demo")
    parser.add_argument(
        "--mode",
        choices=["basic", "interactive", "multi-step"],
        default="basic",
        help="Demo mode to run"
    )

    args = parser.parse_args()

    if args.mode == "basic":
        asyncio.run(demo_basic_usage())
    elif args.mode == "interactive":
        asyncio.run(demo_interactive())
    elif args.mode == "multi-step":
        asyncio.run(demo_multi_step())


if __name__ == "__main__":
    main()
