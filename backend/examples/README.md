# ThoughtLab Agent Examples

This directory contains example scripts demonstrating how to use the ThoughtLab LangGraph agent.

## Prerequisites

1. **Backend running:**
   ```bash
   # From project root
   ./start.sh
   ```

2. **OpenAI API key set:**
   ```bash
   export THOUGHTLAB_OPENAI_API_KEY="sk-..."
   ```

3. **Dependencies installed:**
   ```bash
   cd backend
   uv sync --all-extras
   ```

## Examples

### agent_demo.py

Demonstrates three usage modes:

#### 1. Basic Usage
```bash
python examples/agent_demo.py --mode basic
```

Shows:
- Finding related nodes
- Summarizing nodes
- Context-aware analysis

#### 2. Interactive Mode
```bash
python examples/agent_demo.py --mode interactive
```

Chat with the agent:
```
You: Find nodes related to obs-123
Agent: [Uses tools and responds with findings]

You: Summarize hyp-456 with context
Agent: [Provides comprehensive summary]

You: quit
```

#### 3. Multi-Step Reasoning
```bash
python examples/agent_demo.py --mode multi-step
```

Demonstrates complex queries requiring multiple tool calls.

## Quick Start

```bash
# 1. Validate everything works
cd backend
python validate_agent.py

# 2. Run interactive demo
python examples/agent_demo.py --mode interactive

# 3. Try some queries:
#    - "Find nodes related to [node-id]"
#    - "Summarize [node-id] with context"
#    - "How confident should we be in [node-id]?"
#    - "Explain the relationship [edge-id]"
```

## Creating Your Own Agent

```python
from app.agents import create_thoughtlab_agent, run_agent

# Create agent
agent = create_thoughtlab_agent()

# Use agent
response = await run_agent(
    agent,
    "Your query here"
)

print(response)
```

## See Also

- [LangGraph Integration Guide](../../docs/PHASE_8_LANGGRAPH_INTEGRATION.md)
- [Backend API Specification](../../docs/PHASE_7_API_SPEC.md)
- [Implementation Summary](../../docs/IMPLEMENTATION_SUMMARY.md)
