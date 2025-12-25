# ThoughtLab Agent Quick Start

Get started with the ThoughtLab LangGraph agent in 5 minutes.

## Prerequisites

- Docker and Docker Compose installed
- OpenAI API key
- Python 3.13+
- uv (Python package manager)

## Step 1: Environment Setup

```bash
# Set your OpenAI API key
export THOUGHTLAB_OPENAI_API_KEY="sk-..."

# Optionally customize API URL (defaults to localhost:8000)
# export THOUGHTLAB_API_BASE_URL="http://localhost:8000/api/v1"
```

## Step 2: Start Services

```bash
# Start all services (Neo4j, Redis, Backend, Frontend)
./start.sh
```

This will:
- Start Neo4j database (port 7474, 7687)
- Start Redis (port 6379)
- Start Backend API (port 8000)
- Start Frontend (port 5173)

## Step 3: Validate Installation

```bash
# Validate backend API
cd backend
python validate_tools.py

# Validate agent layer
python validate_agent.py
```

Expected output:
```
[+] All validations passed!
```

## Step 4: Run the Agent

### Interactive Mode

```bash
python examples/agent_demo.py --mode interactive
```

```
You: Find nodes related to obs-123
Agent: [Searches and responds...]

You: Summarize hyp-456 with full context
Agent: [Provides comprehensive analysis...]

You: quit
```

### Programmatic Usage

```python
from app.agents import create_thoughtlab_agent, run_agent

# Create agent
agent = create_thoughtlab_agent()

# Ask a question
response = await run_agent(
    agent,
    "Analyze observation obs-123"
)

print(response)
```

## Example Queries

Try these with your knowledge graph:

1. **Find Related Nodes**
   ```
   Find nodes related to observation obs-123
   ```

2. **Comprehensive Analysis**
   ```
   Give me a comprehensive analysis of hypothesis hyp-456 including all
   supporting and contradicting evidence
   ```

3. **Confidence Assessment**
   ```
   How confident should we be in observation obs-789? Please recalculate
   based on current evidence.
   ```

4. **Relationship Explanation**
   ```
   Why are observation obs-123 and hypothesis hyp-456 connected?
   ```

5. **Multi-Step Research**
   ```
   Find everything related to concept con-abc, summarize the most
   important findings, and assess overall confidence in this area
   ```

## What the Agent Can Do

The agent has access to 5 tools:

1. **find_related_nodes** - Discover semantic connections
2. **summarize_node** - Generate AI summaries
3. **summarize_node_with_context** - Full context analysis
4. **recalculate_node_confidence** - Assess reliability
5. **summarize_relationship** - Explain connections

The agent automatically:
- Selects which tools to use
- Chains multiple tool calls
- Interprets results
- Provides comprehensive answers

## API Access

You can also use the backend API directly:

```bash
# Health check
curl http://localhost:8000/api/v1/tools/health

# List capabilities
curl http://localhost:8000/api/v1/tools/capabilities

# Find related nodes
curl -X POST http://localhost:8000/api/v1/tools/nodes/obs-123/find-related \
  -H "Content-Type: application/json" \
  -d '{"limit": 10, "min_similarity": 0.5}'
```

## Configuration

Create custom agent configuration:

```python
from app.agents import AgentConfig, create_thoughtlab_agent

config = AgentConfig(
    model_name="gpt-4o",      # Use GPT-4 for better reasoning
    temperature=0.0,           # Deterministic responses
    max_iterations=20,         # More reasoning steps
    verbose=True,              # Show tool calls
)

agent = create_thoughtlab_agent(config)
```

## Troubleshooting

### Agent not responding

**Check:** Is OpenAI API key set?
```bash
echo $THOUGHTLAB_OPENAI_API_KEY
```

### Tool calls failing

**Check:** Is backend running?
```bash
curl http://localhost:8000/api/v1/tools/health
```

Expected: `{"status": "healthy", ...}`

### Slow responses

**Solutions:**
- Use `gpt-4o-mini` instead of `gpt-4o`
- Set `temperature=0.0` for faster responses
- Reduce `max_iterations` to limit reasoning steps

## Next Steps

1. **Read the docs:**
   - [LangGraph Integration Guide](docs/PHASE_8_LANGGRAPH_INTEGRATION.md)
   - [Backend API Spec](docs/PHASE_7_API_SPEC.md)
   - [Implementation Summary](docs/IMPLEMENTATION_SUMMARY.md)

2. **Explore examples:**
   - See `backend/examples/` for more demos
   - Try different query patterns
   - Experiment with configurations

3. **Build your own:**
   - Create custom tools
   - Add new API endpoints
   - Customize agent behavior

## Support

- Documentation: `docs/` directory
- Examples: `backend/examples/`
- Issues: GitHub Issues

---

**Ready to explore your knowledge graph with AI!** 🚀
