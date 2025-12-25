# Phase 8: LangGraph Integration Guide

This document describes the LangGraph agent layer for ThoughtLab, which provides intelligent, tool-using agents that interact with the backend API.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     LangGraph Agent Layer                    │
│                                                              │
│  ┌────────────────┐      ┌──────────────┐                  │
│  │  Agent Config  │      │ System Prompt │                  │
│  └────────────────┘      └──────────────┘                  │
│           │                      │                          │
│           └──────────┬───────────┘                          │
│                      ▼                                      │
│            ┌──────────────────┐                            │
│            │  ReAct Agent     │                            │
│            │  (LangGraph)     │                            │
│            └──────────────────┘                            │
│                      │                                      │
│                      ▼                                      │
│            ┌──────────────────┐                            │
│            │   Tool Selection  │                            │
│            │   & Execution     │                            │
│            └──────────────────┘                            │
│                      │                                      │
└──────────────────────│──────────────────────────────────────┘
                       │
                       ▼ HTTP API Calls
┌─────────────────────────────────────────────────────────────┐
│                     Backend API Layer                        │
│                                                              │
│  GET  /api/v1/tools/health                                  │
│  POST /api/v1/tools/nodes/{id}/find-related                 │
│  POST /api/v1/tools/nodes/{id}/summarize                    │
│  POST /api/v1/tools/nodes/{id}/summarize-with-context       │
│  POST /api/v1/tools/nodes/{id}/recalculate-confidence       │
│  POST /api/v1/tools/relationships/{id}/summarize            │
│                                                              │
└─────────────────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                  Backend Services Layer                      │
│                                                              │
│  • ToolService     - Core LLM-powered operations            │
│  • GraphService    - Neo4j database operations              │
│  • AIWorkflow      - Automatic processing                   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Principles

### 1. Complete Separation

The LangGraph layer is **completely separate** from backend logic:
- **No direct database access** - All operations via HTTP API
- **No shared services** - Agent layer doesn't import backend services
- **Pure HTTP client** - Tools are thin wrappers around API calls
- **Independent deployment** - Can run agent separately from backend

### 2. Tool-First Design

Agents interact with the graph via **tools**:
- Each tool maps to one API endpoint
- Tools are async and return formatted strings
- Error handling built into every tool
- Clear documentation for agent reasoning

### 3. ReAct Pattern

Uses LangGraph's **ReAct (Reasoning + Acting)** pattern:
- Agent reasons about which tools to use
- Executes tools to gather information
- Reflects on results before responding
- Can chain multiple tool calls

---

## Module Structure

```
backend/app/agents/
├── __init__.py          # Public exports
├── config.py            # AgentConfig class
├── state.py             # AgentState TypedDict
├── tools.py             # LangGraph tools (HTTP clients)
└── agent.py             # Agent creation & execution
```

---

## Available Tools

### Node Analysis Tools

#### 1. `find_related_nodes`
Find semantically similar nodes using vector embeddings.

**When to use:**
- Discover connections between research
- Find supporting/contradicting evidence
- Identify patterns in the graph

**Parameters:**
- `node_id` (str): Node to find similar nodes for
- `limit` (int): Maximum results (default 10)
- `min_similarity` (float): Minimum score 0-1 (default 0.5)
- `node_types` (list): Optional type filter
- `auto_link` (bool): Auto-create relationships (default False)

#### 2. `summarize_node`
Generate AI summary of node content.

**When to use:**
- Quick understanding of a node
- Generate concise descriptions
- Extract key points

**Parameters:**
- `node_id` (str): Node to summarize
- `max_length` (int): Max characters (default 200)
- `style` ("concise"|"detailed"|"bullet_points")

#### 3. `summarize_node_with_context`
Context-aware summary including relationships.

**When to use:**
- Understand node in broader context
- See supporting/contradicting evidence
- Analyze state of knowledge

**Parameters:**
- `node_id` (str): Node to summarize
- `depth` (int): Relationship hops (default 1)
- `relationship_types` (list): Optional filter
- `max_length` (int): Max characters (default 300)

#### 4. `recalculate_node_confidence`
Re-analyze confidence based on current graph context.

**When to use:**
- After adding new evidence
- Validate claim reliability
- Track confidence changes

**Parameters:**
- `node_id` (str): Node to recalculate
- `factor_in_relationships` (bool): Consider connected nodes (default True)

### Relationship Analysis Tools

#### 5. `summarize_relationship`
Explain connection between nodes in plain language.

**When to use:**
- Understand why nodes are connected
- Get natural language explanation
- Assess connection strength

**Parameters:**
- `edge_id` (str): Relationship to summarize
- `include_evidence` (bool): Show supporting evidence (default True)

---

## Usage Examples

### Basic Usage

```python
from app.agents import create_thoughtlab_agent, run_agent, AgentConfig

# Create agent with default config
agent = create_thoughtlab_agent()

# Ask a question
response = await run_agent(
    agent,
    "Find nodes related to observation obs-123 and summarize the most relevant one"
)
print(response)
```

### Custom Configuration

```python
from app.agents import AgentConfig, create_thoughtlab_agent

# Configure agent
config = AgentConfig(
    model_name="gpt-4o",  # Use GPT-4
    temperature=0.0,       # Deterministic
    max_iterations=20,     # More reasoning steps
    verbose=True,          # Show reasoning
    api_base_url="http://production:8000/api/v1"  # Custom backend
)

agent = create_thoughtlab_agent(config)
```

### Multi-Step Reasoning

```python
# Complex query requiring multiple tools
response = await run_agent(
    agent,
    """Analyze hypothesis hyp-123:
    1. First summarize it with full context
    2. Find related nodes
    3. Assess confidence level

    Give me a comprehensive analysis."""
)
```

### Interactive Chat

```python
agent = create_thoughtlab_agent()

while True:
    user_input = input("You: ")
    response = await run_agent(agent, user_input)
    print(f"Agent: {response}")
```

---

## Configuration

### Environment Variables

```bash
# Required
export THOUGHTLAB_OPENAI_API_KEY="sk-..."

# Optional
export THOUGHTLAB_API_BASE_URL="http://localhost:8000/api/v1"
```

### AgentConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name` | str | "gpt-4o-mini" | OpenAI model |
| `temperature` | float | 0.1 | Response randomness (0-2) |
| `max_tokens` | int? | None | Max response tokens |
| `api_base_url` | str | "http://localhost:8000/api/v1" | Backend API URL |
| `api_timeout` | float | 30.0 | Request timeout |
| `max_iterations` | int | 10 | Max reasoning steps |
| `verbose` | bool | False | Show reasoning |
| `openai_api_key` | str? | None | OpenAI API key |

---

## Testing & Validation

### Validate Installation

```bash
cd backend
python validate_agent.py
```

Expected output:
```
[+] All validations passed!

Next steps:
  1. Set THOUGHTLAB_OPENAI_API_KEY environment variable
  2. Start the backend server
  3. Run: python examples/agent_demo.py
```

### Run Demo

```bash
# Basic demo
python examples/agent_demo.py --mode basic

# Interactive mode
python examples/agent_demo.py --mode interactive

# Multi-step reasoning
python examples/agent_demo.py --mode multi-step
```

---

## Agent Behavior

### System Prompt

The agent is guided by a comprehensive system prompt that instructs it to:

1. **Be proactive** - Use tools to provide comprehensive answers
2. **Provide context** - Explain results and relevance
3. **Chain operations** - Use multiple tools when needed
4. **Be accurate** - Use exact node/edge IDs
5. **Explain results** - Interpret findings clearly
6. **Handle errors gracefully** - Suggest alternatives

### Example Interactions

**User:** "What's related to observation obs-123?"

**Agent:**
1. Uses `find_related_nodes` to discover connections
2. Uses `summarize_node` on top results
3. Returns formatted analysis with explanations

**User:** "Give me a comprehensive view of hypothesis hyp-456"

**Agent:**
1. Uses `summarize_node_with_context` for full picture
2. Shows supporting and contradicting evidence
3. Provides synthesis of overall state

**User:** "How confident should we be in obs-789?"

**Agent:**
1. Uses `recalculate_node_confidence` to analyze
2. Shows factors affecting confidence
3. Explains reasoning

---

## Integration Patterns

### 1. Standalone Agent

```python
# Run agent independently
from app.agents import create_thoughtlab_agent, run_agent

agent = create_thoughtlab_agent()
result = await run_agent(agent, "Analyze node obs-123")
```

### 2. API Endpoint

```python
# Expose agent via FastAPI endpoint
from fastapi import APIRouter
from app.agents import create_thoughtlab_agent, run_agent

router = APIRouter()
agent = create_thoughtlab_agent()

@router.post("/agent/chat")
async def chat(message: str):
    response = await run_agent(agent, message)
    return {"response": response}
```

### 3. MCP Server

```python
# Expose agent via Model Context Protocol
from app.agents import create_thoughtlab_agent

agent = create_thoughtlab_agent()

# Register as MCP tool
@mcp.tool()
async def analyze_knowledge_graph(query: str) -> str:
    """Analyze the knowledge graph using AI."""
    return await run_agent(agent, query)
```

### 4. Background Jobs

```python
# Use agent in background tasks
from arq import create_pool
from app.agents import create_thoughtlab_agent, run_agent

async def process_node_with_agent(ctx, node_id: str):
    agent = create_thoughtlab_agent()
    analysis = await run_agent(
        agent,
        f"Analyze node {node_id} and suggest improvements"
    )
    # Store results...
```

---

## Troubleshooting

### Agent Creation Fails

**Error:** `THOUGHTLAB_OPENAI_API_KEY not set`

**Solution:**
```bash
export THOUGHTLAB_OPENAI_API_KEY="sk-..."
```

### Tool Calls Fail

**Error:** `Error calling API: Connection refused`

**Solution:** Ensure backend server is running:
```bash
./start.sh  # or docker-compose up
```

### Slow Responses

**Issue:** Agent taking too long to respond

**Solutions:**
1. Use faster model: `model_name="gpt-4o-mini"`
2. Reduce temperature: `temperature=0.0`
3. Limit iterations: `max_iterations=5`
4. Increase API timeout: `api_timeout=60.0`

---

## Best Practices

### 1. Configuration

- Use `gpt-4o-mini` for speed and cost
- Use `gpt-4o` for complex reasoning
- Set `temperature=0.0` for deterministic results
- Set `verbose=True` during development

### 2. Error Handling

```python
try:
    response = await run_agent(agent, user_message)
except Exception as e:
    logger.error(f"Agent error: {e}")
    response = "I encountered an error. Please try again."
```

### 3. Rate Limiting

```python
from ratelimit import limits, sleep_and_retry

@sleep_and_retry
@limits(calls=10, period=60)  # 10 calls per minute
async def safe_run_agent(agent, message):
    return await run_agent(agent, message)
```

### 4. Caching

```python
from functools import lru_cache

@lru_cache(maxsize=100)
def get_cached_agent():
    """Reuse agent instance."""
    return create_thoughtlab_agent()
```

---

## Performance

### Typical Response Times

| Operation | Time |  Model |
|-----------|------|--------|
| Single tool call | 2-4s | gpt-4o-mini |
| Multi-step (2-3 tools) | 6-10s | gpt-4o-mini |
| Complex analysis (4+ tools) | 12-20s | gpt-4o-mini |

### Optimization Tips

1. **Use streaming** for long responses
2. **Cache agent instances** instead of recreating
3. **Set max_iterations** to prevent runaway reasoning
4. **Use faster model** when precision isn't critical
5. **Implement timeouts** for production use

---

## Future Enhancements

### Coming Soon

- [ ] Streaming responses for real-time updates
- [ ] Agent memory/conversation history
- [ ] Multi-agent collaboration
- [ ] Custom tool plugins
- [ ] Agent performance monitoring
- [ ] Automatic tool retry logic

### Planned Tools

- [ ] Web search for evidence
- [ ] Node merging with confirmation
- [ ] Bulk operations (analyze multiple nodes)
- [ ] Graph visualization generation
- [ ] Export/report generation

---

## See Also

- [Phase 7 API Specification](./PHASE_7_API_SPEC.md) - Backend API endpoints
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Tool Architecture](./TOOL_ARCHITECTURE.md) - Overall design
