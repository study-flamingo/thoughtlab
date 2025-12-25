# ThoughtLab AI Integration - Quick Reference

Quick reference for all three AI integration layers.

---

## Architecture Overview

```
Claude Desktop → MCP Server → Backend API
LangGraph Agent →  ────────→ Backend API
Direct Access   →  ────────→ Backend API
```

All layers use the same backend. Complete separation maintained.

---

## 🚀 Quick Start

### 1. Start Services

```bash
# Start all services
./start.sh

# Or manually:
docker-compose up -d
cd backend && uv sync --all-extras
.venv/Scripts/python -m uvicorn app.main:app --reload
```

### 2. Set API Key

```bash
export THOUGHTLAB_OPENAI_API_KEY="sk-..."
```

### 3. Validate

```bash
cd backend
python validate_tools.py    # Backend API
python validate_agent.py    # LangGraph
python validate_mcp.py      # MCP Server
```

---

## 🔧 Available Tools

All three layers expose the same operations:

1. **find_related_nodes** - Semantic similarity search
2. **summarize_node** - AI-powered summaries
3. **summarize_node_with_context** - Context-aware analysis
4. **recalculate_node_confidence** - Reliability assessment
5. **summarize_relationship** - Explain connections
6. **check_api_health** - Backend status (MCP only)

---

## 📦 Usage by Layer

### Backend API (Direct)

```bash
# Health check
curl http://localhost:8000/api/v1/tools/health

# Summarize node
curl -X POST http://localhost:8000/api/v1/tools/nodes/obs-123/summarize \
  -H "Content-Type: application/json" \
  -d '{"style": "concise"}'
```

### LangGraph Agent

```python
from app.agents import create_thoughtlab_agent, run_agent

agent = create_thoughtlab_agent()
response = await run_agent(agent, "Analyze obs-123")
```

```bash
# Interactive demo
python examples/agent_demo.py --mode interactive
```

### MCP Server (Claude Desktop)

```bash
# Install for Claude Desktop
fastmcp install claude-desktop mcp_server.py \
  --server-name "ThoughtLab" \
  --env THOUGHTLAB_OPENAI_API_KEY="sk-..."
```

Then in Claude Desktop:
```
You: Find nodes related to obs-123
```

---

## 📝 Configuration

### Environment Variables

```bash
# Required (all layers)
THOUGHTLAB_OPENAI_API_KEY="sk-..."

# Optional
THOUGHTLAB_API_BASE_URL="http://localhost:8000/api/v1"
THOUGHTLAB_API_TIMEOUT="30.0"
```

### Files to Edit

**LangGraph:**
- `backend/app/agents/config.py` - Agent configuration

**MCP:**
- `claude_desktop_config.json` - Claude Desktop config
- See `claude_desktop_config.example.json`

---

## 🧪 Testing

```bash
cd backend

# Validate all layers
python validate_tools.py && \
python validate_agent.py && \
python validate_mcp.py

# Run demos
python examples/agent_demo.py --mode basic
python examples/agent_demo.py --mode interactive
python examples/agent_demo.py --mode multi-step

# Start MCP server
python mcp_server.py
```

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [PHASE_7_API_SPEC.md](docs/PHASE_7_API_SPEC.md) | Backend API reference |
| [PHASE_8_LANGGRAPH_INTEGRATION.md](docs/PHASE_8_LANGGRAPH_INTEGRATION.md) | LangGraph guide |
| [MCP_SERVER_GUIDE.md](docs/MCP_SERVER_GUIDE.md) | MCP server setup |
| [COMPLETE_IMPLEMENTATION_SUMMARY.md](docs/COMPLETE_IMPLEMENTATION_SUMMARY.md) | Full summary |
| [QUICKSTART_AGENT.md](docs/QUICKSTART_AGENT.md) | 5-min quick start |

---

## 🔍 Troubleshooting

### Backend API not responding

```bash
# Check if running
curl http://localhost:8000/health

# Restart
./start.sh
```

### Agent errors

```bash
# Check environment
echo $THOUGHTLAB_OPENAI_API_KEY

# Validate
python validate_agent.py
```

### MCP not working in Claude

1. Check Claude Desktop config file location:
   - macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
   - Windows: `%APPDATA%\Claude\claude_desktop_config.json`

2. Validate JSON syntax

3. Restart Claude Desktop

4. Check logs: Help → View Logs

---

## 📊 Performance Reference

| Operation | Time |
|-----------|------|
| Health check | ~100ms |
| Find related | ~500ms |
| Summarize | ~2-3s |
| With context | ~3-5s |
| Recalculate | ~3-4s |

*Times with gpt-4o-mini*

---

## 🎯 Example Queries

### For LangGraph Agent or Claude Desktop:

1. **Discovery**
   ```
   Find nodes related to observation obs-123
   ```

2. **Analysis**
   ```
   Give me a comprehensive analysis of hypothesis hyp-456 with all supporting and contradicting evidence
   ```

3. **Assessment**
   ```
   How confident should we be in observation obs-789?
   ```

4. **Explanation**
   ```
   Why are observation obs-123 and hypothesis hyp-456 connected?
   ```

5. **Health**
   ```
   Check if the ThoughtLab backend is working
   ```

---

## 🆘 Common Issues

**"Module not found"**
```bash
cd backend && uv sync --all-extras
```

**"API key not set"**
```bash
export THOUGHTLAB_OPENAI_API_KEY="sk-..."
```

**"Connection refused"**
```bash
./start.sh  # Make sure backend is running
```

**"Tools not found in Claude"**
- Restart Claude Desktop
- Check config file syntax
- Verify environment variables in config

---

## 📁 File Locations

```
backend/
├── app/
│   ├── agents/          # LangGraph agents
│   ├── api/routes/      # API endpoints
│   ├── services/        # Backend services
│   └── mcp/            # MCP server
├── examples/            # Demo scripts
├── validate_*.py        # Validation scripts
└── mcp_server.py       # MCP entry point

docs/
├── PHASE_7_API_SPEC.md
├── PHASE_8_LANGGRAPH_INTEGRATION.md
├── MCP_SERVER_GUIDE.md
└── COMPLETE_IMPLEMENTATION_SUMMARY.md
```

---

## ✅ Validation Checklist

- [ ] Backend running on port 8000
- [ ] Neo4j running on port 7687
- [ ] Redis running on port 6379
- [ ] THOUGHTLAB_OPENAI_API_KEY set
- [ ] `validate_tools.py` passes
- [ ] `validate_agent.py` passes
- [ ] `validate_mcp.py` passes
- [ ] Can query backend API with curl
- [ ] Can run agent demo
- [ ] MCP server configured in Claude Desktop (if using)

---

## 🔗 Quick Links

- Backend API: http://localhost:8000/docs
- Health endpoint: http://localhost:8000/api/v1/tools/health
- Neo4j Browser: http://localhost:7474
- FastMCP Docs: https://gofastmcp.com
- LangGraph Docs: https://langchain-ai.github.io/langgraph/

---

**Need help?** See full documentation in `docs/` directory or run validation scripts for diagnostic info.
