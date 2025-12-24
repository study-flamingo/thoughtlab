# Tool Architecture Specification

This document defines the unified tool layer that enables LLM agents, API endpoints, and frontend components to perform operations on the knowledge graph.

## Design Principles

1. **Single Source of Truth**: All graph operations go through the tool layer
2. **Safety First**: Destructive operations require user confirmation
3. **LLM-Friendly**: Tools return structured data with clear success/error indicators
4. **Composable**: Tools can be chained together for complex operations
5. **Observable**: All tool calls logged in Activity Feed

---

## Tool Categories

### 1. Node Analysis Tools

#### `find_related_nodes`
**Purpose**: Find semantically similar nodes using vector embeddings

**Input**:
```python
{
    "node_id": str,              # Node to analyze
    "limit": int = 10,           # Max results
    "min_similarity": float = 0.5,  # Threshold
    "auto_link": bool = False    # Create relationships automatically
}
```

**Output**:
```python
{
    "success": bool,
    "node_id": str,
    "related_nodes": [
        {
            "id": str,
            "type": str,
            "content": str,
            "similarity_score": float,
            "suggested_relationship": str,  # RELATES_TO, SUPPORTS, etc.
            "reasoning": str
        }
    ],
    "links_created": int,  # If auto_link=True
    "message": str
}
```

---

#### `recalculate_node_confidence`
**Purpose**: Re-analyze node confidence based on current graph context

**Input**:
```python
{
    "node_id": str,
    "factor_in_relationships": bool = True  # Consider connected nodes
}
```

**Output**:
```python
{
    "success": bool,
    "node_id": str,
    "old_confidence": float,
    "new_confidence": float,
    "reasoning": str,
    "factors": [
        {"factor": str, "impact": str}  # e.g., "Supporting evidence", "+0.1"
    ]
}
```

---

#### `summarize_node`
**Purpose**: Generate LLM summary of a node

**Input**:
```python
{
    "node_id": str,
    "include_relationships": bool = False,
    "max_length": int = 200  # Characters
}
```

**Output**:
```python
{
    "success": bool,
    "node_id": str,
    "summary": str,
    "key_points": [str],  # Bullet points
    "related_count": int  # If include_relationships=True
}
```

---

#### `summarize_node_with_context`
**Purpose**: Summarize node including its relationships and neighbors

**Input**:
```python
{
    "node_id": str,
    "depth": int = 1,  # How many hops to include
    "relationship_types": List[str] = None  # Filter specific types
}
```

**Output**:
```python
{
    "success": bool,
    "node_id": str,
    "summary": str,
    "context": {
        "supports": [str],      # Summaries of supporting nodes
        "contradicts": [str],   # Summaries of contradicting nodes
        "related": [str]        # Summaries of related nodes
    },
    "synthesis": str  # Overall assessment
}
```

---

#### `search_web_for_evidence`
**Purpose**: Search web for supporting/contradicting evidence

**Input**:
```python
{
    "node_id": str,
    "query_override": str = None,  # Custom search query
    "evidence_type": str = "all",  # "supporting", "contradicting", "all"
    "max_results": int = 5
}
```

**Output**:
```python
{
    "success": bool,
    "node_id": str,
    "results": [
        {
            "url": str,
            "title": str,
            "snippet": str,
            "relevance_score": float,
            "evidence_type": str,  # "supporting", "contradicting", "neutral"
            "reasoning": str
        }
    ],
    "sources_created": int,  # Auto-created Source nodes (optional)
    "message": str
}
```

---

#### `reclassify_node`
**Purpose**: Change node type (e.g., Observation → Hypothesis)

**Input**:
```python
{
    "node_id": str,
    "new_type": str,  # "Observation", "Hypothesis", "Source", etc.
    "preserve_relationships": bool = True
}
```

**Output**:
```python
{
    "success": bool,
    "node_id": str,
    "old_type": str,
    "new_type": str,
    "message": str,
    "warning": str = None  # If data loss possible
}
```

---

### 2. Edge Analysis Tools

#### `recalculate_edge_confidence`
**Purpose**: Re-evaluate relationship confidence

**Input**:
```python
{
    "edge_id": str,
    "consider_graph_structure": bool = True  # Use broader context
}
```

**Output**:
```python
{
    "success": bool,
    "edge_id": str,
    "old_confidence": float,
    "new_confidence": float,
    "reasoning": str
}
```

---

#### `reclassify_relationship`
**Purpose**: Change relationship type

**Input**:
```python
{
    "edge_id": str,
    "new_type": str = None,  # If None, LLM suggests best type
    "preserve_notes": bool = True
}
```

**Output**:
```python
{
    "success": bool,
    "edge_id": str,
    "old_type": str,
    "new_type": str,
    "confidence": float,
    "reasoning": str
}
```

---

#### `summarize_relationship`
**Purpose**: Explain the connection between two nodes

**Input**:
```python
{
    "edge_id": str,
    "include_evidence": bool = True  # Show supporting data
}
```

**Output**:
```python
{
    "success": bool,
    "edge_id": str,
    "from_node": str,
    "to_node": str,
    "relationship_type": str,
    "summary": str,
    "evidence": [str],  # If include_evidence=True
    "strength_assessment": str  # "strong", "moderate", "weak"
}
```

---

#### `merge_nodes`
**Purpose**: Combine duplicate or very similar nodes

**Input**:
```python
{
    "primary_node_id": str,   # Node to keep
    "secondary_node_id": str, # Node to merge in
    "merge_strategy": str = "combine"  # "combine", "prefer_primary", "prefer_secondary"
}
```

**Output**:
```python
{
    "success": bool,
    "primary_node_id": str,
    "secondary_node_id": str,
    "merged_content": str,
    "relationships_transferred": int,
    "message": str,
    "requires_confirmation": bool = True  # ALWAYS true for destructive ops
}
```

---

### 3. CRUD Tools (Standard Operations)

#### `create_node`
```python
{
    "type": str,  # "Observation", "Hypothesis", etc.
    "content": dict,  # Type-specific fields
    "auto_analyze": bool = True  # Run find_related_nodes after creation
}
```

#### `get_node`
```python
{"node_id": str}
```

#### `update_node`
```python
{
    "node_id": str,
    "updates": dict,
    "re_analyze": bool = False  # Re-run AI analysis
}
```

#### `delete_node`
```python
{
    "node_id": str,
    "cascade": bool = False  # Delete relationships too
}
→ ALWAYS requires user confirmation
```

#### `create_relationship`
```python
{
    "from_id": str,
    "to_id": str,
    "relationship_type": str,
    "confidence": float = 1.0,
    "notes": str = None
}
```

#### `delete_relationship`
```python
{"edge_id": str}
→ ALWAYS requires user confirmation
```

---

## User Confirmation Pattern

For destructive operations (`delete_node`, `delete_relationship`, `merge_nodes`), the tool:

1. **Returns pending state**:
```python
{
    "success": False,
    "requires_confirmation": True,
    "operation": "delete_node",
    "details": {
        "node_id": "obs-123",
        "node_type": "Observation",
        "content_preview": "This observation discusses...",
        "relationships_affected": 5
    },
    "message": "This operation requires user confirmation.",
    "pending_operation_id": "pending-abc123"
}
```

2. **User confirms/denies** via UI or API

3. **LLM receives confirmation result**:
```python
# If approved:
{
    "success": True,
    "operation": "delete_node",
    "node_id": "obs-123",
    "message": "Node deleted successfully with user confirmation."
}

# If denied:
{
    "success": False,
    "isError": False,  # Not an error, user choice
    "operation": "delete_node",
    "message": "The deletion operation was denied by the user.",
    "user_feedback": "User chose to preserve this observation."
}
```

---

## Tool Registration

Tools are registered using decorators for automatic discovery:

```python
# backend/app/tools/nodes.py
from app.tools.base import tool, ToolResult

@tool(
    name="find_related_nodes",
    description="Find semantically similar nodes using vector embeddings",
    category="analysis"
)
async def find_related_nodes(
    node_id: str,
    limit: int = 10,
    min_similarity: float = 0.5,
    auto_link: bool = False
) -> ToolResult:
    """Implementation..."""
    # Tool logic here
    return ToolResult(success=True, data={...})
```

**Registry automatically exposes to**:
- LangGraph agents
- API endpoints (`/api/v1/tools/*`)
- Future MCP server
- Frontend tool palette

---

## Activity Feed Integration

Every tool call creates an Activity entry:

```python
{
    "type": "tool_execution",
    "tool_name": "find_related_nodes",
    "node_id": "obs-123",
    "status": "completed",
    "result_summary": "Found 5 related nodes, created 2 relationships",
    "initiated_by": "llm-agent" | "user" | "system",
    "timestamp": "2025-11-30T12:34:56Z"
}
```

---

## Implementation Phases

### Phase 7.1: Core Tool Infrastructure
- [ ] Create `backend/app/tools/` module structure
- [ ] Implement `ToolResult` base class
- [ ] Create `@tool` decorator with registration
- [ ] Build tool registry and discovery
- [ ] Add activity logging for tool calls

### Phase 7.2: Node Analysis Tools
- [ ] `find_related_nodes` (uses existing AI workflow)
- [ ] `summarize_node` (LLM summarization)
- [ ] `summarize_node_with_context` (context-aware)
- [ ] `recalculate_node_confidence` (AI re-analysis)
- [ ] `reclassify_node` (type conversion)

### Phase 7.3: Edge Analysis Tools
- [ ] `recalculate_edge_confidence`
- [ ] `reclassify_relationship`
- [ ] `summarize_relationship`
- [ ] `merge_nodes` (with confirmation)

### Phase 7.4: Web Search Integration
- [ ] `search_web_for_evidence` (Tavily/Google/Bing API)
- [ ] Auto-create Source nodes from results
- [ ] Link sources to originating node

### Phase 7.5: User Confirmation System
- [ ] Pending operations queue (Redis)
- [ ] Confirmation API endpoints
- [ ] Frontend confirmation modal
- [ ] LLM feedback mechanism

### Phase 7.6: LangGraph Agent
- [ ] Natural language interface to tools
- [ ] Intent classification (which tool to use)
- [ ] Multi-step reasoning chains
- [ ] Conversation memory

---

## Example Usage Flows

### Flow 1: User asks "Find evidence for this hypothesis"

```
1. User: "Find evidence supporting hypothesis-456"

2. LangGraph Agent decides: use search_web_for_evidence

3. Tool executes:
   - Searches web for "evidence for [hypothesis content]"
   - Evaluates each result with LLM
   - Creates Source nodes
   - Links to hypothesis

4. Returns to LLM:
   {
     "success": true,
     "results": [...],
     "sources_created": 3
   }

5. LLM responds to user:
   "I found 3 relevant sources supporting your hypothesis:
    - [Source 1]: Strong support (0.85 confidence)
    - [Source 2]: Moderate support (0.65 confidence)
    - [Source 3]: Related context (0.55 confidence)

    I've added these as Source nodes and linked them."
```

### Flow 2: User asks "Delete this observation"

```
1. User: "Delete observation obs-789"

2. LangGraph Agent: use delete_node tool

3. Tool returns:
   {
     "success": false,
     "requires_confirmation": true,
     "details": {
       "relationships_affected": 12,
       "content_preview": "..."
     },
     "pending_operation_id": "pending-123"
   }

4. LLM asks user:
   "This observation has 12 relationships. Are you sure you want to delete it?

    Preview: '...'

    Reply 'yes' to confirm or 'no' to cancel."

5a. User: "yes"
    → Confirmation API called
    → Tool executes deletion
    → LLM: "Observation deleted successfully."

5b. User: "no"
    → Confirmation API called with denial
    → Tool returns: {"message": "Deletion denied by user", "isError": false}
    → LLM: "Understood, I've kept the observation."
```

---

## Testing Strategy

### Unit Tests
- Test each tool independently
- Mock all external dependencies (Neo4j, OpenAI, web APIs)
- Verify ToolResult structure

### Integration Tests
- Test tool chains (find_related → create_relationship)
- Test confirmation flow end-to-end
- Test activity logging

### LLM Agent Tests
- Test intent classification (right tool selected)
- Test multi-step reasoning
- Test error handling and recovery

---

## Security Considerations

1. **Rate Limiting**: Prevent tool abuse (especially web_search)
2. **Cost Tracking**: Log OpenAI tokens per tool call
3. **Audit Trail**: All operations logged with initiator
4. **Confirmation Required**: Destructive ops always need approval
5. **Scope Limits**: Tools can't access system files, env vars, etc.

---

## Future Extensions

- **Batch Operations**: Apply tool to multiple nodes
- **Scheduled Operations**: Re-analyze graph nightly
- **Tool Composition**: Chain tools in workflows
- **Custom Tools**: User-defined operations
- **MCP Exposure**: All tools available via MCP protocol
- **CLI Access**: `thoughtlab find-related obs-123`
