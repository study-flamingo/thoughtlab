# Phase 7: Backend API Specification

This document defines all backend API endpoints needed for LLM tool operations. These endpoints will be consumed by the LangGraph agent layer in Phase 8.

---

## Design Principles

1. **REST-first**: All operations exposed as HTTP endpoints
2. **Synchronous**: Return results immediately (ARQ migration later)
3. **Stateless**: No session state, use node/edge IDs
4. **Self-contained**: Each endpoint is independently testable
5. **Rich responses**: Return detailed results with metadata

---

## Node Analysis Endpoints

### 1. Find Related Nodes

**Endpoint**: `POST /api/v1/nodes/{node_id}/find-related`

**Purpose**: Find semantically similar nodes using vector embeddings.

**Request Body**:
```json
{
  "limit": 10,
  "min_similarity": 0.5,
  "node_types": ["Observation", "Hypothesis"],  // Optional filter
  "auto_link": false  // If true, auto-create relationships
}
```

**Response**:
```json
{
  "success": true,
  "node_id": "obs-123",
  "related_nodes": [
    {
      "id": "obs-456",
      "type": "Observation",
      "content": "Text preview...",
      "similarity_score": 0.87,
      "suggested_relationship": "RELATES_TO",
      "reasoning": "Both discuss quantum entanglement..."
    }
  ],
  "links_created": 0,  // If auto_link=true
  "message": "Found 5 related nodes"
}
```

---

### 2. Summarize Node

**Endpoint**: `POST /api/v1/nodes/{node_id}/summarize`

**Purpose**: Generate LLM summary of a node's content.

**Request Body**:
```json
{
  "max_length": 200,  // Characters
  "style": "concise"  // "concise" | "detailed" | "bullet_points"
}
```

**Response**:
```json
{
  "success": true,
  "node_id": "obs-123",
  "summary": "This observation discusses...",
  "key_points": [
    "Main finding about X",
    "Secondary observation about Y"
  ],
  "word_count": 42
}
```

---

### 3. Summarize Node with Context

**Endpoint**: `POST /api/v1/nodes/{node_id}/summarize-with-context`

**Purpose**: Summarize node including its relationships and connected nodes.

**Request Body**:
```json
{
  "depth": 1,  // How many hops to include
  "relationship_types": ["SUPPORTS", "CONTRADICTS"],  // Optional filter
  "max_length": 300
}
```

**Response**:
```json
{
  "success": true,
  "node_id": "hyp-789",
  "summary": "This hypothesis proposes...",
  "context": {
    "supports": [
      "Observation obs-123: Evidence showing..."
    ],
    "contradicts": [
      "Source src-456: Study found opposite results..."
    ],
    "related": [
      "Concept con-789: Related to quantum mechanics..."
    ]
  },
  "synthesis": "Overall, this hypothesis has strong support from 3 observations but conflicts with recent experimental data.",
  "relationship_count": 7
}
```

---

### 4. Recalculate Node Confidence

**Endpoint**: `POST /api/v1/nodes/{node_id}/recalculate-confidence`

**Purpose**: Re-analyze node confidence based on current graph context.

**Request Body**:
```json
{
  "factor_in_relationships": true  // Consider connected nodes
}
```

**Response**:
```json
{
  "success": true,
  "node_id": "obs-123",
  "old_confidence": 0.7,
  "new_confidence": 0.85,
  "reasoning": "Confidence increased due to supporting evidence from 3 new sources",
  "factors": [
    {"factor": "Supporting evidence", "impact": "+0.1"},
    {"factor": "Source credibility", "impact": "+0.05"}
  ]
}
```

---

### 5. Reclassify Node

**Endpoint**: `POST /api/v1/nodes/{node_id}/reclassify`

**Purpose**: Change node type (e.g., Observation → Hypothesis).

**Request Body**:
```json
{
  "new_type": "Hypothesis",  // "Observation" | "Hypothesis" | "Source" | etc.
  "preserve_relationships": true
}
```

**Response**:
```json
{
  "success": true,
  "node_id": "obs-123",
  "old_type": "Observation",
  "new_type": "Hypothesis",
  "message": "Node reclassified successfully",
  "warning": null,  // Warnings if data loss possible
  "relationships_preserved": 5
}
```

---

## Relationship Analysis Endpoints

### 6. Recalculate Edge Confidence

**Endpoint**: `POST /api/v1/relationships/{edge_id}/recalculate-confidence`

**Purpose**: Re-evaluate relationship confidence using LLM.

**Request Body**:
```json
{
  "consider_graph_structure": true  // Use broader context beyond just the two nodes
}
```

**Response**:
```json
{
  "success": true,
  "edge_id": "rel-abc",
  "from_node_id": "obs-123",
  "to_node_id": "hyp-456",
  "old_confidence": 0.6,
  "new_confidence": 0.75,
  "reasoning": "Relationship strengthened by additional supporting observations in the graph"
}
```

---

### 7. Reclassify Relationship

**Endpoint**: `POST /api/v1/relationships/{edge_id}/reclassify`

**Purpose**: Change relationship type or let LLM suggest best type.

**Request Body**:
```json
{
  "new_type": null,  // If null, LLM suggests best type
  "preserve_notes": true
}
```

**Response**:
```json
{
  "success": true,
  "edge_id": "rel-abc",
  "old_type": "RELATES_TO",
  "new_type": "SUPPORTS",
  "confidence": 0.82,
  "reasoning": "Analysis shows the source provides direct evidence for the hypothesis",
  "notes_preserved": true
}
```

---

### 8. Summarize Relationship

**Endpoint**: `POST /api/v1/relationships/{edge_id}/summarize`

**Purpose**: Explain the connection between two nodes in plain language.

**Request Body**:
```json
{
  "include_evidence": true  // Show supporting data
}
```

**Response**:
```json
{
  "success": true,
  "edge_id": "rel-abc",
  "from_node": {
    "id": "obs-123",
    "type": "Observation",
    "content": "Particles showed entanglement..."
  },
  "to_node": {
    "id": "hyp-456",
    "type": "Hypothesis",
    "content": "Quantum mechanics predicts..."
  },
  "relationship_type": "SUPPORTS",
  "summary": "This observation provides experimental evidence that supports the hypothesis about quantum entanglement",
  "evidence": [
    "Direct measurement of entangled state",
    "Results match theoretical predictions"
  ],
  "strength_assessment": "strong"  // "strong" | "moderate" | "weak"
}
```

---

## Advanced Operations

### 9. Search Web for Evidence

**Endpoint**: `POST /api/v1/tools/search-web-evidence`

**Purpose**: Search web for supporting/contradicting evidence for a node.

**Request Body**:
```json
{
  "node_id": "hyp-123",
  "query_override": null,  // Custom search query (optional)
  "evidence_type": "all",  // "supporting" | "contradicting" | "all"
  "max_results": 5,
  "auto_create_sources": false  // If true, create Source nodes
}
```

**Response**:
```json
{
  "success": true,
  "node_id": "hyp-123",
  "results": [
    {
      "url": "https://example.com/paper",
      "title": "Recent findings on quantum entanglement",
      "snippet": "Study confirms that...",
      "relevance_score": 0.89,
      "evidence_type": "supporting",  // "supporting" | "contradicting" | "neutral"
      "reasoning": "This paper provides experimental validation..."
    }
  ],
  "sources_created": 0,  // If auto_create_sources=true
  "message": "Found 5 relevant results"
}
```

---

### 10. Merge Nodes

**Endpoint**: `POST /api/v1/nodes/merge`

**Purpose**: Combine duplicate or very similar nodes (requires confirmation).

**Request Body**:
```json
{
  "primary_node_id": "obs-123",  // Node to keep
  "secondary_node_id": "obs-456",  // Node to merge in
  "merge_strategy": "combine"  // "combine" | "prefer_primary" | "prefer_secondary"
}
```

**Response** (Requires Confirmation):
```json
{
  "success": false,
  "requires_confirmation": true,
  "operation": "merge_nodes",
  "details": {
    "primary_node_id": "obs-123",
    "secondary_node_id": "obs-456",
    "primary_content": "First observation about...",
    "secondary_content": "Similar observation about...",
    "merged_content_preview": "Combined observation about...",
    "relationships_to_transfer": 3
  },
  "message": "This operation requires user confirmation",
  "pending_operation_id": "pending-xyz123"
}
```

**After Confirmation** (`POST /api/v1/operations/{pending_id}/confirm`):
```json
{
  "success": true,
  "operation": "merge_nodes",
  "primary_node_id": "obs-123",
  "secondary_node_id": "obs-456",
  "merged_content": "Combined text...",
  "relationships_transferred": 3,
  "message": "Nodes merged successfully"
}
```

---

## Confirmation System Endpoints

### 11. Confirm Pending Operation

**Endpoint**: `POST /api/v1/operations/{pending_id}/confirm`

**Purpose**: Approve a pending destructive operation.

**Request Body**:
```json
{
  "approved": true,
  "user_feedback": "Approved by user"  // Optional
}
```

**Response**:
```json
{
  "success": true,
  "operation_id": "pending-xyz123",
  "operation_type": "merge_nodes",
  "result": {
    // Operation-specific result
  }
}
```

---

### 12. Deny Pending Operation

**Endpoint**: `POST /api/v1/operations/{pending_id}/deny`

**Purpose**: Reject a pending operation.

**Request Body**:
```json
{
  "user_feedback": "User chose to preserve both nodes"  // Optional
}
```

**Response**:
```json
{
  "success": true,
  "operation_id": "pending-xyz123",
  "operation_type": "merge_nodes",
  "message": "Operation denied by user"
}
```

---

## Implementation Priority

### High Priority (Core LLM Tools)
1. ✅ Find Related Nodes (leverage existing similarity search)
2. ✅ Summarize Node
3. ✅ Summarize Node with Context
4. ✅ Recalculate Node Confidence
5. ✅ Summarize Relationship

### Medium Priority (Advanced Tools)
6. ⚠️ Search Web for Evidence (requires external API)
7. ⚠️ Recalculate Edge Confidence
8. ⚠️ Reclassify Relationship
9. ⚠️ Reclassify Node

### Low Priority (Complex Operations)
10. 🔄 Merge Nodes (requires confirmation system)
11. 🔄 Pending Operation Confirmation/Denial

---

## Error Handling

All endpoints follow consistent error patterns:

**404 Not Found**:
```json
{
  "success": false,
  "error": "Node not found",
  "node_id": "obs-999"
}
```

**400 Bad Request**:
```json
{
  "success": false,
  "error": "Invalid parameters",
  "details": "min_similarity must be between 0 and 1"
}
```

**500 Internal Server Error**:
```json
{
  "success": false,
  "error": "OpenAI API error",
  "details": "Rate limit exceeded"
}
```

---

## Next Steps

1. Create service methods in `backend/app/services/tool_service.py`
2. Implement API routes in `backend/app/api/routes/tools.py`
3. Test each endpoint independently
4. Document with OpenAPI/Swagger
5. Prepare for Phase 8 (LangGraph integration)
