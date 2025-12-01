# LangChain + LangGraph Implementation Plan

This document outlines the implementation plan for integrating LangChain and LangGraph into ThoughtLab for AI-powered relationship discovery, embedding generation, and a unified tool architecture.

---

## Table of Contents

1. [Overview](#overview)
2. [Unified Tool Architecture](#unified-tool-architecture)
3. [LangChain vs LangGraph](#langchain-vs-langgraph)
4. [Architecture](#architecture)
5. [Dependencies](#dependencies)
6. [Configuration](#configuration)
7. [Module Structure](#module-structure)
8. [Implementation Components](#implementation-components)
9. [Implementation Order](#implementation-order)
10. [Neo4j Vector Indexes](#neo4j-vector-indexes)
11. [ARQ Background Processing Upgrade](#arq-background-processing-upgrade)
12. [MCP Server Integration](#mcp-server-integration)
13. [Chrome Extension](#chrome-extension)
14. [Testing Strategy](#testing-strategy)

---

## Overview

The AI integration enables:

1. **Embedding Generation**: Convert node content to vector embeddings using OpenAI
2. **Similarity Search**: Find related content using Neo4j vector indexes
3. **Relationship Classification**: Use LLM to identify relationship types between nodes
4. **Automated Suggestions**: Create relationship suggestions based on confidence thresholds
5. **Unified Tool Layer**: Common functions accessible by LangGraph agents, MCP server, and frontend

---

## Unified Tool Architecture

The key architectural principle is a **shared tool layer** that can be invoked by multiple interfaces:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           TOOL LAYER (Core Logic)                           │
│                                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐│
│  │ create_node │  │search_similar│ │classify_rel │  │ query_graph         ││
│  │             │  │             │  │             │  │                     ││
│  │ update_node │  │embed_content│  │suggest_rel  │  │ get_node_context    ││
│  │             │  │             │  │             │  │                     ││
│  │ delete_node │  │find_related │  │approve_rel  │  │ analyze_connections ││
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────────────┘│
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
          ┌───────────────────────────┼───────────────────────────┐
          │                           │                           │
          ▼                           ▼                           ▼
┌─────────────────────┐   ┌─────────────────────┐   ┌─────────────────────┐
│    LangGraph Agent  │   │     MCP Server      │   │      Frontend       │
│                     │   │                     │   │                     │
│ - Intelligent tool  │   │ - External AI apps  │   │ - Manual user       │
│   selection         │   │   (Claude, etc.)    │   │   actions           │
│ - Multi-step        │   │ - Standard protocol │   │ - API calls         │
│   reasoning         │   │ - Tool discovery    │   │ - Direct invocation │
│ - Background        │   │                     │   │                     │
│   processing        │   │                     │   │                     │
└─────────────────────┘   └─────────────────────┘   └─────────────────────┘
          │                           │                           │
          └───────────────────────────┼───────────────────────────┘
                                      ▼
                        ┌─────────────────────────┐
                        │   Chrome Extension      │
                        │                         │
                        │ - Capture web content   │
                        │ - Quick-add to graph    │
                        │ - Context menu actions  │
                        └─────────────────────────┘
```

### Why This Matters

1. **Single Source of Truth**: All logic lives in the tool layer, not duplicated across interfaces
2. **Consistent Behavior**: Same validation, same business rules everywhere
3. **Easy Testing**: Test tools once, confidence in all interfaces
4. **Flexibility**: Add new interfaces (CLI, Slack bot, etc.) without rewriting logic

---

## LangChain vs LangGraph

### When to Use LangChain

- Simple linear chains (prompt → LLM → response)
- Basic RAG (retrieve → augment → generate)
- Single-step operations
- Straightforward Q&A

### When to Use LangGraph

- **Complex workflows** with branching logic ✅
- **Intelligent tool selection** from a set of available tools ✅
- **Multi-step reasoning** where each step depends on previous results ✅
- **Human-in-the-loop** at arbitrary points ✅
- **State management** across conversation turns ✅
- **Parallel execution** of independent tasks ✅

### Our Decision: LangGraph

ThoughtLab requires LangGraph because:

| Requirement | Why LangGraph |
|-------------|---------------|
| Relationship discovery | Multi-step: embed → search → classify → decide |
| Confidence-based routing | Branch: auto-create vs suggest vs discard |
| Natural language queries | Agent selects tools based on question type |
| Background processing | Stateful workflows with interrupts |
| MCP compatibility | Tools can be exposed via MCP protocol |

### Architecture Comparison

**LangChain (Linear)**:
```
User Query → Embed → Search → Generate Response
```

**LangGraph (Intelligent)**:
```
User Query → Agent decides:
  ├── "Find related nodes" → search_similar tool
  ├── "Create observation" → create_node tool
  ├── "What supports X?" → query_graph + analyze tool
  └── "Add this source" → create_node + embed + find_related tools
```

### Confidence Thresholds

| Confidence | Action |
|------------|--------|
| ≥ 0.8 | Auto-create relationship (marked as `created_by: system-llm`) |
| 0.6 - 0.8 | Create suggestion in Activity Feed for user review |
| < 0.6 | Discard silently |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              USER CREATES NODE                              │
│                                     │                                       │
│                                     ▼                                       │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                    ProcessingService.process_node()                    │  │
│  │                                                                        │  │
│  │   ┌─────────────┐                                                     │  │
│  │   │   CHUNK     │ RecursiveCharacterSplitter                          │  │
│  │   │  (if long)  │ - chunk_size: 1000                                  │  │
│  │   └──────┬──────┘ - overlap: 200                                      │  │
│  │          │                                                             │  │
│  │          ▼                                                             │  │
│  │   ┌─────────────┐                                                     │  │
│  │   │   EMBED     │ OpenAIEmbeddings                                    │  │
│  │   │             │ - model: text-embedding-3-small                     │  │
│  │   └──────┬──────┘ - dimensions: 1536                                  │  │
│  │          │                                                             │  │
│  │          ▼                                                             │  │
│  │   ┌─────────────┐                                                     │  │
│  │   │   SEARCH    │ Neo4jVector.similarity_search                       │  │
│  │   │  (similar)  │ - min_score: 0.5                                    │  │
│  │   └──────┬──────┘ - limit: 20 candidates                              │  │
│  │          │                                                             │  │
│  │          ▼                                                             │  │
│  │   ┌─────────────┐                                                     │  │
│  │   │  CLASSIFY   │ ChatOpenAI + structured_output                      │  │
│  │   │ (parallel)  │ - model: gpt-4o-mini                                │  │
│  │   └──────┬──────┘ - temperature: 0.1                                  │  │
│  │          │                                                             │  │
│  │          ├── confidence ≥ 0.8 ────► Create relationship               │  │
│  │          ├── confidence ≥ 0.6 ────► Create suggestion activity        │  │
│  │          └── confidence < 0.6 ────► Discard                           │  │
│  │                                                                        │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                     │                                       │
│                                     ▼                                       │
│                          Activity Feed Updates                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Dependencies

Add to `backend/requirements.txt`:

```txt
# LangChain AI Integration
langchain>=0.3.0
langchain-openai>=0.2.0
langchain-neo4j>=0.2.0
langgraph>=0.2.0
```

Install command:
```bash
cd backend
uv pip install langchain langchain-openai langchain-neo4j langgraph
uv pip freeze > requirements.txt  # Update lockfile
```

---

## Configuration

### Environment Variables (`.env`)

Create a `.env` file in the project root:

```env
# OpenAI API
THOUGHTLAB_OPENAI_API_KEY=sk-...

# Model Configuration (optional, defaults shown)
THOUGHTLAB_LLM_MODEL=gpt-4o-mini
THOUGHTLAB_EMBEDDING_MODEL=text-embedding-3-small
THOUGHTLAB_EMBEDDING_DIMENSIONS=1536

# Processing Configuration (optional, defaults shown)
THOUGHTLAB_AUTO_CREATE_THRESHOLD=0.8
THOUGHTLAB_SUGGEST_THRESHOLD=0.6
THOUGHTLAB_SIMILARITY_MIN_SCORE=0.5
THOUGHTLAB_MAX_SIMILAR_NODES=20
THOUGHTLAB_CHUNK_SIZE=1000
THOUGHTLAB_CHUNK_OVERLAP=200
```

### Configuration Class

```python
# backend/app/ai/config.py
from pydantic_settings import BaseSettings

class AIConfig(BaseSettings):
    """Configuration for AI/LLM integration."""
    
    # OpenAI
    openai_api_key: str
    
    # Models
    llm_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536
    
    # Thresholds
    auto_create_threshold: float = 0.8
    suggest_threshold: float = 0.6
    similarity_min_score: float = 0.5
    
    # Processing
    max_similar_nodes: int = 20
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    class Config:
        env_prefix = "THOUGHTLAB_"
        env_file = ".env"
        env_file_encoding = "utf-8"
```

---

## Module Structure

```
backend/app/
├── tools/                           # NEW: Shared tool layer
│   ├── __init__.py                  # Tool registry and exports
│   ├── base.py                      # Base tool class and decorators
│   ├── nodes.py                     # Node CRUD tools
│   ├── search.py                    # Similarity search tools
│   ├── relationships.py             # Relationship tools
│   ├── analysis.py                  # Analysis and query tools
│   └── registry.py                  # Tool discovery for LangGraph/MCP
│
├── ai/                              # NEW: AI integration module
│   ├── __init__.py                  # Exports and lazy initialization
│   ├── config.py                    # AIConfig settings
│   ├── embeddings.py                # OpenAI embeddings + Neo4j storage
│   ├── similarity.py                # Vector similarity search
│   ├── classifier.py                # Relationship classification LLM
│   ├── workflow.py                  # LangGraph processing workflow
│   └── agent.py                     # LangGraph agent with tool selection
│
├── services/
│   ├── embedding_service.py         # UPDATE: Delegate to ai/embeddings.py
│   └── processing_service.py        # UPDATE: Call LangGraph workflow
├── utils/
│   └── chunking.py                  # EXISTING: RecursiveCharacterSplitter
└── core/
    └── config.py                    # UPDATE: Add ai_config property

mcp-server/                          # FUTURE: Companion MCP server
├── src/thoughtlab_mcp/
│   ├── server.py                    # MCP server entry point
│   ├── tools.py                     # Thin wrappers calling backend/app/tools
│   └── client.py                    # ThoughtLab API client
└── pyproject.toml

chrome-extension/                    # FUTURE: Browser extension
├── src/
│   ├── background/
│   ├── content/
│   ├── popup/
│   └── shared/
└── manifest.json
```

### Tool Layer Design

The tool layer is the **single source of truth** for all operations:

```python
# backend/app/tools/base.py
from typing import Callable, TypeVar, Any
from functools import wraps
from pydantic import BaseModel

class ToolResult(BaseModel):
    """Standard result from any tool invocation."""
    success: bool
    data: Any = None
    error: str | None = None
    message: str | None = None

def tool(
    name: str,
    description: str,
    category: str = "general",
):
    """Decorator to register a function as a tool.
    
    Tools registered this way are:
    1. Available to LangGraph agents
    2. Exposable via MCP server
    3. Callable from API routes
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> ToolResult:
            try:
                result = await func(*args, **kwargs)
                return ToolResult(success=True, data=result)
            except Exception as e:
                return ToolResult(success=False, error=str(e))
        
        # Register in tool registry
        wrapper._tool_meta = {
            "name": name,
            "description": description,
            "category": category,
            "parameters": func.__annotations__,
        }
        return wrapper
    return decorator
```

```python
# backend/app/tools/nodes.py
from app.tools.base import tool, ToolResult
from app.services.graph_service import graph_service

@tool(
    name="create_observation",
    description="Create an observation in the knowledge graph",
    category="nodes",
)
async def create_observation(
    text: str,
    confidence: float = 1.0,
    concept_names: list[str] | None = None,
) -> dict:
    """Create an observation node."""
    from app.models.nodes import ObservationCreate
    
    data = ObservationCreate(
        text=text,
        confidence=confidence,
        concept_names=concept_names or [],
    )
    node_id = await graph_service.create_observation(data)
    return {"id": node_id, "type": "Observation"}

@tool(
    name="search_similar",
    description="Find nodes semantically similar to the query",
    category="search",
)
async def search_similar(
    query: str,
    limit: int = 10,
    node_types: list[str] | None = None,
    min_score: float = 0.5,
) -> list[dict]:
    """Search for similar content using vector similarity."""
    from app.ai.similarity import similarity_search
    
    return await similarity_search.find_similar(
        query_text=query,
        node_types=node_types,
        limit=limit,
        min_score=min_score,
    )
```

---

## Implementation Components

### 1. AI Configuration (`ai/config.py`)

```python
from pydantic_settings import BaseSettings
from functools import lru_cache

class AIConfig(BaseSettings):
    """Configuration for AI/LLM integration."""
    
    # OpenAI
    openai_api_key: str = ""
    
    # Models
    llm_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536
    
    # Thresholds
    auto_create_threshold: float = 0.8
    suggest_threshold: float = 0.6
    similarity_min_score: float = 0.5
    
    # Processing
    max_similar_nodes: int = 20
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    @property
    def is_configured(self) -> bool:
        """Check if AI is properly configured."""
        return bool(self.openai_api_key)
    
    class Config:
        env_prefix = "THOUGHTLAB_"
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_ai_config() -> AIConfig:
    """Get cached AI configuration."""
    return AIConfig()
```

### 2. Embedding Manager (`ai/embeddings.py`)

```python
from langchain_openai import OpenAIEmbeddings
from typing import List, Optional
from app.db.neo4j import neo4j_conn
from app.ai.config import AIConfig
import logging

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """Manages embedding generation and storage in Neo4j."""
    
    def __init__(self, config: AIConfig):
        self.config = config
        self._embeddings = None
    
    @property
    def embeddings(self) -> OpenAIEmbeddings:
        """Lazy initialization of embeddings client."""
        if self._embeddings is None:
            self._embeddings = OpenAIEmbeddings(
                model=self.config.embedding_model,
                openai_api_key=self.config.openai_api_key,
            )
        return self._embeddings
    
    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding for text."""
        return await self.embeddings.aembed_query(text)
    
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts (batch)."""
        return await self.embeddings.aembed_documents(texts)
    
    async def store_embedding(
        self,
        node_id: str,
        embedding: List[float],
    ) -> bool:
        """Store embedding on a Neo4j node."""
        query = """
        MATCH (n {id: $node_id})
        SET n.embedding = $embedding
        RETURN n.id as id
        """
        
        async with neo4j_conn.get_session() as session:
            result = await session.run(
                query,
                node_id=node_id,
                embedding=embedding,
            )
            record = await result.single()
            return record is not None
    
    async def embed_and_store(
        self,
        node_id: str,
        text: str,
    ) -> bool:
        """Generate embedding and store on node."""
        try:
            embedding = await self.embed_text(text)
            return await self.store_embedding(node_id, embedding)
        except Exception as e:
            logger.error(f"Failed to embed and store for {node_id}: {e}")
            return False
```

### 3. Similarity Search (`ai/similarity.py`)

```python
from langchain_openai import OpenAIEmbeddings
from typing import List, Optional, Dict, Any
from app.db.neo4j import neo4j_conn
from app.ai.config import AIConfig
import logging

logger = logging.getLogger(__name__)


class SimilaritySearch:
    """Vector similarity search using Neo4j indexes."""
    
    def __init__(self, config: AIConfig):
        self.config = config
        self._embeddings = None
    
    @property
    def embeddings(self) -> OpenAIEmbeddings:
        """Lazy initialization of embeddings client."""
        if self._embeddings is None:
            self._embeddings = OpenAIEmbeddings(
                model=self.config.embedding_model,
                openai_api_key=self.config.openai_api_key,
            )
        return self._embeddings
    
    async def find_similar(
        self,
        query_text: str,
        exclude_node_id: Optional[str] = None,
        node_types: Optional[List[str]] = None,
        limit: int = 20,
        min_score: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """Find similar nodes using vector similarity.
        
        Args:
            query_text: Text to find similar content for
            exclude_node_id: Node ID to exclude (usually the source node)
            node_types: Filter by node types (Observation, Source, etc.)
            limit: Maximum results to return
            min_score: Minimum similarity score (0-1)
            
        Returns:
            List of similar nodes with scores
        """
        # Generate query embedding
        query_embedding = await self.embeddings.aembed_query(query_text)
        
        # Build node type filter
        type_filter = ""
        if node_types:
            labels = " OR ".join([f"n:{t}" for t in node_types])
            type_filter = f"WHERE ({labels})"
        
        # Query using Neo4j vector index
        # Note: This queries all embeddable node types
        query = f"""
        CALL db.index.vector.queryNodes('node_embedding', $limit * 2, $embedding)
        YIELD node as n, score
        {type_filter}
        {"AND" if type_filter else "WHERE"} n.id <> $exclude_id
        AND score >= $min_score
        RETURN n.id as node_id, 
               labels(n)[0] as node_type,
               COALESCE(n.text, n.title, n.name, n.content, n.description) as content,
               score
        ORDER BY score DESC
        LIMIT $limit
        """
        
        async with neo4j_conn.get_session() as session:
            result = await session.run(
                query,
                embedding=query_embedding,
                exclude_id=exclude_node_id or "",
                min_score=min_score,
                limit=limit,
            )
            
            results = []
            async for record in result:
                results.append({
                    "node_id": record["node_id"],
                    "node_type": record["node_type"],
                    "content": record["content"],
                    "score": record["score"],
                })
            
            return results
```

### 4. Relationship Classifier (`ai/classifier.py`)

```python
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import Optional
from app.ai.config import AIConfig
import logging

logger = logging.getLogger(__name__)


class RelationshipClassification(BaseModel):
    """Structured output for relationship classification."""
    
    relationship_type: str = Field(
        description="The type of relationship (SUPPORTS, CONTRADICTS, RELATES_TO, CITES, DERIVED_FROM, DISCUSSES, PART_OF, SIMILAR_TO)"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence score (0-1) for this classification"
    )
    reasoning: str = Field(
        description="Brief explanation for why this relationship exists"
    )
    is_valid: bool = Field(
        description="Whether a meaningful relationship exists between these nodes"
    )


class RelationshipClassifier:
    """Classifies relationships between nodes using LLM."""
    
    SYSTEM_PROMPT = """You are an expert at identifying relationships between 
pieces of information in a knowledge graph for research purposes.

Given two pieces of content, determine if there's a meaningful relationship.

## Relationship Types

- SUPPORTS: Source provides evidence for or validates target
- CONTRADICTS: Source conflicts with or challenges target
- RELATES_TO: General topical connection worth noting
- CITES: Source explicitly references target
- DERIVED_FROM: Source was inspired by or built upon target
- DISCUSSES: Source talks about the same topic as target
- PART_OF: Source is a component or subset of target
- SIMILAR_TO: Content is semantically similar but distinct

## Confidence Guidelines

Be conservative with confidence scores:
- 0.9+ : Very clear, explicit relationship with strong evidence
- 0.7-0.9: Strong implied relationship, high certainty
- 0.5-0.7: Possible relationship, moderate certainty
- <0.5: Weak or speculative - set is_valid to false

## Important

- Set is_valid=false if there's no meaningful research-relevant relationship
- Focus on relationships that would help a researcher understand connections
- Don't create relationships just because topics are vaguely related
"""
    
    def __init__(self, config: AIConfig):
        self.config = config
        self._llm = None
    
    @property
    def llm(self):
        """Lazy initialization of LLM with structured output."""
        if self._llm is None:
            base_llm = ChatOpenAI(
                model=self.config.llm_model,
                openai_api_key=self.config.openai_api_key,
                temperature=0.1,  # Low temp for consistent classification
            )
            self._llm = base_llm.with_structured_output(RelationshipClassification)
        return self._llm
    
    async def classify(
        self,
        source_content: str,
        source_type: str,
        target_content: str,
        target_type: str,
    ) -> Optional[RelationshipClassification]:
        """Classify the relationship between two pieces of content.
        
        Returns None if classification fails.
        """
        # Truncate content to avoid token limits
        source_preview = source_content[:1000] if len(source_content) > 1000 else source_content
        target_preview = target_content[:1000] if len(target_content) > 1000 else target_content
        
        prompt = f"""Analyze if there's a meaningful relationship between these two items.

## Source ({source_type})
{source_preview}

## Target ({target_type})
{target_preview}

Determine the relationship type, confidence, and provide reasoning."""
        
        try:
            result = await self.llm.ainvoke([
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ])
            return result
        except Exception as e:
            logger.error(f"Classification failed: {e}")
            return None
```

### 5. Processing Workflow (`ai/workflow.py`)

```python
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
import logging
import uuid

from app.ai.config import AIConfig, get_ai_config
from app.ai.embeddings import EmbeddingManager
from app.ai.similarity import SimilaritySearch
from app.ai.classifier import RelationshipClassifier, RelationshipClassification
from app.utils.chunking import chunk_text, should_chunk, Chunk
from app.services.graph_service import graph_service
from app.services.activity_service import activity_service
from app.models.activity import ProcessingData, SuggestionData
from app.models.nodes import ChunkCreate

logger = logging.getLogger(__name__)


@dataclass
class ProcessingResult:
    """Result of processing a node."""
    node_id: str
    node_type: str
    success: bool
    chunks_created: int = 0
    embeddings_created: int = 0
    candidates_found: int = 0
    suggestions_created: int = 0
    auto_created_relationships: int = 0
    error: Optional[str] = None


class AIWorkflow:
    """Orchestrates the AI processing workflow for nodes.
    
    This is the main entry point for processing nodes through:
    1. Chunking (for long content)
    2. Embedding generation
    3. Similarity search
    4. Relationship classification
    5. Suggestion/relationship creation
    """
    
    def __init__(self, config: Optional[AIConfig] = None):
        self.config = config or get_ai_config()
        self._embedding_manager = None
        self._similarity_search = None
        self._classifier = None
    
    @property
    def embedding_manager(self) -> EmbeddingManager:
        if self._embedding_manager is None:
            self._embedding_manager = EmbeddingManager(self.config)
        return self._embedding_manager
    
    @property
    def similarity_search(self) -> SimilaritySearch:
        if self._similarity_search is None:
            self._similarity_search = SimilaritySearch(self.config)
        return self._similarity_search
    
    @property
    def classifier(self) -> RelationshipClassifier:
        if self._classifier is None:
            self._classifier = RelationshipClassifier(self.config)
        return self._classifier
    
    @property
    def is_ready(self) -> bool:
        """Check if AI workflow is properly configured."""
        return self.config.is_configured
    
    async def process_node(
        self,
        node_id: str,
        node_type: str,
        content: str,
        node_label: str,
    ) -> ProcessingResult:
        """Process a node through the full AI workflow.
        
        Args:
            node_id: The node to process
            node_type: Node type (Observation, Source, etc.)
            content: Text content to process
            node_label: Display label for activity feed
            
        Returns:
            ProcessingResult with statistics
        """
        result = ProcessingResult(
            node_id=node_id,
            node_type=node_type,
            success=False,
        )
        
        if not self.is_ready:
            result.error = "AI not configured (missing THOUGHTLAB_OPENAI_API_KEY)"
            logger.warning(result.error)
            return result
        
        group_id = f"process-{node_id}-{uuid.uuid4().hex[:8]}"
        
        try:
            # Step 1: Chunking (for long content like Sources)
            chunks = await self._chunk_content(
                node_id, node_type, content, group_id, node_label
            )
            result.chunks_created = len(chunks) if chunks else 0
            
            # Step 2: Embedding
            embeddings_count = await self._embed_content(
                node_id, node_type, content, chunks, group_id, node_label
            )
            result.embeddings_created = embeddings_count
            
            # Step 3: Find similar nodes
            candidates = await self._find_candidates(
                node_id, content, group_id, node_label
            )
            result.candidates_found = len(candidates)
            
            # Step 4: Classify relationships
            suggestions, auto_created = await self._classify_and_create(
                node_id, node_type, content, node_label, candidates, group_id
            )
            result.suggestions_created = suggestions
            result.auto_created_relationships = auto_created
            
            # Step 5: Mark complete
            await self._update_status(
                group_id, "completed",
                f"Processed: {node_label[:50]}... ({suggestions} suggestions)",
                node_id, node_type, node_label,
                chunks_created=result.chunks_created,
                embeddings_created=result.embeddings_created,
                suggestions_found=result.suggestions_created,
            )
            
            result.success = True
            
        except Exception as e:
            logger.exception(f"Error processing node {node_id}")
            result.error = str(e)
            
            await self._update_status(
                group_id, "failed",
                f"Failed: {node_label[:50]}... - {str(e)[:100]}",
                node_id, node_type, node_label,
                error_message=str(e),
            )
        
        return result
    
    async def _chunk_content(
        self,
        node_id: str,
        node_type: str,
        content: str,
        group_id: str,
        node_label: str,
    ) -> Optional[List[Chunk]]:
        """Chunk content if needed."""
        # Only chunk Sources and long content
        if node_type not in ["Source"] or not should_chunk(content):
            return None
        
        await self._update_status(
            group_id, "chunking",
            f"Chunking: {node_label[:50]}...",
            node_id, node_type, node_label,
        )
        
        chunks = chunk_text(
            content,
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )
        
        # Store chunks in Neo4j
        for chunk in chunks:
            chunk_data = ChunkCreate(
                source_id=node_id,
                content=chunk.content,
                chunk_index=chunk.chunk_index,
                start_char=chunk.start_char,
                end_char=chunk.end_char,
                metadata=chunk.metadata,
            )
            await graph_service.create_chunk(chunk_data, created_by="system-llm")
        
        return chunks
    
    async def _embed_content(
        self,
        node_id: str,
        node_type: str,
        content: str,
        chunks: Optional[List[Chunk]],
        group_id: str,
        node_label: str,
    ) -> int:
        """Generate and store embeddings."""
        await self._update_status(
            group_id, "embedding",
            f"Embedding: {node_label[:50]}...",
            node_id, node_type, node_label,
        )
        
        count = 0
        
        if chunks:
            # Embed each chunk
            for chunk in chunks:
                # Create chunk node ID
                chunk_query = """
                MATCH (ch:Chunk {source_id: $source_id, chunk_index: $chunk_index})
                RETURN ch.id as id
                """
                from app.db.neo4j import neo4j_conn
                async with neo4j_conn.get_session() as session:
                    result = await session.run(
                        chunk_query,
                        source_id=node_id,
                        chunk_index=chunk.chunk_index,
                    )
                    record = await result.single()
                    if record:
                        success = await self.embedding_manager.embed_and_store(
                            record["id"],
                            chunk.content,
                        )
                        if success:
                            count += 1
        else:
            # Embed full content on the node
            success = await self.embedding_manager.embed_and_store(node_id, content)
            if success:
                count = 1
        
        return count
    
    async def _find_candidates(
        self,
        node_id: str,
        content: str,
        group_id: str,
        node_label: str,
    ) -> List[Dict[str, Any]]:
        """Find candidate nodes for relationship discovery."""
        await self._update_status(
            group_id, "analyzing",
            f"Finding connections: {node_label[:50]}...",
            node_id, "Unknown", node_label,
        )
        
        return await self.similarity_search.find_similar(
            query_text=content,
            exclude_node_id=node_id,
            limit=self.config.max_similar_nodes,
            min_score=self.config.similarity_min_score,
        )
    
    async def _classify_and_create(
        self,
        node_id: str,
        node_type: str,
        content: str,
        node_label: str,
        candidates: List[Dict[str, Any]],
        group_id: str,
    ) -> tuple[int, int]:
        """Classify relationships and create suggestions/relationships."""
        suggestions_created = 0
        auto_created = 0
        
        for candidate in candidates:
            classification = await self.classifier.classify(
                source_content=content,
                source_type=node_type,
                target_content=candidate["content"] or "",
                target_type=candidate["node_type"],
            )
            
            if classification is None or not classification.is_valid:
                continue
            
            if classification.confidence >= self.config.auto_create_threshold:
                # Auto-create relationship
                rel_id = await graph_service.create_relationship(
                    from_id=node_id,
                    to_id=candidate["node_id"],
                    rel_type=classification.relationship_type,
                    properties={
                        "confidence": classification.confidence,
                        "notes": classification.reasoning,
                    },
                    created_by="system-llm",
                )
                if rel_id:
                    auto_created += 1
                    logger.info(
                        f"Auto-created {classification.relationship_type} "
                        f"from {node_id} to {candidate['node_id']} "
                        f"(confidence: {classification.confidence})"
                    )
                    
            elif classification.confidence >= self.config.suggest_threshold:
                # Create suggestion for user review
                suggestion_data = SuggestionData(
                    from_node_id=node_id,
                    from_node_type=node_type,
                    from_node_label=node_label[:50],
                    to_node_id=candidate["node_id"],
                    to_node_type=candidate["node_type"],
                    to_node_label=(candidate["content"] or "")[:50],
                    relationship_type=classification.relationship_type,
                    confidence=classification.confidence,
                    reasoning=classification.reasoning,
                )
                await activity_service.create_suggestion(suggestion_data)
                suggestions_created += 1
        
        return suggestions_created, auto_created
    
    async def _update_status(
        self,
        group_id: str,
        stage: str,
        message: str,
        node_id: str,
        node_type: str,
        node_label: str,
        **kwargs,
    ):
        """Update processing status in activity feed."""
        processing_data = ProcessingData(
            node_id=node_id,
            node_type=node_type,
            node_label=node_label,
            stage=stage,
            **kwargs,
        )
        
        await activity_service.update_processing_status(
            group_id=group_id,
            stage=stage,
            message=message,
            processing_data=processing_data,
        )


# Global workflow instance
ai_workflow = AIWorkflow()
```

---

## Implementation Order

| Step | Component | Description | Priority |
|------|-----------|-------------|----------|
| 1 | `requirements.txt` | Add LangChain dependencies | High |
| 2 | `.env.example` | Create example env file | High |
| 3 | `ai/config.py` | AIConfig with env loading | High |
| 4 | `ai/embeddings.py` | OpenAI embeddings + Neo4j storage | High |
| 5 | `ai/similarity.py` | Vector similarity search | High |
| 6 | `ai/classifier.py` | Relationship classification | High |
| 7 | `ai/workflow.py` | Main processing workflow | High |
| 8 | Update `core/config.py` | Add AI config property | Medium |
| 9 | Update `processing_service.py` | Call AI workflow | Medium |
| 10 | Vector indexes | Enable in Neo4j init | Medium |
| 11 | Integration tests | Test full workflow | Medium |

---

## Neo4j Vector Indexes

LangChain will create indexes automatically when using `Neo4jVector`. The indexes will be named based on the configuration.

For manual control, add to `docker/neo4j/init.cypher`:

```cypher
// Create unified vector index for all embeddable nodes
// This will be created by LangChain on first use

// Alternative: Per-type indexes (more control, better performance)
CREATE VECTOR INDEX observation_embedding IF NOT EXISTS
FOR (o:Observation) ON o.embedding
OPTIONS {indexConfig: {
  `vector.dimensions`: 1536,
  `vector.similarity_function`: 'cosine'
}};

CREATE VECTOR INDEX source_embedding IF NOT EXISTS
FOR (s:Source) ON s.embedding
OPTIONS {indexConfig: {
  `vector.dimensions`: 1536,
  `vector.similarity_function`: 'cosine'
}};

CREATE VECTOR INDEX hypothesis_embedding IF NOT EXISTS
FOR (h:Hypothesis) ON h.embedding
OPTIONS {indexConfig: {
  `vector.dimensions`: 1536,
  `vector.similarity_function`: 'cosine'
}};

CREATE VECTOR INDEX concept_embedding IF NOT EXISTS
FOR (c:Concept) ON c.embedding
OPTIONS {indexConfig: {
  `vector.dimensions`: 1536,
  `vector.similarity_function`: 'cosine'
}};

CREATE VECTOR INDEX entity_embedding IF NOT EXISTS
FOR (e:Entity) ON e.embedding
OPTIONS {indexConfig: {
  `vector.dimensions`: 1536,
  `vector.similarity_function`: 'cosine'
}};

CREATE VECTOR INDEX chunk_embedding IF NOT EXISTS
FOR (ch:Chunk) ON ch.embedding
OPTIONS {indexConfig: {
  `vector.dimensions`: 1536,
  `vector.similarity_function`: 'cosine'
}};
```

---

## ARQ Background Processing Upgrade

After the synchronous implementation is working, upgrade to ARQ for non-blocking background processing.

### Why ARQ?

- **Async-native**: Matches FastAPI patterns perfectly
- **Redis-backed**: Already using Redis for caching
- **Simple**: Lightweight, easy to configure
- **Reliable**: Job persistence, retries, timeouts

### ARQ Implementation Plan

#### 1. Dependencies

```txt
# Already in requirements.txt
arq>=0.25.0
```

#### 2. Worker Configuration

Create `backend/app/workers/config.py`:

```python
from arq.connections import RedisSettings
from app.core.config import settings

def get_redis_settings() -> RedisSettings:
    return RedisSettings(
        host=settings.redis_host,
        port=settings.redis_port,
        database=settings.redis_db,
    )
```

#### 3. Job Definitions

Create `backend/app/workers/jobs.py`:

```python
from arq import cron
from app.ai.workflow import ai_workflow
from app.services.activity_service import activity_service
from app.models.activity import ActivityType
import logging

logger = logging.getLogger(__name__)


async def process_node_job(
    ctx: dict,
    node_id: str,
    node_type: str,
    content: str,
    node_label: str,
) -> dict:
    """Background job to process a node through AI workflow."""
    logger.info(f"Processing node {node_id} in background")
    
    result = await ai_workflow.process_node(
        node_id=node_id,
        node_type=node_type,
        content=content,
        node_label=node_label,
    )
    
    return {
        "node_id": result.node_id,
        "success": result.success,
        "suggestions": result.suggestions_created,
        "auto_created": result.auto_created_relationships,
    }


async def startup(ctx: dict):
    """Worker startup - initialize connections."""
    logger.info("ARQ worker starting up")
    # Initialize any needed connections
    from app.db.neo4j import neo4j_conn
    await neo4j_conn.connect()


async def shutdown(ctx: dict):
    """Worker shutdown - cleanup."""
    logger.info("ARQ worker shutting down")
    from app.db.neo4j import neo4j_conn
    await neo4j_conn.disconnect()


class WorkerSettings:
    """ARQ worker settings."""
    functions = [process_node_job]
    on_startup = startup
    on_shutdown = shutdown
    
    # Job settings
    max_jobs = 10
    job_timeout = 300  # 5 minutes max per job
    
    # Redis connection
    redis_settings = get_redis_settings()
```

#### 4. Job Enqueueing

Update `backend/app/services/processing_service.py`:

```python
from arq import create_pool
from app.workers.config import get_redis_settings

async def trigger_node_processing_async(
    node_id: str,
    node_type: str,
    content: str,
    node_label: str,
) -> str:
    """Enqueue node processing as background job."""
    redis = await create_pool(get_redis_settings())
    
    job = await redis.enqueue_job(
        "process_node_job",
        node_id=node_id,
        node_type=node_type,
        content=content,
        node_label=node_label,
    )
    
    return job.job_id
```

#### 5. Worker Startup Script

Create `backend/run_worker.py`:

```python
#!/usr/bin/env python
"""Run ARQ worker for background job processing."""

from app.workers.jobs import WorkerSettings

if __name__ == "__main__":
    import arq
    arq.run_worker(WorkerSettings)
```

#### 6. Docker Compose Service

Add to `docker-compose.yml`:

```yaml
services:
  worker:
    build:
      context: ./backend
      dockerfile: Dockerfile
    command: python run_worker.py
    environment:
      - NEO4J_URI=bolt://neo4j:7687
      - NEO4J_USER=neo4j
      - NEO4J_PASSWORD=research_graph_password
      - REDIS_URL=redis://redis:6379
      - THOUGHTLAB_OPENAI_API_KEY=${THOUGHTLAB_OPENAI_API_KEY}
    depends_on:
      - redis
      - neo4j
    volumes:
      - ./backend:/app
    restart: unless-stopped
```

#### 7. API Integration

Update node creation routes to trigger background processing:

```python
@router.post("/nodes/observations")
async def create_observation(data: ObservationCreate):
    node_id = await graph_service.create_observation(data)
    
    # Trigger background processing
    await trigger_node_processing_async(
        node_id=node_id,
        node_type="Observation",
        content=data.text,
        node_label=data.text[:50],
    )
    
    return {"id": node_id, "message": "Observation created, processing started"}
```

### ARQ Migration Steps

1. Ensure synchronous AI workflow is working
2. Add ARQ job definitions
3. Create worker startup script
4. Add worker service to Docker Compose
5. Update API routes to enqueue jobs
6. Test with manual job enqueueing
7. Monitor with ARQ dashboard (optional)

---

## MCP Server Integration

### Overview

The Model Context Protocol (MCP) allows external AI applications (like Claude Desktop) to interact with ThoughtLab's knowledge graph using the same tool layer.

### Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ThoughtLab MCP Server                               │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        Tool Registry                                 │   │
│  │                                                                      │   │
│  │  @mcp.tool("thoughtlab_create_observation")                         │   │
│  │  @mcp.tool("thoughtlab_search_similar")                             │   │
│  │  @mcp.tool("thoughtlab_query_graph")                                │   │
│  │  @mcp.tool("thoughtlab_get_node")                                   │   │
│  │  @mcp.tool("thoughtlab_suggest_relationships")                      │   │
│  │  @mcp.tool("thoughtlab_analyze_connections")                        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     Shared Tool Layer                                │   │
│  │                  (backend/app/tools/*.py)                           │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │     External AI Clients       │
                    │                               │
                    │  - Claude Desktop             │
                    │  - Cursor AI                  │
                    │  - Other MCP-compatible apps  │
                    └───────────────────────────────┘
```

### MCP Server Structure

```
mcp-server/
├── pyproject.toml
├── src/
│   └── thoughtlab_mcp/
│       ├── __init__.py
│       ├── server.py           # MCP server entry point
│       ├── tools.py            # Tool definitions (thin wrappers)
│       └── client.py           # ThoughtLab API client
└── README.md
```

### Example Tool Definition

```python
# mcp-server/src/thoughtlab_mcp/tools.py
from mcp.server import Server
from mcp.types import Tool, TextContent
from thoughtlab_mcp.client import ThoughtLabClient

app = Server("thoughtlab")
client = ThoughtLabClient()

@app.tool()
async def thoughtlab_create_observation(
    text: str,
    confidence: float = 1.0,
    concepts: list[str] = None,
) -> str:
    """Create an observation in the ThoughtLab knowledge graph.
    
    Args:
        text: The observation text (what you noticed or learned)
        confidence: How confident you are (0.0-1.0)
        concepts: Optional list of concept names to tag
    
    Returns:
        The ID of the created observation
    """
    result = await client.create_observation(
        text=text,
        confidence=confidence,
        concept_names=concepts or [],
    )
    return f"Created observation: {result['id']}"

@app.tool()
async def thoughtlab_search_similar(
    query: str,
    limit: int = 10,
    node_types: list[str] = None,
) -> str:
    """Search for nodes similar to the given query.
    
    Args:
        query: Text to find similar content for
        limit: Maximum number of results
        node_types: Filter by node types (Observation, Source, Hypothesis, etc.)
    
    Returns:
        List of similar nodes with relevance scores
    """
    results = await client.search_similar(
        query=query,
        limit=limit,
        node_types=node_types,
    )
    return format_search_results(results)

@app.tool()
async def thoughtlab_query_graph(
    question: str,
) -> str:
    """Ask a natural language question about the knowledge graph.
    
    Args:
        question: Question like "What supports hypothesis X?" or "Show connections to concept Y"
    
    Returns:
        Answer synthesized from graph data
    """
    result = await client.query_graph(question)
    return result["answer"]
```

### MCP Configuration (Claude Desktop)

```json
{
  "mcpServers": {
    "thoughtlab": {
      "command": "uvx",
      "args": ["--from", "thoughtlab-mcp", "thoughtlab-mcp"],
      "env": {
        "THOUGHTLAB_API_URL": "http://localhost:8000/api/v1",
        "THOUGHTLAB_API_KEY": "optional-api-key"
      }
    }
  }
}
```

### Implementation Steps

1. Create separate `mcp-server/` directory in repo
2. Implement thin wrapper tools that call ThoughtLab API
3. Publish to PyPI as `thoughtlab-mcp`
4. Document configuration for Claude Desktop / Cursor

---

## Chrome Extension

### Overview

A Chrome extension to capture web content and quickly add it to the knowledge graph.

### Features

| Feature | Description |
|---------|-------------|
| **Quick Capture** | Highlight text → Right-click → "Add to ThoughtLab" |
| **Page Source** | Save current page as Source node with metadata |
| **Selection → Observation** | Turn highlighted text into an Observation |
| **Context Menu** | Actions for different node types |
| **Popup** | Mini-dashboard with recent activity |
| **Sidebar** | Full graph view in side panel |

### Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Chrome Extension                                   │
│                                                                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────────┐ │
│  │  Content Script │  │ Background SW   │  │        Popup/Sidebar        │ │
│  │                 │  │                 │  │                             │ │
│  │ - Text select   │  │ - API calls     │  │ - Recent nodes              │ │
│  │ - Context menu  │  │ - Auth state    │  │ - Quick search              │ │
│  │ - Page metadata │  │ - Notifications │  │ - Mini graph view           │ │
│  └────────┬────────┘  └────────┬────────┘  └──────────────┬──────────────┘ │
│           │                    │                          │                 │
│           └────────────────────┼──────────────────────────┘                 │
│                                │                                            │
└────────────────────────────────┼────────────────────────────────────────────┘
                                 │
                                 ▼
                    ┌───────────────────────────────┐
                    │    ThoughtLab API             │
                    │    /api/v1/*                  │
                    └───────────────────────────────┘
```

### Extension Structure

```
chrome-extension/
├── manifest.json
├── src/
│   ├── background/
│   │   └── service-worker.ts    # Background script
│   ├── content/
│   │   └── content-script.ts    # Injected into pages
│   ├── popup/
│   │   ├── popup.html
│   │   └── popup.tsx            # React popup
│   ├── sidebar/
│   │   ├── sidebar.html
│   │   └── sidebar.tsx          # React sidebar panel
│   └── shared/
│       ├── api.ts               # ThoughtLab API client
│       └── types.ts             # Shared types
├── public/
│   └── icons/
├── vite.config.ts
└── package.json
```

### Context Menu Actions

```typescript
// src/background/service-worker.ts
chrome.contextMenus.create({
  id: "thoughtlab-add-observation",
  title: "Add as Observation",
  contexts: ["selection"],
});

chrome.contextMenus.create({
  id: "thoughtlab-add-source",
  title: "Save Page as Source",
  contexts: ["page"],
});

chrome.contextMenus.create({
  id: "thoughtlab-find-related",
  title: "Find Related in ThoughtLab",
  contexts: ["selection"],
});

chrome.contextMenus.onClicked.addListener(async (info, tab) => {
  switch (info.menuItemId) {
    case "thoughtlab-add-observation":
      await api.createObservation({
        text: info.selectionText,
        links: [{ url: tab.url, label: tab.title }],
      });
      break;
    case "thoughtlab-add-source":
      await api.createSource({
        title: tab.title,
        url: tab.url,
        content: await getPageContent(tab.id),
      });
      break;
    case "thoughtlab-find-related":
      const results = await api.searchSimilar(info.selectionText);
      showResultsPopup(results);
      break;
  }
});
```

### Implementation Steps

1. Create `chrome-extension/` directory
2. Set up Vite + React + TypeScript build
3. Implement background service worker
4. Implement content script for text selection
5. Build popup UI with recent activity
6. Add sidebar panel with graph view (optional)
7. Publish to Chrome Web Store

---

## Testing Strategy

### Unit Tests

```python
# tests/test_ai_config.py
def test_ai_config_defaults():
    config = AIConfig(openai_api_key="test-key")
    assert config.llm_model == "gpt-4o-mini"
    assert config.embedding_dimensions == 1536
    assert config.auto_create_threshold == 0.8

def test_ai_config_is_configured():
    config = AIConfig(openai_api_key="")
    assert not config.is_configured
    
    config = AIConfig(openai_api_key="sk-test")
    assert config.is_configured
```

### Integration Tests

```python
# tests/test_ai_workflow.py
@pytest.mark.integration
async def test_full_processing_workflow():
    """Test complete node processing with mocked OpenAI."""
    # Mock OpenAI responses
    with patch("app.ai.embeddings.OpenAIEmbeddings") as mock_embeddings:
        mock_embeddings.return_value.aembed_query.return_value = [0.1] * 1536
        
        result = await ai_workflow.process_node(
            node_id="test-123",
            node_type="Observation",
            content="Test observation about quantum computing",
            node_label="Test observation",
        )
        
        assert result.success
        assert result.embeddings_created == 1
```

### End-to-End Tests

```python
# tests/test_e2e_ai.py
@pytest.mark.e2e
@pytest.mark.skipif(not os.getenv("THOUGHTLAB_OPENAI_API_KEY"), reason="No API key")
async def test_real_embedding_generation():
    """Test with real OpenAI API (requires API key)."""
    config = AIConfig()
    manager = EmbeddingManager(config)
    
    embedding = await manager.embed_text("Hello world")
    
    assert len(embedding) == 1536
    assert all(isinstance(x, float) for x in embedding)
```

---

## References

- [LangChain Documentation](https://docs.langchain.com/)
- [LangChain Neo4j Integration](https://python.langchain.com/docs/integrations/vectorstores/neo4jvector)
- [LangGraph Functional API](https://langchain-ai.github.io/langgraph/concepts/low_level/)
- [OpenAI Embeddings](https://platform.openai.com/docs/guides/embeddings)
- [ARQ Documentation](https://arq-docs.helpmanual.io/)
- [Neo4j Vector Indexes](https://neo4j.com/docs/cypher-manual/current/indexes/semantic-indexes/vector-indexes/)
- [Model Context Protocol](https://modelcontextprotocol.io/)
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
- [Chrome Extension Development](https://developer.chrome.com/docs/extensions/)

