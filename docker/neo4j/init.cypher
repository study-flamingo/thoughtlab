// =============================================================================
// Neo4j Initialization Script for ThoughtLab
// =============================================================================
// Run this in Neo4j Browser (http://localhost:7474) after first startup
// Or via: cat init.cypher | cypher-shell -u neo4j -p <password>
// =============================================================================

// -----------------------------------------------------------------------------
// SECTION 1: Node Constraints (Unique IDs)
// -----------------------------------------------------------------------------

// Primary knowledge node types
CREATE CONSTRAINT observation_id IF NOT EXISTS
FOR (o:Observation) REQUIRE o.id IS UNIQUE;

CREATE CONSTRAINT hypothesis_id IF NOT EXISTS
FOR (h:Hypothesis) REQUIRE h.id IS UNIQUE;

CREATE CONSTRAINT source_id IF NOT EXISTS
FOR (s:Source) REQUIRE s.id IS UNIQUE;

CREATE CONSTRAINT concept_id IF NOT EXISTS
FOR (c:Concept) REQUIRE c.id IS UNIQUE;

CREATE CONSTRAINT entity_id IF NOT EXISTS
FOR (e:Entity) REQUIRE e.id IS UNIQUE;

// Chunk nodes for long-form content (linked to parent Source nodes)
CREATE CONSTRAINT chunk_id IF NOT EXISTS
FOR (ch:Chunk) REQUIRE ch.id IS UNIQUE;

// System user for LLM-created content
CREATE CONSTRAINT user_id IF NOT EXISTS
FOR (u:User) REQUIRE u.id IS UNIQUE;

// Activity feed items
CREATE CONSTRAINT activity_id IF NOT EXISTS
FOR (a:Activity) REQUIRE a.id IS UNIQUE;

// -----------------------------------------------------------------------------
// SECTION 2: Text Indexes (for keyword search)
// -----------------------------------------------------------------------------

CREATE TEXT INDEX observation_text IF NOT EXISTS
FOR (o:Observation) ON o.text;

CREATE TEXT INDEX hypothesis_claim IF NOT EXISTS
FOR (h:Hypothesis) ON h.claim;

CREATE TEXT INDEX source_title IF NOT EXISTS
FOR (s:Source) ON s.title;

CREATE TEXT INDEX concept_name IF NOT EXISTS
FOR (c:Concept) ON c.name;

CREATE TEXT INDEX entity_name IF NOT EXISTS
FOR (e:Entity) ON e.name;

CREATE TEXT INDEX chunk_content IF NOT EXISTS
FOR (ch:Chunk) ON ch.content;

// -----------------------------------------------------------------------------
// SECTION 3: Temporal Indexes (for time-based queries)
// -----------------------------------------------------------------------------

CREATE INDEX observation_created IF NOT EXISTS
FOR (o:Observation) ON o.created_at;

CREATE INDEX hypothesis_created IF NOT EXISTS
FOR (h:Hypothesis) ON h.created_at;

CREATE INDEX source_created IF NOT EXISTS
FOR (s:Source) ON s.created_at;

CREATE INDEX concept_created IF NOT EXISTS
FOR (c:Concept) ON c.created_at;

CREATE INDEX entity_created IF NOT EXISTS
FOR (e:Entity) ON e.created_at;

CREATE INDEX chunk_created IF NOT EXISTS
FOR (ch:Chunk) ON ch.created_at;

// Activity feed temporal + filtering indexes
CREATE INDEX activity_created IF NOT EXISTS
FOR (a:Activity) ON a.created_at;

CREATE INDEX activity_type IF NOT EXISTS
FOR (a:Activity) ON a.type;

CREATE INDEX activity_status IF NOT EXISTS
FOR (a:Activity) ON a.status;

CREATE INDEX activity_group IF NOT EXISTS
FOR (a:Activity) ON a.group_id;

CREATE INDEX activity_node IF NOT EXISTS
FOR (a:Activity) ON a.node_id;

// -----------------------------------------------------------------------------
// SECTION 4: Relationship Property Indexes
// -----------------------------------------------------------------------------

// Index for filtering by relationship creator
CREATE INDEX rel_created_by IF NOT EXISTS
FOR ()-[r]-() ON r.created_by;

// Index for finding unapproved/pending relationships
CREATE INDEX rel_approved IF NOT EXISTS
FOR ()-[r]-() ON r.approved;

// Index for relationship IDs (required for lookup/update)
// Note: Need separate indexes per relationship type in Neo4j
// These cover the suggested common types
CREATE INDEX rel_supports_id IF NOT EXISTS FOR ()-[r:SUPPORTS]-() ON (r.id);
CREATE INDEX rel_contradicts_id IF NOT EXISTS FOR ()-[r:CONTRADICTS]-() ON (r.id);
CREATE INDEX rel_relates_to_id IF NOT EXISTS FOR ()-[r:RELATES_TO]-() ON (r.id);
CREATE INDEX rel_cites_id IF NOT EXISTS FOR ()-[r:CITES]-() ON (r.id);
CREATE INDEX rel_derived_from_id IF NOT EXISTS FOR ()-[r:DERIVED_FROM]-() ON (r.id);
CREATE INDEX rel_has_chunk_id IF NOT EXISTS FOR ()-[r:HAS_CHUNK]-() ON (r.id);

// -----------------------------------------------------------------------------
// SECTION 5: Full-Text Search Index (cross-node semantic search)
// -----------------------------------------------------------------------------

// Composite full-text index for natural language queries
// Searches across all primary text fields
CREATE FULLTEXT INDEX knowledge_search IF NOT EXISTS
FOR (n:Observation|Hypothesis|Source|Concept|Entity)
ON EACH [n.text, n.claim, n.title, n.name, n.description, n.content];

// -----------------------------------------------------------------------------
// SECTION 6: Vector Indexes (for semantic similarity search)
// -----------------------------------------------------------------------------
// Dimensions: 1536 for OpenAI text-embedding-3-small
// These indexes enable the AI workflow to find similar content

CREATE VECTOR INDEX observation_embedding IF NOT EXISTS
FOR (o:Observation) ON o.embedding
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

CREATE VECTOR INDEX source_embedding IF NOT EXISTS
FOR (s:Source) ON s.embedding
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

// Unified vector index for querying across all node types
// This is used by the AI workflow's similarity search
// Note: Neo4j 5.11+ supports this syntax
CREATE VECTOR INDEX node_embedding IF NOT EXISTS
FOR (n:Observation|Hypothesis|Source|Concept|Entity|Chunk) ON n.embedding
OPTIONS {indexConfig: {
  `vector.dimensions`: 1536,
  `vector.similarity_function`: 'cosine'
}};

// -----------------------------------------------------------------------------
// SECTION 7: System Data (LLM User Account)
// -----------------------------------------------------------------------------

// Create system user for LLM-generated content
MERGE (u:User {id: 'system-llm'})
SET u.name = 'AI Assistant',
    u.type = 'system',
    u.created_at = datetime()
RETURN u;

// -----------------------------------------------------------------------------
// SECTION 8: Reference - Suggested Relationship Types
// -----------------------------------------------------------------------------
// Relationship types are open strings (not enum-constrained).
// The LLM can create any relationship type, but these are suggested defaults:
//
// SUPPORTS        - Evidence supports a hypothesis/claim
// CONTRADICTS     - Evidence contradicts a hypothesis/claim
// RELATES_TO      - General semantic relationship
// CITES           - Source A cites Source B
// DERIVED_FROM    - Node B was derived/extracted from Node A
// OBSERVED_IN     - Observation was made in Source
// DISCUSSES       - Source discusses Concept/Entity
// INSPIRED_BY     - Weaker form of DERIVED_FROM
// PRECEDES        - Temporal ordering
// CAUSES          - Causal relationship
// PART_OF         - Hierarchical/compositional relationship
// SIMILAR_TO      - Semantic similarity (often auto-generated)
// HAS_CHUNK       - Source contains Chunk (for long content)
//
// Relationship properties (all optional):
// - id: string (UUID, required for update/delete)
// - confidence: float 0-1 (how confident is this relationship?)
// - created_by: string (user ID or 'system-llm')
// - created_at: datetime
// - approved: boolean (has user reviewed LLM suggestion?)
// - feedback_score: float 0-1 (user rating of suggestion quality)
// - notes: string (explanation or context)

// -----------------------------------------------------------------------------
// Verification
// -----------------------------------------------------------------------------

SHOW INDEXES;
SHOW CONSTRAINTS;
