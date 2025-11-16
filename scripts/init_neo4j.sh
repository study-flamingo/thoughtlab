#!/bin/bash
# Initialize Neo4j with indexes and constraints using cypher-shell

echo "Initializing Neo4j database..."

docker exec -it research-graph-neo4j cypher-shell -u neo4j -p research_graph_password <<EOF

// Unique constraints on IDs
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

// Text indexes
CREATE TEXT INDEX observation_text IF NOT EXISTS
FOR (o:Observation) ON o.text;

CREATE TEXT INDEX hypothesis_claim IF NOT EXISTS
FOR (h:Hypothesis) ON h.claim;

CREATE TEXT INDEX source_title IF NOT EXISTS
FOR (s:Source) ON s.title;

// Temporal indexes
CREATE INDEX observation_created IF NOT EXISTS
FOR (o:Observation) ON o.created_at;

CREATE INDEX hypothesis_created IF NOT EXISTS
FOR (h:Hypothesis) ON h.created_at;

CREATE INDEX source_created IF NOT EXISTS
FOR (s:Source) ON s.created_at;

// Verify
SHOW INDEXES;
SHOW CONSTRAINTS;

EOF

echo "Done!"
