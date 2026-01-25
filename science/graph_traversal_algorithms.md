# Graph Traversal Algorithms for Knowledge Graph Analysis

**Date**: 2026-01-25
**Purpose**: Document algorithms for graph navigation, path finding, and structural analysis in ThoughtLab

---

## Problem Statement

While ThoughtLab currently supports basic graph visualization and relationship creation, it lacks:
1. **Advanced path finding** algorithms to discover connection chains
2. **Centrality analysis** to identify influential nodes
3. **Community detection** to find clusters of related concepts
4. **Multi-hop reasoning** capabilities for complex queries
5. **Performance-optimized traversal** for large graphs

This research establishes the scientific foundation for implementing these capabilities using Neo4j's Graph Data Science library and established algorithms.

---

## 1. Shortest Path Algorithms

### 1.1 Breadth-First Search (BFS)

**Concept**: Explore graph layer by layer, guaranteed shortest path in unweighted graphs

**Algorithm**:
```
BFS(start, target):
    queue = [(start, [start])]  # (node, path)
    visited = {start}

    while queue:
        node, path = queue.pop(0)

        if node == target:
            return path

        for neighbor in neighbors(node):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))

    return None  # No path found
```

**Complexity**:
- Time: O(V + E) where V = vertices, E = edges
- Space: O(V)

**Best For**:
- Unweighted graphs (all edges equal)
- Finding minimum hops between nodes
- Social network analysis (degrees of separation)

**Neo4j Implementation**:
```cypher
MATCH (start:Observation {id: $start_id}), (end:Observation {id: $end_id})
MATCH path = shortestPath((start)-[*]-(end))
RETURN path, length(path) as hops
LIMIT 1
```

**Applications in ThoughtLab**:
- Find shortest chain of relationships connecting two observations
- Calculate "degrees of separation" between concepts
- Discover minimal connection paths for citation chains

---

### 1.2 Dijkstra's Algorithm

**Concept**: Find shortest path in weighted graphs using priority queue

**Algorithm**:
```
Dijkstra(start, target):
    dist = {node: ∞ for node in graph}
    dist[start] = 0
    prev = {node: None for node in graph}
    pq = PriorityQueue()
    pq.put((0, start))

    while not pq.empty():
        current_dist, node = pq.get()

        if node == target:
            return reconstruct_path(prev, target)

        for neighbor, weight in neighbors(node):
            distance = current_dist + weight
            if distance < dist[neighbor]:
                dist[neighbor] = distance
                prev[neighbor] = node
                pq.put((distance, neighbor))

    return None  # No path found
```

**Complexity**:
- Time: O((V + E) log V) with binary heap
- Space: O(V)

**Key Requirements**:
- Edge weights must be non-negative
- Weights should represent meaningful distance/cost

**Weighting Strategies for ThoughtLab**:
```python
def relationship_weight(relationship):
    """Calculate weight based on relationship properties."""
    weights = {
        "SUPPORTS": 0.8,      # Strong connection = low weight
        "CONTRADICTS": 0.9,   # Contradiction = slightly higher weight
        "RELATES_TO": 1.0,    # General relation = baseline
        "DISCUSSES": 1.2,     # Indirect = higher weight
    }

    base_weight = weights.get(relationship.type, 1.0)

    # Adjust by confidence
    confidence = relationship.properties.get("confidence", 0.5)
    weight_adjustment = 1.0 - (confidence * 0.5)  # Higher confidence = lower weight

    return base_weight * weight_adjustment
```

**Neo4j Implementation**:
```cypher
MATCH (start:Observation {id: $start_id}), (end:Observation {id: $end_id})
MATCH path = shortestPath((start)-[*]-(end))
WITH path, relationships(path) as rels
WITH path,
     reduce(weight = 0, rel in rels | weight +
           CASE rel.type
             WHEN 'SUPPORTS' THEN 0.8
             WHEN 'CONTRADICTS' THEN 0.9
             WHEN 'RELATES_TO' THEN 1.0
             ELSE 1.2
           END * (1.0 - coalesce(rel.confidence, 0.5) * 0.5)) as total_weight
RETURN path, total_weight, length(path) as hops
ORDER BY total_weight ASC
LIMIT 5
```

**Applications in ThoughtLab**:
- Find strongest argument chains (high confidence = low weight)
- Discover optimal citation paths
- Identify argument weaknesses (contradictions as higher weight)

---

### 1.3 A* (A-Star) Algorithm

**Concept**: Dijkstra's with heuristic guidance for faster search

**Algorithm**:
```
A_star(start, target):
    g_score = {node: ∞ for node in graph}  # Cost from start
    f_score = {node: ∞ for node in graph}  # Estimated total cost
    g_score[start] = 0
    f_score[start] = heuristic(start, target)

    open_set = PriorityQueue()
    open_set.put((f_score[start], start))

    while not open_set.empty():
        current = open_set.get()[1]

        if current == target:
            return reconstruct_path(current)

        for neighbor, cost in neighbors(current):
            tentative_g = g_score[current] + cost

            if tentative_g < g_score[neighbor]:
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + heuristic(neighbor, target)
                open_set.put((f_score[neighbor], neighbor))

    return None
```

**Heuristic Functions for Knowledge Graphs**:
```python
def semantic_heuristic(node_a, node_b):
    """Heuristic based on semantic similarity (if embeddings available)."""
    emb_a = get_embedding(node_a)
    emb_b = get_embedding(node_b)
    similarity = cosine_similarity(emb_a, emb_b)
    return 1.0 - similarity  # Convert similarity to distance

def structural_heuristic(node_a, node_b):
    """Heuristic based on graph structure."""
    # If they share neighbors, likely closer
    common_neighbors = len(
        set(get_neighbors(node_a)) & set(get_neighbors(node_b))
    )
    return max(0, 1.0 - (common_neighbors * 0.1))

def hybrid_heuristic(node_a, node_b):
    """Combine semantic and structural heuristics."""
    return 0.7 * semantic_heuristic(node_a, node_b) + \
           0.3 * structural_heuristic(node_a, node_b)
```

**Complexity**:
- Time: O(E) in practice (much faster than Dijkstra's)
- Space: O(V)

**Requirements**:
- Heuristic must be admissible (never overestimates)
- Heuristic should be consistent

**Neo4j Implementation** (using APOC):
```cypher
MATCH (start:Observation {id: $start_id}), (end:Observation {id: $end_id})
CALL apoc.algo.aStar(start, end, '>', 'weight') YIELD path, weight
RETURN path, weight
LIMIT 5
```

**Applications in ThoughtLab**:
- Fast discovery of relevant connection paths
- Scalable search in large graphs
- Integration with semantic similarity for guided search

---

### 1.4 Bidirectional Search

**Concept**: Search from both start and target simultaneously

**Algorithm**:
```
BidirectionalSearch(start, target):
    forward_queue = [start]
    backward_queue = [target]
    forward_visited = {start: [start]}
    backward_visited = {target: [target]}

    while forward_queue and backward_queue:
        # Expand forward
        current = forward_queue.pop(0)
        for neighbor in neighbors(current):
            if neighbor not in forward_visited:
                forward_visited[neighbor] = forward_visited[current] + [neighbor]
                forward_queue.append(neighbor)

                if neighbor in backward_visited:
                    # Found meeting point
                    return forward_visited[neighbor] + backward_visited[neighbor][::-1][1:]

        # Expand backward (similar logic)
        current = backward_queue.pop(0)
        # ... symmetric expansion

    return None  # No path found
```

**Complexity**:
- Time: O(√E) for regular graphs (theoretical)
- Space: O(√V)

**Best For**:
- Large graphs where one-directional search is too slow
- When you have target constraints

**Applications in ThoughtLab**:
- Efficiently find connections between two specific concepts
- Bridge discovery in large citation networks

---

## 2. Centrality Algorithms

### 2.1 Degree Centrality

**Concept**: Count direct connections (simplest centrality measure)

**Formula**:
```
C_D(v) = degree(v) = |N(v)|
```

Where N(v) is the set of neighbors of v

**Types**:
- **In-degree**: Number of incoming relationships
- **Out-degree**: Number of outgoing relationships
- **Undirected**: Total relationships

**Neo4j Implementation**:
```cypher
MATCH (n:Observation)
RETURN n.id as node_id,
       size((n)-[]->()) as out_degree,
       size((n)<-[]-)() as in_degree,
       size((n)-[]-)() as total_degree
ORDER BY total_degree DESC
LIMIT 10
```

**Interpretation in ThoughtLab**:
- **High out-degree**: Observation that cites many other sources (argumentative hub)
- **High in-degree**: Observation that many others cite (influential/central)
- **High total degree**: Well-connected concept in the graph

**Applications**:
- Identify key observations in arguments
- Find over-cited vs under-cited sources
- Discover potential bottlenecks in argument structure

---

### 2.2 Betweenness Centrality

**Concept**: How often is a node on shortest paths between other nodes?

**Formula**:
```
C_B(v) = Σ_{s ≠ v ≠ t} (σ_{st}(v) / σ_{st})
```

Where:
- σ_{st} = total number of shortest paths from s to t
- σ_{st}(v) = number of shortest paths from s to t that pass through v

**Interpretation**:
- **High betweenness**: Node acts as bridge/connector
- **Low betweenness**: Node in isolated region or on periphery

**Neo4j Implementation** (GDS Library):
```cypher
CALL gds.betweenness.stream({
  nodeQuery: 'MATCH (n) RETURN id(n) as id',
  relationshipQuery: 'MATCH (n)-[r]->(m) RETURN id(n) as source, id(m) as target, type(r) as type'
})
YIELD nodeId, score
WITH gds.util.asNode(nodeId) as node, score
WHERE node:Observation OR node:Hypothesis
RETURN node.id, node.text, score
ORDER BY score DESC
LIMIT 10
```

**Complexity**:
- Time: O(V × E) for exact computation
- Approximate: O(V × log V + E) using Brandes' algorithm

**Applications in ThoughtLab**:
- **Bridge nodes**: Discover observations that connect different argument threads
- **Information flow**: Identify which concepts control information propagation
- **Vulnerability analysis**: Nodes whose removal would fragment the graph

**Research Finding**: Betweenness is particularly valuable in knowledge graphs as it identifies "connector" concepts that bridge different research areas or argument threads.

**Source**: [Graph Analytics in 2026](https://research.aimultiple.com/graph-analytics/)

---

### 2.3 Closeness Centrality

**Concept**: Average distance from a node to all other nodes

**Formula**:
```
C_C(v) = (n - 1) / Σ_{t ≠ v} d(v, t)
```

Where d(v, t) is shortest path distance

**Interpretation**:
- **High closeness**: Node can reach others quickly (influential)
- **Low closeness**: Node far from others (isolated)

**Neo4j Implementation**:
```cypher
CALL gds.closeness.stream({
  nodeQuery: 'MATCH (n) RETURN id(n) as id',
  relationshipQuery: 'MATCH (n)-[r]->(m) RETURN id(n) as source, id(m) as target'
})
YIELD nodeId, score
WITH gds.util.asNode(nodeId) as node, score
WHERE node:Observation
RETURN node.id, node.text, score
ORDER BY score DESC
LIMIT 10
```

**Applications in ThoughtLab**:
- **Accessible concepts**: Find observations that can reach many others quickly
- **Research starting points**: Identify good entry points for exploring the graph
- **Core vs peripheral**: Distinguish central ideas from edge cases

---

### 2.4 Eigenvector Centrality & PageRank

**Concept**: Importance based on connection to other important nodes

**Eigenvector Centrality Formula**:
```
λx_i = Σ_{j} A_{ij} x_j
```

Where A is adjacency matrix, λ is eigenvalue

**PageRank Formula** (damped version):
```
PR(p_i) = (1-d)/N + d * Σ_{j∈M(p_i)} PR(p_j)/L(p_j)
```

Where:
- d = damping factor (typically 0.85)
- N = total number of nodes
- L(p_j) = out-degree of page j

**Neo4j Implementation**:
```cypher
CALL gds.pageRank.stream({
  nodeQuery: 'MATCH (n) RETURN id(n) as id',
  relationshipQuery: 'MATCH (n)-[r]->(m) RETURN id(n) as source, id(m) as target',
  maxIterations: 20
})
YIELD nodeId, score
WITH gds.util.asNode(nodeId) as node, score
WHERE node:Observation OR node:Hypothesis
RETURN node.id, node.text, score
ORDER BY score DESC
LIMIT 10
```

**Parameters**:
- `maxIterations`: 20-50 typically sufficient
- `dampingFactor`: 0.85 (standard)
- `relationshipWeightProperty`: Use confidence as weight

**Applications in ThoughtLab**:
- **Influential observations**: Find observations that influence many others
- **Research impact**: Identify highly-cited sources/concepts
- **Argument strength**: Weighted by relationship confidence

**Research Finding**: PageRank is particularly effective for knowledge graphs as it captures the "endorsement" nature of citations and relationships.

---

## 3. Community Detection Algorithms

### 3.1 Louvain Modularity

**Concept**: Maximize modularity to find communities (dense subgraphs)

**Modularity Formula**:
```
Q = (1/(2m)) * Σ_{ij} [A_{ij} - (k_i * k_j)/(2m)] * δ(c_i, c_j)
```

Where:
- A_{ij} = adjacency matrix
- k_i = degree of node i
- m = total edges
- δ(c_i, c_j) = 1 if same community, 0 otherwise

**Algorithm**:
1. Start with each node in its own community
2. Move nodes between communities to maximize modularity
3. Aggregate graph and repeat
4. Iterate until no improvement

**Neo4j Implementation**:
```cypher
CALL gds.louvain.stream({
  nodeQuery: 'MATCH (n) RETURN id(n) as id',
  relationshipQuery: 'MATCH (n)-[r]->(m) RETURN id(n) as source, id(m) as target',
  includeIntermediateCommunities: true
})
YIELD nodeId, communityId, intermediateCommunityIds
WITH gds.util.asNode(nodeId) as node, communityId
WHERE node:Observation OR node:Hypothesis
RETURN communityId,
       count(node) as community_size,
       collect(node.text)[..5] as sample_nodes
ORDER BY community_size DESC
LIMIT 10
```

**Complexity**: O(E log V) in practice

**Applications in ThoughtLab**:
- **Research themes**: Discover clusters of related observations
- **Argument families**: Group similar or supporting arguments
- **Topic modeling**: Automatic categorization without labels

**Example Output**:
```
Community 1 (42 nodes):
- Quantum entanglement observations
- Related hypotheses about non-locality
- Supporting experimental sources

Community 2 (28 nodes):
- Biological mechanisms
- Neuroscience observations
- Cellular level hypotheses
```

---

### 3.2 Label Propagation

**Concept**: Nodes adopt label of majority neighbors (fast, scalable)

**Algorithm**:
1. Initialize each node with unique label
2. Update node label to most frequent label among neighbors
3. Repeat until convergence or max iterations

**Neo4j Implementation**:
```cypher
CALL gds.labelPropagation.stream({
  nodeQuery: 'MATCH (n) RETURN id(n) as id',
  relationshipQuery: 'MATCH (n)-[r]->(m) RETURN id(n) as source, id(m) as target',
  maxIterations: 10
})
YIELD nodeId, communityId
WITH gds.util.asNode(nodeId) as node, communityId
RETURN communityId, count(node) as size
ORDER BY size DESC
```

**Complexity**: O(E) - very fast

**Best For**:
- Large graphs where modularity optimization is too slow
- Real-time community detection
- Streaming graph updates

---

### 3.3 Strongly Connected Components (SCC)

**Concept**: Find subgraphs where every node is reachable from every other node

**Algorithm**: Kosaraju's or Tarjan's algorithm

**Neo4j Implementation**:
```cypher
CALL gds.wcc.stream({
  nodeQuery: 'MATCH (n) RETURN id(n) as id',
  relationshipQuery: 'MATCH (n)-[r]->(m) RETURN id(n) as source, id(m) as target'
})
YIELD nodeId, componentId
WITH componentId, collect(gds.util.asNode(nodeId)) as nodes
WHERE size(nodes) > 1
RETURN componentId, size(nodes) as component_size, nodes
ORDER BY component_size DESC
```

**Applications in ThoughtLab**:
- **Self-contained arguments**: Find argument loops (circular reasoning detection)
- **Research silos**: Isolated communities that don't cite outside work
- **Citation cycles**: Mutual citation networks

---

## 4. Path Analysis Algorithms

### 4.1 All-Pairs Shortest Paths

**Concept**: Compute shortest path between all node pairs (for analysis)

**Implementation** (Neo4j GDS):
```cypher
CALL gds.allShortestPaths.stream({
  nodeQuery: 'MATCH (n) RETURN id(n) as id',
  relationshipQuery: 'MATCH (n)-[r]->(m) RETURN id(n) as source, id(m) as target, r.confidence as weight',
  relationshipWeightProperty: 'weight'
})
YIELD sourceNodeIds, targetNodeIds, path, distance
WITH gds.util.asNode(sourceNodeIds[0]) as source,
     gds.util.asNode(targetNodeIds[0]) as target,
     distance
WHERE source:Observation AND target:Observation
RETURN source.id, target.id, distance
ORDER BY distance ASC
LIMIT 100
```

**Applications in ThoughtLab**:
- **Distance matrix**: Build similarity between concepts based on graph distance
- **Research landscapes**: Visualize conceptual distance between observations
- **Argument clustering**: Group concepts by their connectivity patterns

**Complexity Warning**: O(V²) memory usage - use sampling for large graphs

---

### 4.2 Random Walks

**Concept**: Simulate random traversal to discover graph structure

**Algorithm**:
```
RandomWalk(start_node, length=20):
    path = [start_node]
    for i in range(length):
        current = path[-1]
        neighbors = get_neighbors(current)
        if not neighbors:
            break
        next_node = random.choice(neighbors)
        path.append(next_node)
    return path
```

**Neo4j Implementation**:
```cypher
MATCH (start:Observation {id: $start_id})
CALL apoc.path.expandConfig(start, {
  relationshipFilter: '>',
  minLevel: 1,
  maxLevel: 10,
  limit: 100,
  terminatorNodes: [],
  sequence: []
}) YIELD path
RETURN nodes(path) as path_nodes, relationships(path) as path_rels
LIMIT 10
```

**Applications in ThoughtLab**:
- **Graph embeddings**: Generate training data for Node2Vec/DeepWalk
- **Exploratory search**: Discover related but non-obvious concepts
- **Graph sampling**: Generate representative subgraphs for analysis

**Research Finding**: Random walks are the foundation of modern graph embedding techniques (Node2Vec, DeepWalk, GraphSAGE).

---

## 5. Multi-Hop Reasoning & Query Expansion

### 5.1 Path Query Patterns

**Concept**: Find patterns of relationships over multiple hops

**Example Queries for ThoughtLab**:

**Support Chain Discovery**:
```cypher
MATCH path = (hypothesis:Hypothesis)-[*1..4]->(observation:Observation)
WHERE hypothesis.id = $hypothesis_id
AND ALL(r in relationships(path) WHERE r.type = 'SUPPORTS')
RETURN path, length(path) as hops
ORDER BY length(path) ASC
LIMIT 10
```

**Contradiction Detection**:
```cypher
MATCH (hypothesis:Hypothesis {id: $hypothesis_id})
MATCH (other:Hypothesis)-[r:CONTRADICTS]->(hypothesis)
OPTIONAL MATCH path = (other)-[*1..3]-(source:Source)
WHERE source.id IS NOT NULL
RETURN other, collect(distinct source) as supporting_sources
```

**Citation Network Analysis**:
```cypher
MATCH path = (source1:Source)-[*2..5]-(source2:Source)
WHERE source1.id = $source_id
AND ANY(r in relationships(path) WHERE r.type = 'CITES')
RETURN source2,
       length(path) as distance,
       count(path) as connection_strength
ORDER BY connection_strength DESC
LIMIT 10
```

### 5.2 Graph Pattern Matching

**Concept**: Find subgraphs matching specific patterns

**Cypher Pattern Examples**:

**Triangle Pattern** (3 connected nodes):
```cypher
MATCH (a)-[r1]->(b)-[r2]->(c)-[r3]->(a)
WHERE a:Observation AND b:Hypothesis AND c:Source
RETURN a, b, c, r1, r2, r3
```

**Star Pattern** (central node with many connections):
```cypher
MATCH (center)-[r]->(satellite)
WHERE center.id = $node_id
WITH center, count(r) as degree
WHERE degree > 5
RETURN center, collect(satellite) as satellites
```

**Chain Pattern** (linear progression):
```cypher
MATCH path = (start)-[*]->(end)
WHERE start:Observation AND end:Hypothesis
AND all(r in relationships(path) WHERE r.type = 'SUPPORTS')
WITH path, [n in nodes(path) | n.id] as node_ids
WHERE length(path) >= 3
RETURN path, node_ids
```

---

## 6. Performance Optimization

### 6.1 Indexing Strategy

**Essential Indexes for Traversal**:
```cypher
-- Node ID indexes (always needed)
CREATE INDEX node_id_index IF NOT EXISTS FOR (n) ON (n.id);

-- Label indexes for common traversals
CREATE INDEX observation_label_index IF NOT EXISTS FOR (o:Observation) ON (o.id);
CREATE INDEX hypothesis_label_index IF NOT EXISTS FOR (h:Hypothesis) ON (h.id);

-- Relationship type indexes (if using property-based traversal)
CREATE INDEX rel_type_index IF NOT EXISTS FOR ()-[r]-() ON (type(r));

-- Composite indexes for common query patterns
CREATE INDEX node_type_confidence_index IF NOT EXISTS
FOR (n) ON (n.type, n.confidence);
```

### 6.2 Query Optimization Patterns

**Bad Pattern (N+1 queries)**:
```cypher
-- ❌ DON'T: Multiple queries in application code
MATCH (n:Observation {id: $id})
RETURN n

-- Then for each relationship:
MATCH (n)-[r]->(m) WHERE n.id = $id RETURN r, m
```

**Good Pattern (Single query)**:
```cypher
-- ✅ DO: Single efficient query
MATCH (n:Observation {id: $id})-[r]->(m)
RETURN n, collect({rel: r, node: m}) as connections
```

### 6.3 Path Finding Performance Tips

**Limit Depth Early**:
```cypher
-- ✅ Efficient: Limit early
MATCH path = (start)-[*1..3]-(end)
WHERE start.id = $start_id AND end.id = $end_id
RETURN path

-- ❌ Inefficient: Search all paths then filter
MATCH path = (start)-[*]-(end)
WHERE start.id = $start_id AND end.id = $end_id
AND length(path) <= 3
RETURN path
```

**Use Relationship Direction**:
```cypher
-- ✅ Directional (faster)
MATCH (start)-[:SUPPORTS*1..3]->(end)

-- ❌ Bidirectional (slower)
MATCH (start)-[*1..3]-(end)
WHERE ANY(r in relationships(r) WHERE type(r) = 'SUPPORTS')
```

**Apply Labels Early**:
```cypher
-- ✅ Labeled start (uses index)
MATCH (start:Observation {id: $id})-[*1..3]-(end:Hypothesis)
RETURN start, end

-- ❌ Unlabeled (scans all nodes)
MATCH (start {id: $id})-[*1..3]-(end)
WHERE start:Observation AND end:Hypothesis
RETURN start, end
```

### 6.4 Memory Management for Large Graphs

**Sampling for Expensive Operations**:
```cypher
-- For betweenness centrality on large graphs
CALL gds.betweenness.sampled.stream({
  nodeQuery: 'MATCH (n) RETURN id(n) as id',
  relationshipQuery: 'MATCH (n)-[r]->(m) RETURN id(n) as source, id(m) as target',
  strategy: 'random',
  probability: 0.1  -- Sample 10% of source-target pairs
})
YIELD nodeId, score
RETURN gds.util.asNode(nodeId).id as node_id, score
ORDER BY score DESC
LIMIT 100
```

**Batch Processing**:
```cypher
-- Process large graphs in batches
UNWIND range(0, 9) as batch
CALL {
  WITH batch
  MATCH (n:Observation)
  WHERE id(n) % 10 = batch
  CALL gds.pageRank.stream({
    nodeIds: [id(n)],
    maxIterations: 5
  })
  RETURN n, score
}
RETURN n.id, score
```

---

## 7. Neo4j Graph Data Science (GDS) Library

### 7.1 Installation & Setup

**Enterprise Edition** (Required for GDS):
```yaml
# docker-compose.yml
neo4j:
  image: neo4j:5.13.0-enterprise
  environment:
    - NEO4J_ACCEPT_LICENSE_AGREEMENT=yes
    - NEO4J_AUTH=neo4j/research_graph_password
    - NEO4J_PLUGINS=["graph-data-science"]
  ports:
    - "7474:7474"
    - "7687:7687"
```

**Verify Installation**:
```cypher
RETURN gds.version() as version;
```

### 7.2 GDS Algorithm Categories

**Centrality**:
- `gds.betweenness.stream`
- `gds.closeness.stream`
- `gds.degree.stream`
- `gds.eigenvector.stream`
- `gds.pageRank.stream`

**Community Detection**:
- `gds.louvain.stream`
- `gds.labelPropagation.stream`
- `gds.wcc.stream` (weakly connected components)

**Path Finding**:
- `gds.shortestPath.stream`
- `gds.allShortestPaths.stream`
- `gds.deltaStepping.stream`

**Similarity**:
- `gds.similarity.cosine.stream`
- `gds.similarity.euclidean.stream`
- `gds.similarity.pearson.stream`

### 7.3 Memory Estimation

**Always estimate before running**:
```cypher
CALL gds.pageRank.stream.estimate({
  nodeQuery: 'MATCH (n) RETURN id(n) as id',
  relationshipQuery: 'MATCH (n)-[r]->(m) RETURN id(n) as source, id(m) as target'
})
YIELD requiredMemory, treeClassification
RETURN requiredMemory, treeClassification
```

**Results**:
- `requiredMemory`: Memory needed in bytes
- `treeClassification`: "Yes" if fits in memory, "No" if requires out-of-core

---

## 8. ThoughtLab-Specific Implementations

### 8.1 Argument Strength Analysis

**Combine Multiple Centrality Measures**:
```cypher
WITH $node_id as target_id
MATCH (n) WHERE n.id = target_id

// Calculate multiple centrality measures
CALL gds.pageRank.stream({
  nodeQuery: 'MATCH (n) RETURN id(n) as id',
  relationshipQuery: 'MATCH (n)-[r]->(m) RETURN id(n) as source, id(m) as target, r.confidence as weight',
  relationshipWeightProperty: 'weight'
})
YIELD nodeId, score as pagerank
WITH n, score as pagerank_score
WHERE gds.util.asNode(nodeId).id = n.id

CALL gds.betweenness.stream({
  nodeQuery: 'MATCH (n) RETURN id(n) as id',
  relationshipQuery: 'MATCH (n)-[r]->(m) RETURN id(n) as source, id(m) as target'
})
YIELD nodeId, score as betweenness
WITH n, pagerank_score, score as betweenness_score
WHERE gds.util.asNode(nodeId).id = n.id

CALL gds.degree.stream({
  nodeQuery: 'MATCH (n) RETURN id(n) as id',
  relationshipQuery: 'MATCH (n)-[r]->(m) RETURN id(n) as source, id(m) as target'
})
YIELD nodeId, score as degree
WITH n, pagerank_score, betweenness_score, score as degree_score
WHERE gds.util.asNode(nodeId).id = n.id

// Composite strength score
RETURN n.id as node_id,
       n.text as content,
       (pagerank_score * 0.4 + betweenness_score * 0.3 + degree_score * 0.3) as strength_score,
       pagerank_score, betweenness_score, degree_score
ORDER BY strength_score DESC
LIMIT 10
```

### 8.2 Research Theme Discovery

**Multi-level Community Detection**:
```cypher
CALL gds.louvain.stream({
  nodeQuery: 'MATCH (n:Observation) RETURN id(n) as id',
  relationshipQuery: 'MATCH (n)-[r]->(m) WHERE m:Observation OR m:Hypothesis RETURN id(n) as source, id(m) as target',
  includeIntermediateCommunities: true
})
YIELD nodeId, communityId, intermediateCommunityIds
WITH communityId,
     collect({
       id: gds.util.asNode(nodeId).id,
       text: gds.util.asNode(nodeId).text,
       type: labels(gds.util.asNode(nodeId))[0]
     }) as members
WHERE size(members) > 2  // Filter small communities
RETURN communityId,
       size(members) as community_size,
       members[..5] as sample_members,  // First 5 members
       [m in members | m.type] as types
ORDER BY community_size DESC
LIMIT 20
```

### 8.3 Connection Path Discovery

**Find Optimal Argument Paths**:
```cypher
WITH $hypothesis_id as hypothesis_id, $observation_id as obs_id
MATCH (h:Hypothesis {id: hypothesis_id}), (o:Observation {id: obs_id})

CALL gds.shortestPath.stream({
  sourceNode: h,
  targetNode: o,
  nodeQuery: 'MATCH (n) RETURN id(n) as id',
  relationshipQuery: 'MATCH (n)-[r]->(m) RETURN id(n) as source, id(m) as target, r.confidence as weight',
  relationshipWeightProperty: 'weight'
})
YIELD path, distance

WITH path, distance, relationships(path) as rels
RETURN {
  path: [n in nodes(path) | {id: n.id, text: n.text, type: labels(n)[0]}],
  distance: distance,
  avg_confidence: avg(r.confidence) as avg_confidence,
  min_confidence: min(r.confidence) as min_confidence,
  relationship_types: collect(distinct type(r)) as types
} as result
ORDER BY distance ASC, avg_confidence DESC
LIMIT 5
```

### 8.4 Influential Sources Detection

**Weighted PageRank with Recency**:
```cypher
CALL gds.pageRank.stream({
  nodeQuery: 'MATCH (n) RETURN id(n) as id',
  relationshipQuery: '''
    MATCH (n)-[r]->(m)
    WITH n, m, r
    WHERE r.created_at IS NOT NULL
    RETURN id(n) as source, id(m) as target,
           r.confidence * exp(-age_in_days(r.created_at) / 365) as weight
  ''',
  relationshipWeightProperty: 'weight',
  maxIterations: 20
})
YIELD nodeId, score
WITH gds.util.asNode(nodeId) as node, score
WHERE node:Source
RETURN node.id, node.title, score as influence_score
ORDER BY score DESC
LIMIT 20
```

---

## 9. Implementation Roadmap for ThoughtLab

### Phase 1: Basic Path Finding (Week 1-2)

**Implement**:
1. **Shortest path** between two nodes
2. **N-hop neighbors** (find nodes within N relationships)
3. **Relationship exploration** (get all connections)

**API Endpoints**:
```python
# GET /api/v1/graph/shortest-path?from=<id>&to=<id>
# GET /api/v1/graph/neighbors/<id>?depth=2
# GET /api/v1/graph/connections/<id>?types=SUPPORTS,CONTRADICTS
```

### Phase 2: Centrality Analysis (Month 1)

**Implement**:
1. **PageRank** for influential node detection
2. **Betweenness centrality** for bridge identification
3. **Degree analysis** for connection patterns

**API Endpoints**:
```python
# POST /api/v1/graph/analysis/centrality
# {
#   "algorithm": "pagerank",
#   "node_types": ["Observation", "Hypothesis"],
#   "limit": 50
# }
```

### Phase 3: Community Detection (Month 2)

**Implement**:
1. **Louvain communities** for topic clustering
2. **Connected components** for isolated groups
3. **Community summaries** with sample nodes

**API Endpoints**:
```python
# POST /api/v1/graph/analysis/communities
# {
#   "algorithm": "louvain",
#   "min_community_size": 5,
#   "include_intermediate": true
# }
```

### Phase 4: Advanced Analytics (Month 3)

**Implement**:
1. **Multi-hop queries** with filters
2. **Path recommendations** (optimal argument chains)
3. **Graph metrics dashboard** (summary statistics)
4. **Anomaly detection** (unusual connection patterns)

---

## 10. Validation & Testing

### 10.1 Algorithm Validation

**Performance Benchmarks**:
```python
def benchmark_traversal_algorithms(graph_size=10000):
    """Benchmark different traversal algorithms."""
    results = {}

    # Test shortest path algorithms
    for algo in ['dijkstra', 'astar', 'bidirectional']:
        time, memory, accuracy = measure_performance(algo, graph_size)
        results[algo] = {
            'time_ms': time,
            'memory_mb': memory,
            'accuracy': accuracy
        }

    return results

def measure_centrality_accuracy(graph, ground_truth=None):
    """Validate centrality measures against known important nodes."""
    # Compare against manual validation
    # Calculate precision@k, recall@k
    pass
```

### 10.2 Quality Metrics

**For Path Finding**:
- **Optimality**: Is returned path actually shortest?
- **Diversity**: Do we get multiple path options?
- **Relevance**: Are paths semantically meaningful?

**For Centrality**:
- **Interpretability**: Do high-centrality nodes make sense?
- **Stability**: Do results change significantly with small graph changes?
- **Correlation**: How do different centrality measures correlate?

**For Communities**:
- **Modularity**: Q > 0.3 indicates meaningful communities
- **Coherence**: Do community members share semantic similarity?
- **Stability**: Do communities persist across algorithm runs?

---

## 11. Key Academic Sources

### 11.1 Graph Algorithms & Theory
1. **"Network Science" by Albert-László Barabási** - Foundation of network analysis
2. **"Graph Theory and Applications"** - Classic algorithms reference
3. **"Centrality Measures in Networks"** (2016) - Comprehensive survey

### 11.2 Neo4j-Specific References
1. **Neo4j Graph Data Science Library Manual** - Official documentation
   - [Online](https://neo4j.com/docs/graph-data-science/current/)
2. **"Graph Algorithms" by Mark Needham & Amy Hodler** (O'Reilly)
   - Practical Neo4j algorithm implementations
3. **Neo4j GDS GitHub** - Examples and best practices
   - [GitHub](https://github.com/neo4j/graph-data-science)

### 11.3 Application-Specific Research
1. **"Knowledge Graph Analytics for Scholarly Communication"** (2020)
   - Academic citation network analysis
2. **"Community Detection in Bibliographic Networks"** (2018)
   - Research community discovery
3. **"Betweenness Centrality in Scientific Citation Networks"** (2019)
   - Bridge detection in research fields

---

## 12. Quick Start Guide

### Step 1: Install Neo4j Enterprise + GDS
```yaml
# Update docker-compose.yml
services:
  neo4j:
    image: neo4j:5.13.0-enterprise
    environment:
      - NEO4J_ACCEPT_LICENSE_AGREEMENT=yes
      - NEO4J_PLUGINS=["graph-data-science"]
```

### Step 2: Verify Installation
```cypher
RETURN gds.version() as gds_version,
       dbms.components() as neo4j_version
```

### Step 3: Create Essential Indexes
```cypher
CREATE INDEX node_id IF NOT EXISTS FOR (n) ON (n.id);
CREATE INDEX rel_type IF NOT EXISTS FOR ()-[r]-() ON (type(r));
```

### Step 4: Test Basic Algorithms
```cypher
-- Test betweenness centrality (small sample)
CALL gds.betweenness.stream({
  nodeQuery: 'MATCH (n) RETURN id(n) as id LIMIT 100',
  relationshipQuery: 'MATCH (n)-[r]->(m) RETURN id(n) as source, id(m) as target'
})
YIELD nodeId, score
RETURN gds.util.asNode(nodeId).id, score
ORDER BY score DESC
LIMIT 10
```

### Step 5: Monitor Memory Usage
```cypher
-- Always estimate before running large algorithms
CALL gds.pageRank.stream.estimate({
  nodeQuery: 'MATCH (n) RETURN id(n) as id',
  relationshipQuery: 'MATCH (n)-[r]->(m) RETURN id(n) as source, id(m) as target'
})
YIELD requiredMemory
RETURN requiredMemory
```

---

## 13. Decision Framework

### Algorithm Selection Guide

**For Path Finding**:
- **Small graphs (<10K nodes)**: Dijkstra's or A*
- **Large graphs (>100K nodes)**: Bidirectional or heuristic-guided search
- **Weighted paths**: Use relationship weights (confidence, recency)
- **Unweighted**: BFS is sufficient

**For Centrality**:
- **Overall importance**: PageRank (weighted by confidence)
- **Bridge detection**: Betweenness centrality
- **Quick estimates**: Degree centrality
- **Influence propagation**: Eigenvector centrality

**For Communities**:
- **Quality over speed**: Louvain modularity
- **Speed over quality**: Label propagation
- **Connected subgraphs**: Weakly Connected Components

**For Large Graphs**:
- **Sampling**: Use 10-20% samples for expensive algorithms
- **Batches**: Process in chunks when memory is limited
- **Approximation**: Use approximate algorithms where possible

---

## 14. Open Questions for ThoughtLab

### 14.1 Graph Characteristics
1. **Scale**: How many nodes/relationships expected?
2. **Dynamics**: How often does the graph change?
3. **Density**: Expected average degree per node?

### 14.2 Use Cases
1. **Real-time queries**: Do users expect instant results?
2. **Batch analysis**: Need for overnight processing?
3. **Interactive exploration**: Graph visualization requirements?

### 14.3 Performance Requirements
1. **Query latency**: What's acceptable for path finding? (<100ms? <1s?)
2. **Memory budget**: What's available for GDS algorithms?
3. **Update frequency**: How often to recalculate centrality/communities?

---

## 15. Summary of Recommendations

### Immediate Actions (Week 1-2)
1. **Upgrade to Neo4j 5.13+ Enterprise** (for GDS library)
2. **Implement shortest path API** using `gds.shortestPath.stream`
3. **Add basic centrality metrics** (PageRank, degree)

### Short-term (Month 1)
1. **Add betweenness centrality** for bridge detection
2. **Implement community detection** (Louvain)
3. **Build connection strength metrics** (weighted by confidence)

### Medium-term (Month 2-3)
1. **Multi-hop query interface** for complex exploration
2. **Graph metrics dashboard** showing overall statistics
3. **Path recommendation engine** for optimal argument chains

### Long-term (Month 4-6)
1. **Real-time graph analytics** (incremental updates)
2. **ML-enhanced traversal** (learned path costs)
3. **Advanced pattern matching** for research questions

---

## Expected Impact

**For Users**:
- **Discovery**: Find unexpected connections between concepts
- **Analysis**: Understand argument structure and influence
- **Exploration**: Navigate complex research topics efficiently

**For Research Quality**:
- **Bridge identification**: Discover interdisciplinary connections
- **Influence tracking**: Identify key observations/concepts
- **Argument validation**: Check for logical consistency

**For System Performance**:
- **Query speed**: 10-100× faster for complex traversals
- **Scalability**: Handle graphs with 100K+ nodes
- **Rich insights**: Provide analytics impossible with simple queries

---

**Next Document**: [Knowledge Graph Embedding Techniques](./graph_embedding_techniques.md)
**Previous Document**: [Semantic Similarity Algorithms](./semantic_similarity_algorithms.md)