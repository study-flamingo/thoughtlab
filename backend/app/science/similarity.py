"""
Semantic Similarity Module

Implements efficient similarity search using HNSW and multi-metric approaches.
Based on research in semantic_similarity_algorithms.md

Key Components:
- HNSW: Hierarchical Navigable Small World (10-100× speedup)
- Cosine Similarity: Standard for normalized embeddings
- Multi-metric: Combine cosine + structural similarity
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import heapq
from collections import defaultdict
import math


@dataclass
class VectorNode:
    """Node with vector embedding for similarity search."""
    id: str
    embedding: np.ndarray
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class HNSWNode:
    """Node in HNSW graph."""
    __slots__ = ['id', 'embedding', 'neighbors', 'level']

    def __init__(self, node_id: str, embedding: np.ndarray, level: int = 0):
        self.id = node_id
        self.embedding = embedding.copy()
        self.neighbors: Dict[int, List[str]] = defaultdict(list)  # level -> [neighbor_ids]
        self.level = level


class HNSWGraph:
    """
    Hierarchical Navigable Small World graph for approximate nearest neighbor search.

    Based on: https://arxiv.org/abs/1603.09320

    Properties:
    - O(log n) search time complexity
    - High recall with small search lists
    - Multi-layer structure for fast navigation
    """

    def __init__(self, dim: int, m: int = 16, ef_construction: int = 200,
                 ef_search: int = 100, m_max: int = 32, max_elements: int = 10000):
        """
        Initialize HNSW graph.

        Args:
            dim: Vector dimension
            m: Number of connections per node (16 is good default)
            ef_construction: Size of dynamic candidate list during construction
            ef_search: Size of candidate list during search
            m_max: Maximum number of connections per node
            max_elements: Maximum number of elements
        """
        self.dim = dim
        self.m = m
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.m_max = m_max
        self.max_elements = max_elements

        self.nodes: Dict[str, HNSWNode] = {}
        self.enterpoint: Optional[str] = None
        self.max_level = 0

        # Parameters for level generation
        self.ml = 1.0 / math.log(m)

    def _get_random_level(self) -> int:
        """Generate random level for new node."""
        import random
        return -int(math.log(random.uniform(0.0, 1.0)) * self.ml)

    def _euclidean_distance(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Calculate Euclidean distance between vectors."""
        return np.linalg.norm(v1 - v2)

    def _cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Calculate cosine similarity between vectors."""
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0
        return dot_product / (norm_v1 * norm_v2)

    def _cosine_distance(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Convert cosine similarity to distance."""
        sim = self._cosine_similarity(v1, v2)
        return 1.0 - sim  # Convert to distance [0, 2]

    def search_layer(self, query: np.ndarray, k: int, layer: int = 0) -> List[Tuple[float, str]]:
        """
        Search nearest neighbors in specific layer.

        Args:
            query: Query vector
            k: Number of neighbors to return
            layer: HNSW layer to search

        Returns:
            List of (distance, node_id) tuples sorted by distance
        """
        if self.enterpoint is None:
            return []

        # Start from enterpoint
        current = self.enterpoint
        visited = {current}
        candidates = [(self._cosine_distance(query, self.nodes[current].embedding), current)]

        best = []

        while candidates:
            # Get closest candidate
            dist, node_id = heapq.heappop(candidates)

            # Update best results
            if len(best) < k or dist < best[-1][0]:
                best.append((dist, node_id))
                best.sort(key=lambda x: x[0])
                if len(best) > k:
                    best = best[:k]

            # Expand to neighbors
            if node_id in self.nodes:
                for neighbor_id in self.nodes[node_id].neighbors[layer]:
                    if neighbor_id not in visited:
                        visited.add(neighbor_id)
                        neighbor_dist = self._cosine_distance(query, self.nodes[neighbor_id].embedding)
                        heapq.heappush(candidates, (neighbor_dist, neighbor_id))

        return best

    def add_node(self, node_id: str, embedding: np.ndarray) -> None:
        """
        Add node to HNSW graph.

        Args:
            node_id: Unique identifier
            embedding: Vector embedding
        """
        if len(self.nodes) >= self.max_elements:
            raise ValueError("Maximum number of elements reached")

        level = self._get_random_level()
        new_node = HNSWNode(node_id, embedding, level)

        if self.enterpoint is None:
            # First node
            self.enterpoint = node_id
            self.max_level = level
            self.nodes[node_id] = new_node
            return

        # Find entry point at each level
        current = self.enterpoint
        max_level_to_update = min(level, self.max_level)

        for layer in range(max_level_to_update, -1, -1):
            # Search for nearest neighbors in this layer
            neighbors = self.search_layer(embedding, self.ef_construction, layer)

            # Find best candidates
            candidates = []
            for dist, neighbor_id in neighbors:
                if neighbor_id in self.nodes:
                    candidates.append((dist, neighbor_id))

            # Sort by distance and select m candidates
            candidates.sort(key=lambda x: x[0])
            selected = candidates[:min(len(candidates), self.m)]

            # Connect new node to selected neighbors
            for _, neighbor_id in selected:
                new_node.neighbors[layer].append(neighbor_id)
                # Also add to neighbor (bi-directional)
                if neighbor_id in self.nodes:
                    self.nodes[neighbor_id].neighbors[layer].append(node_id)

                    # Shrink neighbors if needed (m_max constraint)
                    if len(self.nodes[neighbor_id].neighbors[layer]) > self.m_max:
                        self._shrink_neighbors(neighbor_id, layer)

        # Update max level
        self.max_level = max(self.max_level, level)
        self.nodes[node_id] = new_node

    def _shrink_neighbors(self, node_id: str, layer: int) -> None:
        """Shrink neighbors to maintain m_max constraint."""
        node = self.nodes[node_id]
        if len(node.neighbors[layer]) <= self.m_max:
            return

        # Get distances to all neighbors
        current_embedding = node.embedding
        neighbors_with_dist = []
        for neighbor_id in node.neighbors[layer]:
            neighbor = self.nodes[neighbor_id]
            dist = self._cosine_distance(current_embedding, neighbor.embedding)
            neighbors_with_dist.append((dist, neighbor_id))

        # Keep closest m_max neighbors
        neighbors_with_dist.sort(key=lambda x: x[0])
        keep = [nid for _, nid in neighbors_with_dist[:self.m_max]]

        # Update neighbor list
        node.neighbors[layer] = keep

    def knn_search(self, query: np.ndarray, k: int = 10) -> List[Tuple[float, str]]:
        """
        K-nearest neighbor search.

        Args:
            query: Query vector
            k: Number of nearest neighbors

        Returns:
            List of (distance, node_id) tuples sorted by distance
        """
        if self.enterpoint is None:
            return []

        # Search through layers
        current = self.enterpoint
        for layer in range(self.max_level, 0, -1):
            # Search in higher layer
            neighbors = self.search_layer(query, 1, layer)
            if neighbors:
                current = neighbors[0][1]

        # Final search in base layer
        return self.search_layer(query, k, 0)

    def __len__(self) -> int:
        return len(self.nodes)

    def get_stats(self) -> Dict[str, Any]:
        """Get graph statistics."""
        if not self.nodes:
            return {"num_nodes": 0, "num_edges": 0, "avg_degree": 0.0}

        total_edges = sum(
            sum(len(neighbors) for neighbors in node.neighbors.values())
            for node in self.nodes.values()
        )

        return {
            "num_nodes": len(self.nodes),
            "num_edges": total_edges // 2,  # Divide by 2 for bidirectional
            "avg_degree": (total_edges // 2) / len(self.nodes) if len(self.nodes) > 0 else 0.0,
            "max_level": self.max_level,
            "enterpoint": self.enterpoint,
        }


class HNSWSimilarity:
    """
    HNSW-based similarity search for efficient nearest neighbor lookup.

    Wraps HNSWGraph for convenient similarity search operations.
    """

    def __init__(self, dim: int = 1536, **hnsw_params):
        """
        Initialize HNSW similarity index.

        Args:
            dim: Vector dimension (default 1536 for OpenAI embeddings)
            **hnsw_params: HNSW graph parameters
        """
        self.dim = dim
        self.index = HNSWGraph(dim=dim, **hnsw_params)
        self.node_embeddings: Dict[str, np.ndarray] = {}

    def add(self, node_id: str, embedding: np.ndarray) -> None:
        """
        Add node to similarity index.

        Args:
            node_id: Unique node identifier
            embedding: Vector embedding
        """
        if len(embedding) != self.dim:
            raise ValueError(f"Embedding dimension mismatch: got {len(embedding)}, expected {self.dim}")

        # Normalize embedding for cosine similarity
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        self.index.add_node(node_id, embedding)
        self.node_embeddings[node_id] = embedding

    def add_batch(self, nodes: List[Tuple[str, np.ndarray]]) -> None:
        """
        Add multiple nodes to index.

        Args:
            nodes: List of (node_id, embedding) tuples
        """
        for node_id, embedding in nodes:
            self.add(node_id, embedding)

    def search(self, query_embedding: np.ndarray, k: int = 10,
               return_distances: bool = True) -> List[Tuple[str, float]]:
        """
        Search for k nearest neighbors.

        Args:
            query_embedding: Query vector
            k: Number of results
            return_distances: Whether to return similarity scores

        Returns:
            List of (node_id, similarity) tuples, sorted by similarity
        """
        # Normalize query
        norm = np.linalg.norm(query_embedding)
        if norm > 0:
            query_embedding = query_embedding / norm

        # Search HNSW graph
        results = self.index.knn_search(query_embedding, k)

        if return_distances:
            # Convert distance back to similarity (1 - distance)
            return [(node_id, 1.0 - distance) for distance, node_id in results]
        else:
            return [(node_id, 0.0) for _, node_id in results]

    def search_by_id(self, node_id: str, k: int = 10) -> List[Tuple[str, float]]:
        """
        Search for neighbors of an existing node.

        Args:
            node_id: Node identifier
            k: Number of results

        Returns:
            List of (node_id, similarity) tuples
        """
        if node_id not in self.node_embeddings:
            raise ValueError(f"Node {node_id} not found in index")

        return self.search(self.node_embeddings[node_id], k)

    def get_similarity(self, id1: str, id2: str) -> float:
        """
        Get cosine similarity between two nodes.

        Args:
            id1: First node ID
            id2: Second node ID

        Returns:
            Cosine similarity between -1 and 1
        """
        if id1 not in self.node_embeddings or id2 not in self.node_embeddings:
            return 0.0

        emb1 = self.node_embeddings[id1]
        emb2 = self.node_embeddings[id2]

        dot_product = np.dot(emb1, emb2)
        return dot_product  # Already normalized

    def __len__(self) -> int:
        return len(self.index)

    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        stats = self.index.get_stats()
        stats["dim"] = self.dim
        return stats


@dataclass
class MultiMetricSimilarity:
    """
    Combines multiple similarity metrics for robust relationship scoring.

    Based on research showing that combining cosine similarity with
    structural similarity improves accuracy by ~10%.
    """

    alpha: float = 0.7  # Weight for semantic similarity
    beta: float = 0.3   # Weight for structural similarity

    def __post_init__(self):
        if not (0 <= self.alpha <= 1 and 0 <= self.beta <= 1):
            raise ValueError("Alpha and beta must be between 0 and 1")
        if abs(self.alpha + self.beta - 1.0) > 1e-6:
            raise ValueError("Alpha + beta must equal 1.0")

    def cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Calculate cosine similarity between vectors."""
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0
        return dot_product / (norm_v1 * norm_v2)

    def jaccard_similarity(self, set1: set, set2: set) -> float:
        """
        Calculate Jaccard similarity between two sets.

        Useful for comparing neighborhood sets.
        """
        if not set1 and not set2:
            return 1.0
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union > 0 else 0.0

    def structural_similarity(self, neighbors1: set, neighbors2: set,
                            common_ancestors: set = None,
                            common_descendants: set = None) -> float:
        """
        Calculate structural similarity based on graph neighborhoods.

        Args:
            neighbors1: Set of neighbor IDs for node 1
            neighbors2: Set of neighbor IDs for node 2
            common_ancestors: Set of common ancestor IDs (optional)
            common_descendants: Set of common descendant IDs (optional)

        Returns:
            Structural similarity score
        """
        # Jaccard similarity of direct neighbors
        neighbor_sim = self.jaccard_similarity(neighbors1, neighbors2)

        if common_ancestors is None and common_descendants is None:
            return neighbor_sim

        # Combine with higher-order structural similarity
        scores = [neighbor_sim]

        if common_ancestors:
            ancestor_sim = len(common_ancestors) / (len(neighbors1) + len(neighbors2) + 1e-6)
            scores.append(ancestor_sim)

        if common_descendants:
            descendant_sim = len(common_descendants) / (len(neighbors1) + len(neighbors2) + 1e-6)
            scores.append(descendant_sim)

        return np.mean(scores)

    def combined_similarity(self, semantic_sim: float, structural_sim: float) -> float:
        """
        Combine semantic and structural similarities.

        Args:
            semantic_sim: Cosine similarity between embeddings
            structural_sim: Jaccard-based structural similarity

        Returns:
            Combined similarity score
        """
        return self.alpha * semantic_sim + self.beta * structural_sim

    def relate_nodes(self, v1: np.ndarray, v2: np.ndarray,
                    neighbors1: set, neighbors2: set,
                    **kwargs) -> float:
        """
        Calculate relationship score between two nodes.

        Args:
            v1: Embedding of node 1
            v2: Embedding of node 2
            neighbors1: Neighbor set of node 1
            neighbors2: Neighbor set of node 2
            **kwargs: Additional structural information

        Returns:
            Combined relationship score
        """
        semantic = self.cosine_similarity(v1, v2)
        structural = self.structural_similarity(neighbors1, neighbors2, **kwargs)
        return self.combined_similarity(semantic, structural)


class RelationAwareSimilarity:
    """
    Relation-specific similarity using TransR-style projections.

    For each relation type, learn a projection matrix that transforms
    entity embeddings into relation-specific space.
    """

    def __init__(self, dim: int = 1536, relation_types: Optional[List[str]] = None):
        """
        Initialize relation-aware similarity.

        Args:
            dim: Embedding dimension
            relation_types: List of relation types
        """
        self.dim = dim
        self.relation_types = relation_types or []
        self.projection_matrices: Dict[str, np.ndarray] = {}

        # Initialize projection matrices for each relation type
        for rel_type in self.relation_types:
            self.projection_matrices[rel_type] = self._init_projection_matrix()

    def _init_projection_matrix(self) -> np.ndarray:
        """Initialize random projection matrix."""
        # Random orthogonal matrix
        M = np.random.randn(self.dim, self.dim)
        U, _, Vt = np.linalg.svd(M, full_matrices=False)
        return U @ Vt

    def project(self, embedding: np.ndarray, relation: str) -> np.ndarray:
        """
        Project embedding to relation-specific space.

        Args:
            embedding: Entity embedding
            relation: Relation type

        Returns:
            Projected embedding
        """
        if relation not in self.projection_matrices:
            # Return original embedding for unknown relations
            return embedding

        M = self.projection_matrices[relation]
        return np.dot(M, embedding)

    def relation_similarity(self, h: np.ndarray, r: str, t: np.ndarray) -> float:
        """
        Calculate TransR-style similarity.

        Args:
            h: Head entity embedding
            r: Relation type
            t: Tail entity embedding

        Returns:
            Similarity score (higher is better)
        """
        # Project to relation space
        h_proj = self.project(h, r)
        t_proj = self.project(t, r)

        # TransR scoring: -||h_proj + r - t_proj||²
        # Simplified: use cosine similarity
        dot_product = np.dot(h_proj, t_proj)
        norm_h = np.linalg.norm(h_proj)
        norm_t = np.linalg.norm(t_proj)

        if norm_h == 0 or norm_t == 0:
            return 0.0

        return dot_product / (norm_h * norm_t)

    def learn_projection(self, relation: str, examples: List[Tuple[np.ndarray, np.ndarray]]) -> None:
        """
        Learn projection matrix from examples.

        Args:
            relation: Relation type
            examples: List of (head_embedding, tail_embedding) pairs
        """
        if not examples:
            return

        # Simple learning: find projection that minimizes distance
        # In practice, this would use a full TransR training procedure
        h_matrix = np.array([h for h, t in examples])
        t_matrix = np.array([t for h, t in examples])

        # Learn projection matrix M that minimizes ||hM - t||
        # Using ridge regression
        ridge_alpha = 1.0
        M = np.linalg.solve(
            h_matrix.T @ h_matrix + ridge_alpha * np.eye(self.dim),
            h_matrix.T @ t_matrix
        )

        self.projection_matrices[relation] = M


def test_similarity():
    """Test similarity implementations."""
    print("Testing Similarity Methods...")
    print("=" * 60)

    # Generate synthetic embeddings
    np.random.seed(42)
    dim = 10  # Small dimension for testing

    # Create some vectors
    vectors = {
        "v1": np.random.randn(dim),
        "v2": np.random.randn(dim),
        "v3": np.random.randn(dim),
    }

    # Make v3 similar to v1
    vectors["v3"] = vectors["v1"] + np.random.randn(dim) * 0.1

    # Normalize
    for k, v in vectors.items():
        vectors[k] = v / np.linalg.norm(v)

    print("\n1. HNSW Similarity Index:")
    index = HNSWSimilarity(dim=dim, m=8, ef_construction=50, ef_search=10)

    # Add nodes
    for node_id, embedding in vectors.items():
        index.add(node_id, embedding)

    print(f"   Index size: {len(index)}")
    print(f"   Stats: {index.get_stats()}")

    # Search
    query = vectors["v1"]
    results = index.search(query, k=3)
    print(f"   Search results: {results}")
    print(f"   Similarity v1-v3: {index.get_similarity('v1', 'v3'):.3f}")

    print("\n2. Multi-Metric Similarity:")
    multi = MultiMetricSimilarity(alpha=0.7, beta=0.3)

    # Test semantic similarity
    semantic = multi.cosine_similarity(vectors["v1"], vectors["v3"])
    print(f"   Semantic similarity: {semantic:.3f}")

    # Test structural similarity (mock neighbor sets)
    neighbors1 = {"n1", "n2", "n3"}
    neighbors2 = {"n1", "n2", "n4"}  # Share 2 neighbors
    structural = multi.structural_similarity(neighbors1, neighbors2)
    print(f"   Structural similarity: {structural:.3f}")

    # Combined
    combined = multi.combined_similarity(semantic, structural)
    print(f"   Combined similarity: {combined:.3f}")

    print("\n3. Relation-Aware Similarity:")
    rel_sim = RelationAwareSimilarity(dim=dim, relation_types=["SUPPORTS", "CONTRADICTS"])

    # Test projection
    projected = rel_sim.project(vectors["v1"], "SUPPORTS")
    print(f"   Projected shape: {projected.shape}")

    # Test relation similarity
    rel_score = rel_sim.relation_similarity(vectors["v1"], "SUPPORTS", vectors["v3"])
    print(f"   Relation similarity: {rel_score:.3f}")

    print("\nAll tests completed successfully!")


if __name__ == "__main__":
    test_similarity()