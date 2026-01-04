# Relationship Confidence Scoring Algorithms

Research findings on established algorithms for calculating confidence scores for knowledge graph relationships.

**Date**: 2026-01-03
**Purpose**: Inform ThoughtLab's relationship confidence scoring implementation

---

## Problem Statement

Simple semantic similarity is insufficient for relationship confidence scoring. We need algorithms that factor in:
- Semantic similarity between entities
- Relationship type (which may be arbitrary/user-defined)
- Graph structure and context

---

## Established Algorithms

### 1. Translation-Based Embeddings (TransE/TransR Family)

Foundational algorithms that model relationships as translations in vector space.

#### TransE (Translating Embeddings)

**Core Idea**: Represent entities and relations as vectors where relationships act as translations.

**Scoring Function**:
```
score(h, r, t) = ||h + r - t||
```
Using L1 or L2 norm.

**Characteristics**:
- Simple and efficient
- Struggles with one-to-many relationships
- All entities and relations in same vector space

**Source**: [TransE Overview](https://www.emergentmind.com/topics/transe)

#### TransR (Translation in Relation Space)

**Core Idea**: Project entities into relation-specific spaces before computing translation.

**Scoring Function**:
```
score(h, r, t) = ||h_r + r - t_r||²

where:
  h_r = h × M_r  (head entity projected into relation space)
  t_r = t × M_r  (tail entity projected into relation space)
  M_r = projection matrix for relation r
```

**Key Advantages**:
- Different relations focus on different aspects of entities
- Better handles complex relationship types
- Learns relation-specific projection matrices
- Outperforms TransE and TransH consistently

**Source**: [TransR Paper (AAAI 2015)](https://linyankai.github.io/publications/aaai2015_transr.pdf)

#### CTransR (Clustered TransR)

Extension that clusters entity pairs within each relation to model implicit subtypes.

---

### 2. UKGE (Uncertain Knowledge Graph Embedding)

Specifically designed for confidence score prediction.

**Approach**:
- Uses **Probabilistic Soft Logic (PSL)** framework
- Infers confidence scores for unseen relational triples
- Maps scoring function results to confidence via mapping functions

**Mapping Functions**:
1. **Logistic Function**: `σ(x) = 1 / (1 + e^(-x))`
2. **Bounded Rectifier**: Clamps output to [0, 1]

**Source**: [Embedding Uncertain Knowledge Graphs (UCLA)](https://web.cs.ucla.edu/~yzsun/papers/2019_AAAI_UKG.pdf)

---

### 3. SUKE (Structure-aware Uncertain KG Embedding)

Improves upon UKGE by better utilizing structural information.

**Two-Component Architecture**:

1. **Evaluator**: Computes two scores per fact
   - Structure score (graph topology)
   - Uncertainty score (confidence measure)

2. **Confidence Generator**:
   - Produces confidence from uncertainty scores
   - Uses DistMult plausibility measure

---

### 4. UKGSE (Transformer-based Subgraph Structure Embedding)

**Core Idea**: Encode neighborhood structure around entities using Transformers.

**Approach**:
1. Encode neighbors of head entity
2. Encode neighbors of tail entity
3. Model interaction information between neighborhoods
4. Predict confidence from structural embeddings

**Key Insight**: Triples with similar neighborhood structures should have similar confidence scores.

**Source**: [IOP Science Paper](https://iopscience.iop.org/article/10.1088/1742-6596/2833/1/012001/pdf)

---

### 5. CKRL (Confidence-aware Knowledge Representation Learning)

**Hierarchical Confidence Integration**:

```
E(T) = Σ E(h,r,t) · C(h,r,t)
```

**Confidence Components**:
- **Local Triple Confidence**: Direct triple plausibility
- **Global Path Confidence**: Multi-hop path support
- **Prior Path Confidence**: Historical path patterns
- **Adaptive Path Confidence**: Context-dependent weighting

---

### 6. NIC (Neighborhood Intervention Consistency)

From IJCAI 2021 - uses causal intervention for confidence measurement.

**Approach**:
1. Generate prediction for triple (h, r, t)
2. Actively perturb/intervene on input entity vectors
3. Measure consistency of predictions under perturbation
4. Higher consistency = higher confidence

**Key Insight**: Robust predictions that don't change under small perturbations are more trustworthy.

**Source**: [IJCAI 2021 Proceedings](https://www.ijcai.org/proceedings/2021/288)

---

### 7. BEUrRE (Box Embeddings for Uncertain Relations)

**Core Idea**: Model entities as geometric boxes (axis-aligned hyperrectangles).

**Approach**:
- Entities = boxes in embedding space
- Relations = affine transformations on boxes
- Confidence = intersection volume between boxes

**Advantages**:
- Calibrated probabilistic semantics
- Efficient volume/intersection calculations
- Natural uncertainty representation

**Source**: [arXiv Paper](https://ar5iv.labs.arxiv.org/html/2104.04597)

---

### 8. ConfE (Confidence-aware Embedding)

**Architecture**:
- Entities and entity types in separate spaces
- Asymmetric matrix specifies interaction

**Energy Function**:
```
G(e, τ) = e^T × M × τ

where:
  e = entity embedding
  τ = entity type embedding
  M = asymmetric interaction matrix
```

**Confidence Components**:
- Local tuple confidence (structural information)
- Global triple confidence (graph-wide patterns)

---

### 9. FocusE

Combines traditional scoring with numerical edge values.

**Approach**:
- Nonlinear softplus transformation
- Weights margins between true and corrupted triples
- Incorporates edge metadata in scoring

---

## Production System: IBM Patent (US20180060733A1)

A practical, deployed approach for confidence scoring.

**Core Formula**:
```
Confidence Score = Feature Vector × Learned Weights
```

**Feature Categories**:

| Category | Features |
|----------|----------|
| Static | Extraction method, source type, relationship type |
| Dynamic | Author reputation, usage patterns, click-log data |

**Joint Promotion Score**:
```
s_joint = s_conf × s_robust
```

Triples qualify for automatic promotion when:
- `s_joint >= t_joint`
- `s_conf >= t_conf`
- `s_robust >= t_robust`

**Error Rate Testing**:
```
E = Σ(Xi - Si)² × n/N

where:
  Xi = calculated scores
  Si = target scores
```

**Source**: [US Patent US20180060733A1](https://patents.google.com/patent/US20180060733A1/en)

---

## Google Cloud Enterprise Knowledge Graph

**Reconciliation Confidence Score**:
- Measures confidence that an entity belongs to its assigned cluster
- Value range: [0, 1]
- Factors in cluster density and distance from other clusters

**Key Principle**: If other clusters are close to an entity, its confidence is diminished according to distance.

**Source**: [Google Cloud Docs](https://cloud.google.com/enterprise-knowledge-graph/docs/confidence-score)

---

## Recommended Hybrid Approach for ThoughtLab

Given the requirement to handle arbitrary relationship types, a composite approach is recommended:

### Architecture

| Component | Algorithm | Purpose |
|-----------|-----------|---------|
| Semantic similarity | Embedding distance | Base plausibility between entities |
| Relation-type weighting | Relation-specific projections (TransR-style) | Type-aware scoring |
| Structural confidence | Neighborhood encoding (UKGSE-style) | Graph context |
| Calibration | Platt scaling / isotonic regression | Probability normalization |

### Composite Formula

```python
def confidence(h, r, t, graph):
    """
    Calculate relationship confidence score.

    Args:
        h: head entity
        r: relationship type
        t: tail entity
        graph: knowledge graph for structural context

    Returns:
        float: confidence score in [0, 1]
    """
    # Component scores
    semantic = semantic_similarity(h, t)
    relation = relation_type_score(h, r, t)
    structural = structural_score(
        neighbors(h, graph),
        neighbors(t, graph)
    )

    # Weighted combination
    raw_score = (
        α * semantic +
        β * relation +
        γ * structural
    )

    # Normalize to [0, 1] with sigmoid
    return sigmoid(raw_score)
```

### Weight Configuration

Weights (α, β, γ) can be:
1. **Fixed**: Based on domain knowledge
2. **Learned**: From labeled confidence data
3. **Adaptive**: Per relationship type

### Relation-Type Projection (TransR-style)

```python
def relation_type_score(h, r, t):
    """
    Project entities into relation-specific space.
    """
    # Get or create projection matrix for this relation type
    M_r = get_projection_matrix(r)

    # Project entities
    h_r = h @ M_r
    t_r = t @ M_r

    # Get relation vector
    r_vec = get_relation_vector(r)

    # Translation distance (lower = better)
    distance = norm(h_r + r_vec - t_r)

    # Convert to score (higher = better)
    return 1.0 / (1.0 + distance)
```

---

## Key Academic Sources

### Survey Papers
1. [Uncertainty Management in the Construction of Knowledge Graphs: A Survey](https://arxiv.org/html/2405.16929v2) - Comprehensive 2024 overview

### Link Prediction & Confidence
2. [Evaluating the Calibration of KGE for Trustworthy Link Prediction](https://arxiv.org/abs/2004.01168) - EMNLP 2020
3. [Knowledge Graph Embedding for Link Prediction: Comparative Analysis](https://arxiv.org/abs/2002.00819) - ACM TKDD 2021
4. [Neighborhood Intervention Consistency (NIC)](https://www.ijcai.org/proceedings/2021/288) - IJCAI 2021

### Uncertain Knowledge Graphs
5. [Embedding Uncertain Knowledge Graphs](https://web.cs.ucla.edu/~yzsun/papers/2019_AAAI_UKG.pdf) - AAAI 2019
6. [Probabilistic Box Embeddings for Uncertain KG Reasoning](https://ar5iv.labs.arxiv.org/html/2104.04597)

### Relation Extraction
7. [A Comprehensive Survey on Relation Extraction](https://arxiv.org/abs/2306.02051) - ACM Computing Surveys

### Implementation References
8. [DGL-KE Documentation](https://aws-dglke.readthedocs.io/en/latest/kg.html) - Practical KG embedding
9. [KB2E GitHub (TransE/TransR implementations)](https://github.com/thunlp/KB2E)

---

## Open Questions for ThoughtLab

1. **Training Data**: Do we have labeled confidence scores to train from, or do we need unsupervised approaches?

2. **Relationship Type Handling**:
   - Are relationship types predefined or user-created?
   - Should each type have its own learned projection matrix?

3. **Real-time vs Batch**:
   - Do we need real-time confidence scoring?
   - Can we precompute embeddings?

4. **Calibration**:
   - How important is calibrated probability output vs relative ranking?

5. **Cold Start**:
   - How to handle new relationship types with no training examples?

---

## Next Steps

1. Review ThoughtLab's current data model for relationships
2. Determine available training signals (user feedback, validation data)
3. Prototype simple baseline (e.g., TransE + semantic similarity)
4. Iterate with more sophisticated components as needed
