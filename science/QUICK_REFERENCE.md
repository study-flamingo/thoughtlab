# ThoughtLab Scientific Foundations - Quick Reference

**One-page summary of key algorithms and implementation priorities**

---

## 🎯 Top 5 Immediate Actions (Week 1-2)

### 1. Deploy HNSW Vector Indexing (Semantic Similarity)
**File**: `semantic_similarity_algorithms.md`
**Why**: 10-100× faster similarity search
**How**:
```cypher
-- Update Neo4j to 5.13+ and create HNSW index
CREATE VECTOR INDEX node_embedding_hnsw
FOR (n:Node) ON (n.embedding)
OPTIONS {indexConfig: {
  `vector.dimensions`: 1536,
  `vector.similarity_function`: 'cosine',
  `vector.hnsw`: {`m`: 16, `ef_construction`: 200, `ef_search`: 100}
}}
```
**Expected Impact**: Query time from 100ms → 5-10ms
**Priority**: P0 (Do this first!)

---

### 2. Implement Beta Distribution Tracking (Uncertainty)
**File**: `uncertainty_quantification.md`
**Why**: Calibrated confidence scores, better trust
**How**:
```python
class BetaUncertainty:
    def __init__(self, successes=1, failures=1):
        self.alpha = successes  # Correct predictions
        self.beta = failures    # Incorrect predictions

    def update(self, correct):
        if correct: self.alpha += 1
        else: self.beta += 1

    def confidence(self):
        return self.alpha / (self.alpha + self.beta)

    def uncertainty(self):
        return (self.alpha * self.beta) / \
               ((self.alpha + self.beta) ** 2 * (self.alpha + self.beta + 1))
```
**Track per relationship type** (SUPPORTS, CONTRADICTS, etc.)
**Priority**: P0 (Foundation for all ML)

---

### 3. Enable Neo4j Graph Data Science (Graph Traversal)
**File**: `graph_traversal_algorithms.md`
**Why**: Advanced analytics, path finding, community detection
**How**:
```bash
# Update docker-compose.yml
neo4j:
  image: neo4j:5.13.0-enterprise
  environment:
    - NEO4J_PLUGINS=["graph-data-science"]
```
**Key Algorithms**:
- `gds.shortestPath.stream` - Find connection paths
- `gds.pageRank.stream` - Find influential nodes
- `gds.louvain.stream` - Discover research communities

**Priority**: P0 (Enables user value)

---

### 4. Add Confidence Calibration (Uncertainty)
**File**: `uncertainty_quantification.md`
**Why**: Fix "80% confidence but 60% accuracy" problem
**How** (Simplest - Temperature Scaling):
```python
class TemperatureScaling(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def calibrate(self, logits):
        return logits / self.temperature

# Optimize temperature on validation set
# Goal: ECE (Expected Calibration Error) < 0.05
```
**Priority**: P0 (Improves trust immediately)

---

### 5. Start Tracking Relationship Uncertainty (All Docs)
**File**: All documents
**Why**: Foundation for automated suggestions
**How**:
```python
# For each relationship type
uncertainty_models = {
    'SUPPORTS': BetaUncertainty(successes=1, failures=1),
    'CONTRADICTS': BetaUncertainty(successes=1, failures=1),
    'RELATES_TO': BetaUncertainty(successes=1, failures=1)
}

# Only suggest when:
confidence > 0.8 AND uncertainty < 0.1 AND prob_gt_threshold > 0.8
```
**Priority**: P0 (Immediate quality improvement)

---

## 📊 Algorithms by Use Case

### When you need to find related nodes
**Best approach**:
1. **HNSW + Cosine Similarity** (Fast baseline)
2. **Add Structural Similarity** (Jaccard/Adamic-Adar)
3. **Relation-specific projections** (TransR-style)

**Performance**:
- Linear search: 100-500ms (current)
- HNSW: 5-50ms (10-100× faster)
- With structural: +10% accuracy

**Code location**: `semantic_similarity_algorithms.md:Section 5`

---

### When you need relationship confidence
**Best approach**:
1. **Beta distributions** (Simple, effective)
2. **Multi-source combination** (Text + Graph + Model)
3. **Bayesian methods** (Advanced, better uncertainty)

**Formula**:
```
final_confidence = 0.6 × text_sim + 0.3 × graph_sim + 0.1 × model_conf
uncertainty = combined_variance
```

**Code location**: `uncertainty_quantification.md:Section 7`

---

### When you need to navigate the graph
**Best approach**:
1. **Shortest path** (Dijkstra for weighted, BFS for unweighted)
2. **Centrality analysis** (PageRank for influence, Betweenness for bridges)
3. **Community detection** (Louvain for research themes)

**Neo4j Commands**:
```cypher
-- Shortest path with weights
CALL gds.shortestPath.stream({...})

-- PageRank for influence
CALL gds.pageRank.stream({...})

-- Communities
CALL gds.louvain.stream({...})
```

**Code location**: `graph_traversal_algorithms.md:Sections 1-3`

---

### When you need to predict missing relationships
**Best approach**:
1. **ComplEx** (Fast, handles asymmetric relations)
2. **RGCN** (Uses graph structure, best accuracy)
3. **Hybrid** (ComplEx for speed, RGCN for critical predictions)

**Performance**:
- TransE: 75-85% Hits@10
- ComplEx: 85-90% Hits@10
- RGCN: 90-95% Hits@10

**Code location**: `graph_embedding_techniques.md:Sections 3-4`

---

### When you need to extract entities from text
**Best approach**:
1. **Rule-based** (Week 1, 70% F1)
2. **BERT fine-tuned** (Month 1, 90% F1)
3. **Active learning** (Month 3, 95% F1)

**Training data needed**:
- Rule-based: 0 examples
- BERT: 500-1000 labeled sentences
- Active learning: Start with 500, improve with feedback

**Code location**: `graph_construction_entity_linking.md:Section 1`

---

### When you need to extract relationships from text
**Best approach**:
1. **BERT relation classifier** (85-92% F1)
2. **REBEL** (End-to-end, 85-90% F1)
3. **Pattern-based** (Fast, 60-80% F1, good for structured text)

**Training data needed**:
- BERT: 1000-2000 labeled pairs
- REBEL: 2000-5000 labeled triples
- Patterns: 0 (but manual pattern creation)

**Code location**: `graph_construction_entity_linking.md:Section 1.2`

---

## 🎯 Performance Targets

| Operation | Current | Target | Method |
|-----------|---------|--------|--------|
| Similarity Search | 100-500ms | 5-50ms | HNSW indexing |
| Shortest Path | ~100ms | 1-10ms | Neo4j GDS |
| Relationship Prediction | Manual | 85-92% F1 | BERT fine-tune |
| Link Prediction | N/A | 90% Hits@10 | RGCN |
| Entity Extraction | Manual | 90-95% F1 | BERT NER |
| Confidence Calibration | N/A | ECE < 0.05 | Temperature scaling |

---

## 📈 Implementation Progression

### Stage 1: Foundation (Week 1-2)
```
Goal: 70% of benefit, 20% of effort
Tasks:
✅ HNSW indexing (1 day)
✅ Beta tracking (1 day)
✅ Neo4j GDS setup (1 day)
✅ Basic calibration (2 days)
✅ Uncertainty thresholds (1 day)

Total: ~6 days for P0 improvements
```

### Stage 2: Intelligence (Month 1)
```
Goal: 90% of benefit, 50% of effort
Tasks:
• BERT fine-tuning (1 week)
• ComplEx embeddings (1 week)
• Calibration system (3 days)
• User feedback loop (1 week)
• A/B testing framework (1 week)

Total: ~4 weeks for P1 improvements
```

### Stage 3: Advanced (Month 2-3)
```
Goal: 95% of benefit, 80% of effort
Tasks:
• RGCN implementation (2 weeks)
• Community detection (1 week)
• Active learning (2 weeks)
• Risk-adjusted decisions (1 week)
• Monitoring dashboard (1 week)

Total: ~7 weeks for P2 improvements
```

### Stage 4: Production (Month 4-6)
```
Goal: 99% of benefit, 100% of effort
Tasks:
• Multi-task learning (3 weeks)
• PLM integration (2 weeks)
• Distributed processing (3 weeks)
• Advanced monitoring (2 weeks)
• Research publications (ongoing)

Total: ~10 weeks for P3 improvements
```

---

## 🎓 Learning Path (2 Weeks)

### Week 1: Foundations
**Day 1-2**: Probability & Statistics
- Read: Murphy (Ch. 1-3)
- Practice: Mean, variance, confidence intervals
- Apply: Beta distribution tracking

**Day 3-4**: Deep Learning Basics
- Read: Goodfellow (Ch. 1-6)
- Practice: PyTorch tensors, autograd
- Apply: Simple neural network

**Day 5-7**: Graph Basics
- Read: Neo4j GDS documentation
- Practice: Cypher queries, basic algorithms
- Apply: Run PageRank on your data

### Week 2: Applied ML
**Day 8-9**: Transformers & BERT
- Read: BERT paper (2018)
- Practice: HuggingFace tutorials
- Apply: Fine-tune on toy dataset

**Day 10-11**: Graph Neural Networks
- Read: RGCN paper (2017)
- Practice: PyTorch Geometric tutorials
- Apply: Simple link prediction

**Day 12-14**: Uncertainty & Bayesian Methods
- Read: Dropout as Bayesian Approximation (2016)
- Practice: Implement MC Dropout
- Apply: Calibrate a model

---

## 🔧 Quick Commands

### Install Dependencies
```bash
# Python (backend)
cd backend
pip install torch torch-geometric transformers scikit-learn scipy
pip install pyro-ppl  # Bayesian inference
pip install wandb  # Experiment tracking

# Node.js (frontend - optional for visualization)
cd frontend
npm install @antv/g6  # Graph visualization
npm install @tensorflow/tfjs  # Optional: browser ML
```

### Initialize Neo4j GDS
```bash
# Check version
cypher-shell "RETURN gds.version()"

# Run basic algorithm
cypher-shell "
CALL gds.pageRank.stream({
  nodeQuery: 'MATCH (n) RETURN id(n) as id',
  relationshipQuery: 'MATCH (n)-[r]->(m) RETURN id(n) as source, id(m) as target'
})
YIELD nodeId, score
RETURN gds.util.asNode(nodeId).id as node_id, score
ORDER BY score DESC
LIMIT 10
"
```

### Test Your Setup
```python
# test_uncertainty.py
from science.uncertainty_quantification import BetaUncertainty

# Test Beta uncertainty
unc = BetaUncertainty(successes=5, failures=3)
print(f"Confidence: {unc.mean():.2f}")
print(f"Uncertainty: {unc.variance():.3f}")
print(f"95% CI: {unc.credible_interval()}")

# Test update
unc.update(correct=True)
print(f"After update: {unc.mean():.2f}")
```

---

## 📊 Metrics Dashboard (What to Track)

### Daily Metrics
```python
metrics = {
    # Performance
    'query_latency_p95': measure_latency(),
    'memory_usage_gb': measure_memory(),

    # Quality
    'relationship_suggestions': count_suggestions(),
    'acceptance_rate': accepted / total,
    'avg_confidence': avg_confidence(),
    'avg_uncertainty': avg_uncertainty(),

    # Learning
    'feedback_collected': count_feedback(),
    'model_updates': count_updates(),
}
```

### Weekly Review
- **Calibration**: Is ECE < 0.05?
- **Acceptance Rate**: Is it > 70%?
- **User Trust**: Survey score > 4.0/5?
- **Graph Growth**: New nodes/relationships added?

### Monthly Review
- **A/B Test Results**: Did improvements win?
- **Model Performance**: F1 scores, Hits@10 trending up?
- **System Scalability**: Handling 2× more data?
- **User Satisfaction**: Feedback trends?

---

## 🚨 Common Pitfalls & Solutions

### Pitfall 1: Overconfident Predictions
**Symptom**: Model says 95% confidence but only 70% accuracy
**Solution**:
- Implement calibration (temperature scaling)
- Track Beta distributions per type
- Use uncertainty thresholds for review

**Code**: `uncertainty_quantification.md:Section 6`

---

### Pitfall 2: Slow Similarity Search
**Symptom**: Queries take 500ms+ as graph grows
**Solution**:
- Switch from linear to HNSW search
- Use Neo4j 5.13+ vector indexes
- Cache popular queries

**Code**: `semantic_similarity_algorithms.md:Section 2`

---

### Pitfall 3: Poor Relationship Quality
**Symptom**: Users reject most automated suggestions
**Solution**:
- Increase confidence threshold (0.8+)
- Add uncertainty threshold (0.2-)
- Use multi-source evidence
- Implement human review for medium confidence

**Code**: `uncertainty_quantification.md:Section 9`

---

### Pitfall 4: Cold Start Problem
**Symptom**: New entities have no relationships
**Solution**:
- Use semantic similarity from day 1
- Pre-train on external KGs
- Active learning: prioritize new entity feedback
- Use transfer learning from related domains

**Code**: `graph_embedding_techniques.md:Section 5`

---

### Pitfall 5: Model Drift
**Symptom**: Performance degrades over time
**Solution**:
- Monitor calibration weekly
- Retrain on user feedback monthly
- A/B test all changes
- Track concept drift

**Code**: `uncertainty_quantification.md:Section 10`

---

## 📞 Getting Help

### Questions about specific algorithms?
- **Similarity**: `semantic_similarity_algorithms.md`
- **Traversal**: `graph_traversal_algorithms.md`
- **Embeddings**: `graph_embedding_techniques.md`
- **Construction**: `graph_construction_entity_linking.md`
- **Uncertainty**: `uncertainty_quantification.md`

### Questions about implementation?
- **Phase 1**: See "Top 5 Immediate Actions" above
- **Priorities**: Check priority matrices in each doc
- **Timeline**: See "Implementation Progression" above

### Questions about research direction?
- **Agenda**: `RESEARCH_AGENDA.md`
- **Open Questions**: Section 5 in RESEARCH_AGENDA
- **Learning Path**: Section 8 in RESEARCH_AGENDA

---

## ✅ Checklist: Week 1

- [ ] Upgrade Neo4j to 5.13+ Enterprise
- [ ] Install HNSW vector index
- [ ] Implement Beta uncertainty tracking
- [ ] Enable Neo4j GDS plugin
- [ ] Set up confidence calibration
- [ ] Create uncertainty thresholds (0.8 conf, 0.2 unc)
- [ ] Collect baseline metrics
- [ ] Schedule team learning sessions

**Estimated time**: 6 days
**Expected impact**: 50% improvement in speed, 2× improvement in trust

---

## 🎯 Success Criteria

### Week 1 Success
- ✅ Query latency < 50ms (was 200ms+)
- ✅ Confidence scores calibrated (ECE < 0.1)
- ✅ Uncertainty tracked per relationship type
- ✅ Graph algorithms working (PageRank, communities)

### Month 1 Success
- ✅ BERT models fine-tuned
- ✅ Automated suggestions with >70% acceptance
- ✅ A/B test showing improvement
- ✅ User feedback loop active

### Month 3 Success
- ✅ RGCN link prediction at 90%+ accuracy
- ✅ Community detection revealing insights
- ✅ Active learning reducing manual work by 50%
- ✅ User trust score > 4.0/5

### Month 6 Success
- ✅ State-of-the-art performance
- ✅ Handle 100K+ entities
- ✅ Continuous self-improvement
- ✅ Production-ready system

---

**Quick Reference Version**: 1.0
**Last Updated**: 2026-01-25
**Next Update**: After Phase 1 completion (Week 2)