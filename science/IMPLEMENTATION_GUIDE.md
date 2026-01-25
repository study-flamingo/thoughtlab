# ThoughtLab Scientific Foundations - Implementation Guide

**Date**: 2026-01-25
**Purpose**: Priority roadmap for implementing scientific algorithms in ThoughtLab

---

## Overview

This guide synthesizes research from 6 key areas into actionable implementation priorities:

1. **Relationship Confidence Scoring** ✅ Complete
2. **Semantic Similarity Algorithms** ✅ Complete
3. **Graph Traversal Algorithms** ✅ Complete
4. **Knowledge Graph Embedding Techniques** ✅ Complete
5. **Graph Construction & Entity Linking** ✅ Complete
6. **Uncertainty Quantification** ✅ Complete

---

## Quick Reference: Priority Matrix

| Impact | Low Effort | Medium Effort | High Effort |
|--------|------------|---------------|-------------|
| **High** | P0: Week 1-2 | P1: Month 1 | P2: Month 2-3 |
| **Medium** | P1: Month 1 | P2: Month 2-3 | P3: Month 4-6 |
| **Low** | P2: Month 2-3 | P3: Month 4-6 | P4: Future |

---

## Phase 0: Foundation (Week 1)

### 1. Set Up Scientific Infrastructure

**Dependencies**:
```bash
# Core ML libraries
pip install torch==2.0.1 torch-geometric==2.3.1
pip install transformers==4.30.0 scikit-learn==1.3.0
pip install scipy==1.11.0 numpy==1.24.0

# NLP & KG processing
pip install spacy==3.6.0 scispacy==0.5.1
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_core_sci_sm-0.5.1.tar.gz

# Evaluation & monitoring
pip install wandb==0.15.0 optuna==3.0.0
```

**Directory Structure**:
```
backend/app/science/
├── uncertainty/          # Beta distributions, calibration
├── embeddings/           # TransE, ComplEx, RGCN
├── similarity/           # HNSW, multi-metric
├── extraction/           # NER, relation extraction
├── evaluation/           # Metrics & benchmarks
└── monitoring/           # Dashboard & drift detection
```

### 2. Initialize Scientific Tracking

```python
# backend/app/science/__init__.py

class ScientificFoundation:
    """Main coordinator for scientific algorithms."""

    def __init__(self, config):
        from .uncertainty import UncertaintyTracker
        from .embeddings import GraphEmbeddings
        from .similarity import SimilarityEngine
        from .extraction import KGConstructor

        self.uncertainty = UncertaintyTracker()
        self.embeddings = GraphEmbeddings()
        self.similarity = SimilarityEngine()
        self.constructor = KGConstructor()

        self.config = config

    async def initialize(self):
        """Load models and prepare for inference."""
        await self.embeddings.load_pretrained()
        await self.similarity.initialize_index()
        await self.uncertainty.initialize_priors()
```

---

## Phase 1: Quick Wins (Week 1-2)

### 1. Beta Distribution Uncertainty (P0)

**Why**: Immediate improvement in confidence reliability

**Implementation**:
```python
# backend/app/science/uncertainty/beta_tracker.py

from scipy.stats import beta
import numpy as np

class BetaUncertaintyTracker:
    """Track uncertainty using Beta distributions."""

    def __init__(self):
        # Prior: Laplace smoothing (1 success, 1 failure)
        self.distributions = {
            'SUPPORTS': {'successes': 1, 'failures': 1},
            'CONTRADICTS': {'successes': 1, 'failures': 1},
            'RELATES_TO': {'successes': 1, 'failures': 1},
            'default': {'successes': 1, 'failures': 1}
        }

    def get_confidence(self, relation_type):
        """Get calibrated confidence with uncertainty bounds."""
        dist = self.distributions.get(relation_type, self.distributions['default'])
        successes, failures = dist['successes'], dist['failures']

        mean = successes / (successes + failures)
        variance = (successes * failures) / \
                   ((successes + failures) ** 2 * (successes + failures + 1))

        # 95% credible interval
        lower = beta.ppf(0.025, successes, failures)
        upper = beta.ppf(0.975, successes, failures)

        # Probability that confidence > threshold
        prob_gt_70 = 1 - beta.cdf(0.7, successes, failures)

        return {
            'mean': float(mean),
            'variance': float(variance),
            'credible_interval': [float(lower), float(upper)],
            'prob_gt_threshold': float(prob_gt_70),
            'n_observations': successes + failures - 2  # Subtract prior
        }

    def update(self, relation_type, correct):
        """Update distribution with new observation."""
        if relation_type not in self.distributions:
            self.distributions[relation_type] = {'successes': 1, 'failures': 1}

        if correct:
            self.distributions[relation_type]['successes'] += 1
        else:
            self.distributions[relation_type]['failures'] += 1

    def get_recommendation_strength(self, relation_type):
        """Calculate recommendation score for UI."""
        conf = self.get_confidence(relation_type)

        # Weighted combination
        score = (
            0.6 * conf['mean'] +
            0.3 * (1 - conf['variance']) +
            0.1 * conf['prob_gt_threshold']
        )

        return score
```

**Integration Points**:
1. Update in `tool_service.py` when relationship suggestions are accepted/rejected
2. Return uncertainty in API responses
3. Use for recommendation filtering in frontend

**Expected Impact**:
- 20-30% improvement in relationship recommendation quality
- Better user trust (clear uncertainty bounds)
- Foundation for automated decision making

---

### 2. Confidence Calibration (P0)

**Why**: Fix miscalibrated confidence scores

**Implementation**:
```python
# backend/app/science/uncertainty/calibration.py

import torch
import torch.nn as nn

class TemperatureScaling:
    """Post-hoc calibration using temperature parameter."""

    def __init__(self, model):
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1))

    def calibrate(self, val_loader, epochs=50):
        """Optimize temperature on validation set."""
        # Freeze model parameters
        for param in self.model.parameters():
            param.requires_grad = False

        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=epochs)
        criterion = nn.CrossEntropyLoss()

        def eval_nll():
            loss = 0
            n = 0
            for inputs, labels in val_loader:
                with torch.no_grad():
                    logits = self.model(inputs)
                loss += criterion(logits / self.temperature, labels).item()
                n += len(labels)
            return loss / n

        optimizer.step(eval_nll)
        return self.temperature.item()

    def predict(self, logits):
        """Apply temperature scaling."""
        calibrated = logits / self.temperature
        probs = torch.softmax(calibrated, dim=-1)
        return probs

def evaluate_calibration(probs, labels, n_bins=10):
    """Calculate Expected Calibration Error."""
    confidences, preds = probs.max(dim=1)
    accuracies = (preds == labels).float()

    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    ece = 0

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.float().mean()

        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_conf_in_bin = confidences[in_bin].mean()
            ece += (avg_conf_in_bin - accuracy_in_bin).abs() * prop_in_bin

    return ece.item()
```

**Integration Points**:
1. Calibrate relationship classifier after training
2. Monitor ECE weekly to detect drift
3. Re-calibrate when ECE > 0.05

**Expected Impact**:
- Confidence scores become reliable (80% confidence = 80% accuracy)
- Better decision thresholds
- Reduced user confusion

---

### 3. Multi-Source Similarity (P1)

**Why**: Improve relationship discovery beyond simple cosine

**Implementation**:
```python
# backend/app/science/similarity/multi_metric.py

class MultiMetricSimilarity:
    """Combine embedding similarity with structural features."""

    def __init__(self, embedding_model, kg):
        self.embedding_model = embedding_model
        self.kg = kg
        self.weights = {
            'embedding': 0.6,
            'structural': 0.3,
            'type_match': 0.1
        }

    def compute_similarity(self, node_a, node_b, relation_type=None):
        """Compute similarity using multiple metrics."""

        # 1. Embedding similarity (semantic)
        emb_sim = self.embedding_similarity(node_a, node_b)

        # 2. Structural similarity (graph topology)
        struct_sim = self.structural_similarity(node_a, node_b)

        # 3. Type compatibility
        type_sim = self.type_compatibility(node_a, node_b, relation_type)

        # Weighted combination
        combined = (
            self.weights['embedding'] * emb_sim +
            self.weights['structural'] * struct_sim +
            self.weights['type_match'] * type_sim
        )

        return {
            'combined': combined,
            'breakdown': {
                'embedding': emb_sim,
                'structural': struct_sim,
                'type_match': type_sim
            }
        }

    def embedding_similarity(self, node_a, node_b):
        """Cosine similarity of embeddings."""
        emb_a = self.embedding_model.get_embedding(node_a['id'])
        emb_b = self.embedding_model.get_embedding(node_b['id'])
        return cosine_similarity(emb_a, emb_b)

    def structural_similarity(self, node_a, node_b):
        """Jaccard similarity of neighborhoods."""
        neighbors_a = set(self.kg.get_neighbors(node_a['id']))
        neighbors_b = set(self.kg.get_neighbors(node_b['id']))

        intersection = len(neighbors_a & neighbors_b)
        union = len(neighbors_a | neighbors_b)

        return intersection / union if union > 0 else 0

    def type_compatibility(self, node_a, node_b, relation_type):
        """Check if relationship type makes sense."""
        type_a = node_a['type']
        type_b = node_b['type']

        # Pre-defined compatibility matrix
        compat_matrix = {
            'OBS': {'HYP': 0.9, 'SRC': 0.7, 'CON': 0.6},
            'HYP': {'OBS': 0.9, 'SRC': 0.8, 'CON': 0.7},
            'SRC': {'OBS': 0.7, 'HYP': 0.8, 'CON': 0.5},
            'CON': {'OBS': 0.6, 'HYP': 0.7, 'SRC': 0.5}
        }

        return compat_matrix.get(type_a, {}).get(type_b, 0.5)
```

**Integration Points**:
1. Replace `find_similar()` in `similarity.py`
2. Add to relationship recommendation pipeline
3. Use for graph exploration features

**Expected Impact**:
- 25-35% improvement in relevant relationship discovery
- Better handling of diverse relationship types
- More robust similarity calculations

---

## Phase 2: ML Enhancement (Month 1)

### 1. Fine-tune BERT for NER & Relation Extraction (P1)

**Why**: State-of-the-art accuracy with domain adaptation

**Implementation**:
```python
# backend/app/science/extraction/bert_models.py

from transformers import BertForTokenClassification, BertTokenizer
from transformers import BertForSequenceClassification

class BERTEntityExtractor:
    """Fine-tuned BERT for entity extraction."""

    def __init__(self, model_path=None):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        if model_path:
            self.model = BertForTokenClassification.from_pretrained(model_path)
        else:
            # Initialize for 5 entity types
            self.model = BertForTokenClassification.from_pretrained(
                'bert-base-uncased',
                num_labels=5  # OBS, HYP, SRC, CON, ENT
            )

    def train(self, train_data, epochs=3, learning_rate=2e-5):
        """Fine-tune on ThoughtLab domain."""
        # Prepare dataset
        dataset = self.prepare_dataset(train_data)

        # Training arguments
        training_args = {
            'learning_rate': learning_rate,
            'per_device_train_batch_size': 16,
            'num_train_epochs': epochs,
            'weight_decay': 0.01,
            'evaluation_strategy': 'epoch',
            'save_strategy': 'epoch',
            'load_best_model_at_end': True
        }

        # Train
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset['train'],
            eval_dataset=dataset['val']
        )

        trainer.train()
        return trainer

    def predict(self, text):
        """Extract entities from text."""
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True)

        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)

        # Convert to entity spans
        entities = self.decode_predictions(predictions, inputs['attention_mask'])
        return entities

class BERTRelationExtractor:
    """Fine-tuned BERT for relation classification."""

    def __init__(self, model_path=None):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        if model_path:
            self.model = BertForSequenceClassification.from_pretrained(model_path)
        else:
            self.model = BertForSequenceClassification.from_pretrained(
                'bert-base-uncased',
                num_labels=8  # Number of relation types
            )

    def prepare_input(self, text, head_text, tail_text):
        """Format with entity markers."""
        marked = text.replace(
            head_text, f"[E1]{head_text}[/E1]"
        ).replace(
            tail_text, f"[E2]{tail_text}[/E2]"
        )
        return f"[CLS] {marked} [SEP]"

    def predict(self, text, head, tail):
        """Predict relation between head and tail."""
        input_text = self.prepare_input(text, head, tail)
        inputs = self.tokenizer(input_text, return_tensors='pt')

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)

        return probs
```

**Training Data Requirements**:
- Minimum: 500 labeled sentences
- Recommended: 2000+ labeled sentences
- Format: IOB for entities, (head, relation, tail) for relations

**Integration Points**:
1. Create training data from user corrections
2. Fine-tune weekly with new data
3. Deploy as prediction service
4. A/B test against rule-based baseline

**Expected Impact**:
- 85-92% F1-score for entity extraction (vs 70-80% rule-based)
- 85-92% F1-score for relation extraction (vs 60-70% pattern-based)
- 3-5× more entities/relationships extracted

---

### 2. Monte Carlo Dropout for Uncertainty (P1)

**Why**: Bayesian uncertainty without full Bayesian NN

**Implementation**:
```python
# backend/app/science/uncertainty/mc_dropout.py

class MCDropoutModel(nn.Module):
    """Neural network with MC dropout for uncertainty."""

    def __init__(self, input_dim, hidden_dims, output_dim, dropout_rate=0.2):
        super().__init__()
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x, train=False):
        """Forward pass (dropout active at train time AND inference)."""
        if train:
            self.train()
        else:
            self.eval()

        return self.network(x)

    def predict_with_uncertainty(self, x, n_samples=50):
        """Monte Carlo sampling for uncertainty."""
        self.train()  # Keep dropout active
        predictions = []

        with torch.no_grad():
            for _ in range(n_samples):
                pred = self.forward(x, train=False)
                predictions.append(pred)

        predictions = torch.stack(predictions)  # [n_samples, batch, output]
        mean = predictions.mean(dim=0)
        variance = predictions.var(dim=0)

        return {
            'mean': mean,
            'variance': variance,
            'std': predictions.std(dim=0),
            'aleatoric': self.estimate_aleatoric(predictions),
            'epistemic': variance
        }

    def estimate_aleatoric(self, predictions):
        """Estimate aleatoric (data) uncertainty."""
        # For classification: entropy of mean prediction
        mean_pred = predictions.mean(dim=0)
        probs = torch.softmax(mean_pred, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
        return entropy

def train_with_uncertainty(model, train_loader, val_loader, epochs=10):
    """Train MC dropout model."""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()

            # Multiple forward passes for uncertainty
            losses = []
            for _ in range(5):  # 5 MC samples
                pred = model(batch_x, train=True)
                loss = criterion(pred, batch_y)
                losses.append(loss)

            # Average loss
            avg_loss = torch.stack(losses).mean()
            avg_loss.backward()
            optimizer.step()

        # Validate
        model.eval()
        with torch.no_grad():
            val_loss = 0
            for batch_x, batch_y in val_loader:
                pred = model(batch_x, train=False)
                val_loss += criterion(pred, batch_y).item()

        print(f"Epoch {epoch}: Val Loss = {val_loss/len(val_loader):.4f}")
```

**Integration Points**:
1. Replace confidence predictions in relationship classifier
2. Use for uncertainty-aware recommendations
3. Monitor epistemic vs aleatoric uncertainty

**Expected Impact**:
- Well-calibrated uncertainty estimates
- 20% reduction in false positive recommendations
- Better understanding of model ignorance

---

### 3. HNSW Vector Index (P0)

**Why**: 10-100× faster similarity search

**Implementation**:
```python
# backend/app/science/similarity/hnsw_index.py

import faiss
import numpy as np

class HNSWVectorIndex:
    """HNSW index for efficient similarity search."""

    def __init__(self, dimension=1536, max_elements=1000000):
        self.dimension = dimension
        self.max_elements = max_elements

        # HNSW parameters
        self.M = 16              # Number of connections per node
        self.ef_construction = 200  # Search breadth during construction
        self.ef_search = 100     # Search breadth during query

        # Initialize index
        self.index = faiss.IndexHNSWFlat(dimension, self.M)
        self.index.hnsw.efConstruction = self.ef_construction
        self.index.hnsw.efSearch = self.ef_search

        self.id_to_node = {}  # Map FAISS IDs to node IDs
        self.next_id = 0

    def add_embedding(self, node_id, embedding):
        """Add embedding to index."""
        if isinstance(embedding, list):
            embedding = np.array(embedding, dtype=np.float32)

        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)

        # Add to index
        self.index.add(embedding)

        # Store mapping
        self.id_to_node[self.next_id] = node_id
        self.next_id += 1

    def search(self, embedding, k=10, min_score=None):
        """Find nearest neighbors."""
        if isinstance(embedding, list):
            embedding = np.array(embedding, dtype=np.float32)

        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)

        # Search
        distances, ids = self.index.search(embedding, k)

        # Convert to results
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], ids[0])):
            if idx < 0:  # Invalid result
                continue

            node_id = self.id_to_node.get(idx)
            if node_id is None:
                continue

            # FAISS uses L2 distance, convert to cosine similarity
            # cos_sim = 1 - (dist / 2)  # Approximation for normalized vectors
            cos_sim = 1 - dist / 2

            if min_score and cos_sim < min_score:
                continue

            results.append({
                'node_id': node_id,
                'score': float(cos_sim),
                'distance': float(dist)
            })

        return results

    def build_from_kg(self, kg, embedding_func):
        """Build index from entire knowledge graph."""
        print("Building HNSW index from knowledge graph...")

        nodes = kg.get_all_nodes()
        embeddings = []
        node_ids = []

        for node in nodes:
            emb = embedding_func(node['id'])
            if emb is not None:
                embeddings.append(emb)
                node_ids.append(node['id'])

        # Add in batches for efficiency
        batch_size = 10000
        for i in range(0, len(embeddings), batch_size):
            batch_emb = embeddings[i:i + batch_size]
            batch_ids = node_ids[i:i + batch_size]

            for node_id, emb in zip(batch_ids, batch_emb):
                self.add_embedding(node_id, emb)

            print(f"Processed {i + len(batch_emb)}/{len(embeddings)} embeddings")

        print(f"HNSW index built: {len(embeddings)} nodes")

def optimize_hnsw_parameters(kg, embedding_func, test_queries):
    """Find optimal HNSW parameters for your data."""
    param_grid = {
        'M': [8, 16, 24, 32],
        'ef_construction': [100, 200, 400],
        'ef_search': [50, 100, 200]
    }

    results = []

    for M in param_grid['M']:
        for ef_con in param_grid['ef_construction']:
            for ef_search in param_grid['ef_search']:
                print(f"Testing M={M}, ef_con={ef_con}, ef_search={ef_search}")

                # Build index
                index = HNSWVectorIndex(dimension=1536, max_elements=len(kg.get_all_nodes()))
                index.M = M
                index.ef_construction = ef_con
                index.ef_search = ef_search

                index.build_from_kg(kg, embedding_func)

                # Test queries
                query_times = []
                recalls = []

                for query, true_neighbors in test_queries:
                    start = time.time()
                    results = index.search(query, k=10)
                    query_times.append(time.time() - start)

                    # Calculate recall
                    found = sum(1 for r in results if r['node_id'] in true_neighbors)
                    recall = found / len(true_neighbors) if true_neighbors else 0
                    recalls.append(recall)

                results.append({
                    'params': {'M': M, 'ef_con': ef_con, 'ef_search': ef_search},
                    'avg_query_time': np.mean(query_times),
                    'avg_recall': np.mean(recalls),
                    'memory_mb': index.index.memory()
                })

    # Find best trade-off
    best = max(results, key=lambda x: x['avg_recall'] / x['avg_query_time'])
    return best
```

**Integration Points**:
1. Replace Neo4j vector index queries
2. Build index nightly or on-demand
3. Use for all similarity search operations

**Expected Impact**:
- 10-100× faster similarity queries
- Handle 10× more entities
- Better recall than linear search

---

## Phase 3: Graph Intelligence (Month 2-3)

### 1. Graph Embeddings (ComplEx) (P2)

**Why**: Relationship-aware embeddings for better predictions

**Implementation**:
```python
# backend/app/science/embeddings/complex.py

class ComplEx(nn.Module):
    """ComplEx: Complex embeddings for KG completion."""

    def __init__(self, num_entities, num_relations, dim=100):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.dim = dim

        # Complex embeddings (real + imaginary parts)
        self.entity_real = nn.Embedding(num_entities, dim)
        self.entity_imag = nn.Embedding(num_entities, dim)
        self.relation_real = nn.Embedding(num_relations, dim)
        self.relation_imag = nn.Embedding(num_relations, dim)

        # Initialize
        self.reset_parameters()

    def reset_parameters(self):
        """Xavier initialization."""
        nn.init.xavier_uniform_(self.entity_real.weight)
        nn.init.xavier_uniform_(self.entity_imag.weight)
        nn.init.xavier_uniform_(self.relation_real.weight)
        nn.init.xavier_uniform_(self.relation_imag.weight)

    def score(self, head, relation, tail):
        """Score triple using complex multiplication."""
        h_r = self.entity_real(head)
        h_i = self.entity_imag(head)
        r_r = self.relation_real(relation)
        r_i = self.relation_imag(relation)
        t_r = self.entity_real(tail)
        t_i = self.entity_imag(tail)

        # Complex multiplication: (a + bi)(c + di) = (ac - bd) + (ad + bc)i
        real_part = h_r * r_r * t_r + h_i * r_i * t_i
        imag_part = h_r * r_i * t_i + h_i * r_r * t_r

        # Take real part of final complex number
        score = (real_part * t_r + imag_part * t_i).sum(dim=1)
        return score

    def predict_tails(self, head, relation, top_k=10):
        """Predict possible tails for (head, relation, ?)."""
        scores = []
        for tail_id in range(self.num_entities):
            score = self.score(
                torch.tensor([head]),
                torch.tensor([relation]),
                torch.tensor([tail_id])
            )
            scores.append((tail_id, score.item()))

        return sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]

    def predict_relations(self, head, tail, top_k=10):
        """Predict possible relations for (head, ?, tail)."""
        scores = []
        for rel_id in range(self.num_relations):
            score = self.score(
                torch.tensor([head]),
                torch.tensor([rel_id]),
                torch.tensor([tail])
            )
            scores.append((rel_id, score.item()))

        return sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]

class ComplExTrainer:
    """Trainer for ComplEx model."""

    def __init__(self, model, kg):
        self.model = model
        self.kg = kg
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    def prepare_training_data(self):
        """Extract triples from knowledge graph."""
        triples = []
        for rel in self.kg.get_all_relationships():
            head_id = rel['from_id']
            tail_id = rel['to_id']
            rel_type = rel['type']

            # Map to indices
            head_idx = self.kg.get_entity_index(head_id)
            tail_idx = self.kg.get_entity_index(tail_id)
            rel_idx = self.kg.get_relation_index(rel_type)

            triples.append((head_idx, rel_idx, tail_idx))

        return triples

    def train(self, epochs=100, batch_size=128, negative_samples=5):
        """Train ComplEx on knowledge graph."""
        triples = self.prepare_training_data()
        n_triples = len(triples)

        for epoch in range(epochs):
            total_loss = 0
            np.random.shuffle(triples)

            for i in range(0, n_triples, batch_size):
                batch = triples[i:i + batch_size]

                # Positive triples
                pos_h = torch.tensor([h for h, r, t in batch])
                pos_r = torch.tensor([r for h, r, t in batch])
                pos_t = torch.tensor([t for h, r, t in batch])

                # Negative sampling
                neg_h = []
                neg_r = []
                neg_t = []

                for h, r, t in batch:
                    for _ in range(negative_samples):
                        if np.random.random() < 0.5:
                            # Corrupt head
                            neg_h.append(np.random.randint(0, self.model.num_entities))
                            neg_r.append(r)
                            neg_t.append(t)
                        else:
                            # Corrupt tail
                            neg_h.append(h)
                            neg_r.append(r)
                            neg_t.append(np.random.randint(0, self.model.num_entities))

                neg_h = torch.tensor(neg_h)
                neg_r = torch.tensor(neg_r)
                neg_t = torch.tensor(neg_t)

                # Forward pass
                pos_score = self.model.score(pos_h, pos_r, pos_t)
                neg_score = self.model.score(neg_h, neg_r, neg_t)

                # Loss: margin-based ranking
                margin = 1.0
                loss = torch.relu(margin - pos_score.unsqueeze(1) + neg_score.view(-1, negative_samples)).mean()

                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss = {total_loss / len(triples):.4f}")
```

**Training Requirements**:
- Minimum: 500 relationships
- Recommended: 5000+ relationships
- Training time: 1-4 hours on GPU

**Integration Points**:
1. Train on existing graph
2. Use for relationship prediction
3. Generate entity embeddings for similarity
4. Update weekly with new data

**Expected Impact**:
- 85-90% Hits@10 for link prediction
- Better relationship suggestions
- Entity embeddings capturing graph structure

---

### 2. Graph Traversal Algorithms (P2)

**Why**: Enable advanced query capabilities

**Implementation**:
```python
# backend/app/science/graph/traversal.py

class GraphTraversal:
    """Advanced graph traversal algorithms."""

    def __init__(self, kg):
        self.kg = kg

    def shortest_path(self, start_id, end_id, relationship_types=None, weight_by='confidence'):
        """
        Find shortest path between two nodes.

        Args:
            start_id: Starting node ID
            end_id: Target node ID
            relationship_types: Filter by relationship types
            weight_by: 'confidence' or 'type' for edge weights

        Returns:
            List of (node_id, relationship_type) tuples
        """
        from collections import deque

        # BFS with weights
        queue = deque([(start_id, [], 0)])  # (current, path, total_weight)
        visited = {start_id}

        while queue:
            current, path, total_weight = queue.popleft()

            if current == end_id:
                return {
                    'path': path + [end_id],
                    'weight': total_weight,
                    'hops': len(path) + 1
                }

            # Get neighbors
            neighbors = self.kg.get_neighbors(current, relationship_types)

            for neighbor, rel_type in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)

                    # Calculate edge weight
                    if weight_by == 'confidence':
                        rel = self.kg.get_relationship(current, neighbor, rel_type)
                        weight = rel.get('confidence', 0.5)
                    else:
                        weight = 1.0

                    new_path = path + [(current, rel_type)]
                    new_weight = total_weight + weight

                    queue.append((neighbor, new_path, new_weight))

        return None  # No path found

    def find_communities(self, algorithm='louvain'):
        """Detect communities in the graph."""
        if algorithm == 'louvain':
            return self.louvain_communities()
        elif algorithm == 'label_propagation':
            return self.label_propagation_communities()
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

    def louvain_communities(self):
        """Louvain modularity maximization."""
        import networkx as nx

        # Build NetworkX graph
        G = nx.Graph()

        # Add nodes
        for node in self.kg.get_all_nodes():
            G.add_node(node['id'], **node)

        # Add edges
        for rel in self.kg.get_all_relationships():
            G.add_edge(rel['from_id'], rel['to_id'],
                      weight=rel.get('confidence', 0.5))

        # Use community detection
        try:
            import community as community_louvain
            partition = community_louvain.best_partition(G, weight='weight')

            # Convert to communities
            communities = {}
            for node, comm_id in partition.items():
                if comm_id not in communities:
                    communities[comm_id] = []
                communities[comm_id].append(node)

            return communities
        except ImportError:
            print("python-louvain not installed. Install with: pip install python-louvain")
            return {}

    def centrality_measures(self, nodes=None):
        """Calculate centrality measures for nodes."""
        import networkx as nx

        # Build NetworkX graph
        G = nx.DiGraph()

        for node in self.kg.get_all_nodes():
            G.add_node(node['id'])

        for rel in self.kg.get_all_relationships():
            G.add_edge(rel['from_id'], rel['to_id'],
                      weight=rel.get('confidence', 0.5))

        # Calculate measures
        results = {}

        if nodes is None:
            nodes = list(G.nodes())

        # Degree centrality
        degree = nx.degree_centrality(G)

        # Betweenness centrality (sampled for large graphs)
        if len(G) > 1000:
            betweenness = nx.betweenness_centrality(G, k=100, weight='weight')
        else:
            betweenness = nx.betweenness_centrality(G, weight='weight')

        # PageRank
        pagerank = nx.pagerank(G, weight='weight')

        for node in nodes:
            results[node] = {
                'degree': degree.get(node, 0),
                'betweenness': betweenness.get(node, 0),
                'pagerank': pagerank.get(node, 0),
                'combined': (degree.get(node, 0) * 0.3 +
                           betweenness.get(node, 0) * 0.3 +
                           pagerank.get(node, 0) * 0.4)
            }

        return results

    def path_analysis(self, start_id, max_depth=3):
        """Analyze all paths from start_id up to max_depth."""
        from collections import defaultdict

        paths = defaultdict(list)

        def dfs(current, path, depth):
            if depth > max_depth:
                return

            if depth > 0:
                paths[depth].append(path.copy())

            if depth == max_depth:
                return

            neighbors = self.kg.get_neighbors(current)
            for neighbor, rel_type in neighbors:
                if neighbor not in [p[0] for p in path]:  # Avoid cycles
                    path.append((neighbor, rel_type))
                    dfs(neighbor, path, depth + 1)
                    path.pop()

        dfs(start_id, [], 0)
        return dict(paths)
```

**API Endpoints**:
```python
# Add to backend/app/api/routes/graph.py

@router.get("/graph/shortest-path")
async def get_shortest_path(
    start: str,
    end: str,
    relationship_types: Optional[List[str]] = None,
    weight_by: str = "confidence"
):
    """Find shortest path between two nodes."""
    traversal = GraphTraversal(graph_service)
    path = traversal.shortest_path(start, end, relationship_types, weight_by)

    if path:
        return {
            "success": True,
            "path": path['path'],
            "weight": path['weight'],
            "hops": path['hops']
        }
    else:
        return {
            "success": False,
            "message": "No path found"
        }

@router.get("/graph/communities")
async def get_communities(algorithm: str = "louvain"):
    """Detect communities in the graph."""
    traversal = GraphTraversal(graph_service)
    communities = traversal.find_communities(algorithm)

    return {
        "success": True,
        "algorithm": algorithm,
        "n_communities": len(communities),
        "communities": [
            {
                "id": comm_id,
                "size": len(nodes),
                "sample_nodes": nodes[:5]
            }
            for comm_id, nodes in communities.items()
        ]
    }

@router.get("/graph/centrality")
async def get_centrality(nodes: Optional[List[str]] = None):
    """Calculate centrality measures."""
    traversal = GraphTraversal(graph_service)
    centrality = traversal.centrality_measures(nodes)

    return {
        "success": True,
        "measures": centrality
    }
```

**Expected Impact**:
- Advanced graph exploration capabilities
- Community detection for topic analysis
- Centrality analysis for key concept identification
- Path discovery for argument analysis

---

### 3. Active Learning Pipeline (P2)

**Why**: Continuously improve from user feedback

**Implementation**:
```python
# backend/app/science/learning/active_learning.py

class ActiveLearningPipeline:
    """Active learning for KG construction and refinement."""

    def __init__(self, kg, ner_model, relation_model):
        self.kg = kg
        self.ner_model = ner_model
        self.relation_model = relation_model

        self.feedback_queue = []
        self.uncertainty_threshold = 0.2  # Review if uncertainty > 20%

    def process_with_uncertainty(self, text, metadata=None):
        """Process text and identify uncertain predictions."""
        # Entity extraction with uncertainty
        entities = self.ner_model.predict_with_uncertainty(text)

        # Relation extraction with uncertainty
        relations = self.relation_model.predict_with_uncertainty(text, entities)

        # Identify uncertain items
        uncertain_items = []

        for entity in entities:
            if entity['uncertainty'] > self.uncertainty_threshold:
                uncertain_items.append({
                    'type': 'entity',
                    'text': entity['text'],
                    'span': entity['span'],
                    'predicted_type': entity['type'],
                    'confidence': entity['confidence'],
                    'uncertainty': entity['uncertainty'],
                    'context': self.get_context(text, entity['span'])
                })

        for rel in relations:
            if rel['uncertainty'] > self.uncertainty_threshold:
                uncertain_items.append({
                    'type': 'relation',
                    'head': rel['head']['text'],
                    'tail': rel['tail']['text'],
                    'predicted_type': rel['type'],
                    'confidence': rel['confidence'],
                    'uncertainty': rel['uncertainty'],
                    'context': rel.get('evidence', '')
                })

        return {
            'predictions': {
                'entities': entities,
                'relations': relations
            },
            'uncertain_items': uncertain_items,
            'requires_review': len(uncertain_items) > 0
        }

    def request_user_feedback(self, items, user_id):
        """Present uncertain items to user for correction."""
        feedback_request = {
            'user_id': user_id,
            'timestamp': datetime.now(),
            'items': items,
            'responses': []
        }

        # Store for later collection
        self.feedback_queue.append(feedback_request)

        return feedback_request

    def collect_feedback(self, request_id, user_corrections):
        """Collect user corrections."""
        for feedback in self.feedback_queue:
            if feedback['request_id'] == request_id:
                feedback['responses'] = user_corrections
                self.process_feedback(feedback)
                return True
        return False

    def process_feedback(self, feedback):
        """Process user feedback and update models."""
        corrections = feedback['responses']

        # Separate into entity and relation corrections
        entity_corrections = []
        relation_corrections = []

        for corr in corrections:
            if corr['type'] == 'entity':
                entity_corrections.append({
                    'text': corr['context'],
                    'label': corr['correct_label']
                })
            elif corr['type'] == 'relation':
                relation_corrections.append({
                    'text': corr['context'],
                    'head': corr['head'],
                    'tail': corr['tail'],
                    'label': corr['correct_label']
                })

        # Update uncertainty trackers
        for corr in corrections:
            if corr['type'] == 'relation':
                # Update Beta distribution
                rel_type = corr['predicted_type']
                correct = corr['correct_label'] == corr['predicted_type']
                self.uncertainty_tracker.update(rel_type, correct)

        # Retrain if enough feedback
        if len(corrections) >= 10:
            self.retrain_models(entity_corrections, relation_corrections)

    def retrain_models(self, entity_data, relation_data):
        """Fine-tune models on new feedback."""
        print(f"Retraining with {len(entity_data)} entity and {len(relation_data)} relation examples")

        # Fine-tune NER
        if len(entity_data) >= 5:
            self.ner_model.fine_tune(entity_data, epochs=1)

        # Fine-tune relation extractor
        if len(relation_data) >= 5:
            self.relation_model.fine_tune(relation_data, epochs=1)

        # Clear processed feedback
        self.feedback_queue = [f for f in self.feedback_queue if not f.get('processed')]

    def get_prioritized_samples(self, n=10):
        """Get n most uncertain samples for labeling."""
        all_uncertain = []

        for feedback in self.feedback_queue:
            if 'items' in feedback:
                for item in feedback['items']:
                    all_uncertain.append({
                        'item': item,
                        'uncertainty': item.get('uncertainty', 0)
                    })

        # Sort by uncertainty
        all_uncertain.sort(key=lambda x: x['uncertainty'], reverse=True)

        return [x['item'] for x in all_uncertain[:n]]
```

**Integration Points**:
1. Hook into relationship suggestion acceptance/rejection
2. Present uncertain predictions for user review
3. Trigger retraining when enough feedback collected
4. Monitor improvement over time

**Expected Impact**:
- Continuous model improvement
- 20-30% reduction in uncertainty over 100 feedback cycles
- Better adaptation to user domain
- Reduced long-term maintenance

---

## Phase 4: Production System (Month 4-6)

### 1. Full Bayesian KG Embeddings (P3)

**Why**: State-of-the-art uncertainty in KG operations

**Implementation**:
```python
# backend/app/science/embeddings/bayesian_transE.py

class BayesianTransE(nn.Module):
    """Bayesian TransE with uncertainty in embeddings."""

    def __init__(self, num_entities, num_relations, dim=100):
        super().__init__()
        # Entity embeddings: mean + log variance
        self.entity_mu = nn.Embedding(num_entities, dim)
        self.entity_logvar = nn.Embedding(num_entities, dim)

        # Relation embeddings: mean + log variance
        self.relation_mu = nn.Embedding(num_relations, dim)
        self.relation_logvar = nn.Embedding(num_relations, dim)

        # Initialize
        nn.init.normal_(self.entity_mu.weight, mean=0, std=0.1)
        nn.init.normal_(self.relation_mu.weight, mean=0, std=0.1)
        nn.init.constant_(self.entity_logvar.weight, -5.0)  # Low variance
        nn.init.constant_(self.relation_logvar.weight, -5.0)

    def sample_embedding(self, embed_mu, embed_logvar):
        """Reparameterized sampling."""
        std = torch.exp(0.5 * embed_logvar)
        eps = torch.randn_like(std)
        return embed_mu + eps * std

    def score_with_uncertainty(self, head, relation, tail, n_samples=10):
        """Score triple with uncertainty estimates."""
        scores = []
        embeddings = []

        for _ in range(n_samples):
            h_emb = self.sample_embedding(
                self.entity_mu(head),
                self.entity_logvar(head)
            )
            r_emb = self.sample_embedding(
                self.relation_mu(relation),
                self.relation_logvar(relation)
            )
            t_emb = self.sample_embedding(
                self.entity_mu(tail),
                self.entity_logvar(tail)
            )

            # TransE scoring
            score = -torch.norm(h_emb + r_emb - t_emb, p=2)
            scores.append(score)

            embeddings.append({
                'head': h_emb.detach(),
                'relation': r_emb.detach(),
                'tail': t_emb.detach()
            })

        scores = torch.stack(scores)

        return {
            'mean_score': scores.mean().item(),
            'std_score': scores.std().item(),
            'scores': scores.detach().numpy(),
            'epistemic_uncertainty': scores.var().item(),  # Model uncertainty
            'samples': embeddings
        }

    def elbo(self, head, relation, tail, target=1.0):
        """Evidence Lower Bound for training."""
        # Reconstruction (scoring)
        score_stats = self.score_with_uncertainty(head, relation, tail, n_samples=1)
        reconstruction_loss = F.mse_loss(
            torch.tensor([score_stats['mean_score']]),
            torch.tensor([target])
        )

        # KL divergence
        kl_loss = 0
        for logvar in [self.entity_logvar, self.relation_logvar]:
            kl_loss += -0.5 * torch.sum(1 + logvar - logvar.exp() - logvar.pow(2))

        return reconstruction_loss + 0.01 * kl_loss

    def predict_tails_with_uncertainty(self, head, relation, top_k=10):
        """Predict possible tails with uncertainty."""
        predictions = []

        for tail_id in range(self.num_entities):
            stat = self.score_with_uncertainty(
                torch.tensor([head]),
                torch.tensor([relation]),
                torch.tensor([tail_id]),
                n_samples=20
            )

            predictions.append({
                'tail_id': tail_id,
                'mean_score': stat['mean_score'],
                'std_score': stat['std_score'],
                'uncertainty': stat['std_score'] / (abs(stat['mean_score']) + 1e-8),
                'confidence': 1 / (1 + stat['std_score'])  # Inverse uncertainty
            })

        return sorted(predictions, key=lambda x: x['mean_score'], reverse=True)[:top_k]
```

**Training Requirements**:
- GPU with 8GB+ memory
- 10K+ relationships for good uncertainty estimates
- Training time: 2-6 hours

**Expected Impact**:
- State-of-the-art link prediction (90-95% Hits@10)
- Well-calibrated uncertainty for predictions
- Ability to detect rare/ambiguous relationships

---

### 2. Uncertainty-Aware Decision System (P2)

**Why**: Automated decision making with risk management

**Implementation**:
```python
# backend/app/science/decision/uncertainty_decisions.py

class UncertaintyAwareDecisionSystem:
    """Make decisions considering uncertainty and risk."""

    def __init__(self, config=None):
        self.config = config or {
            'risk_aversion': 0.5,  # 0-1, higher = more conservative
            'auto_accept_threshold': 0.8,  # Minimum confidence for auto-accept
            'uncertainty_penalty': 0.3,  # Penalty for high uncertainty
            'review_threshold': 0.2,  # Uncertainty threshold for review
            'cost_fp': 1.0,  # Cost of false positive
            'cost_fn': 2.0,  # Cost of false negative
            'cost_review': 0.1  # Cost of human review
        }

    def evaluate_prediction(self, prediction):
        """Evaluate prediction with uncertainty."""
        confidence = prediction['confidence']
        uncertainty = prediction.get('uncertainty', 0)
        pred_type = prediction.get('type', 'unknown')

        # Risk-adjusted confidence
        risk_adjusted = confidence - self.config['risk_aversion'] * uncertainty

        # Expected utility (simplified)
        expected_utility = (
            confidence * (1 - self.config['cost_fp']) +
            (1 - confidence) * (1 - self.config['cost_fn'])
        ) - self.config['cost_review'] * (uncertainty > self.config['review_threshold'])

        # Decision
        if risk_adjusted >= self.config['auto_accept_threshold']:
            action = 'auto_accept'
        elif uncertainty > self.config['review_threshold']:
            action = 'require_review'
        else:
            action = 'auto_reject'

        return {
            'action': action,
            'confidence': confidence,
            'uncertainty': uncertainty,
            'risk_adjusted': risk_adjusted,
            'expected_utility': expected_utility,
            'requires_review': action == 'require_review'
        }

    def optimize_thresholds(self, validation_data):
        """Optimize decision thresholds on validation data."""
        from scipy.optimize import minimize

        def objective(thresholds):
            auto_threshold, review_threshold, risk_aversion = thresholds

            # Apply current thresholds
            decisions = []
            for item in validation_data:
                temp_config = self.config.copy()
                temp_config['auto_accept_threshold'] = auto_threshold
                temp_config['review_threshold'] = review_threshold
                temp_config['risk_aversion'] = risk_aversion

                system = UncertaintyAwareDecisionSystem(temp_config)
                decision = system.evaluate_prediction(item)
                decisions.append(decision)

            # Calculate cost
            total_cost = 0
            for decision, true_label in zip(decisions, validation_data):
                if decision['action'] == 'auto_accept':
                    if true_label['correct'] == False:
                        total_cost += self.config['cost_fp']
                elif decision['action'] == 'auto_reject':
                    if true_label['correct'] == True:
                        total_cost += self.config['cost_fn']
                else:  # require_review
                    total_cost += self.config['cost_review']

            return total_cost

        # Optimize
        result = minimize(
            objective,
            x0=[0.8, 0.2, 0.5],
            bounds=[(0.5, 1.0), (0.1, 0.5), (0.0, 1.0)]
        )

        return {
            'auto_accept_threshold': result.x[0],
            'review_threshold': result.x[1],
            'risk_aversion': result.x[2],
            'cost': result.fun
        }

    def batch_decision_making(self, predictions):
        """Make decisions for batch of predictions."""
        decisions = []
        review_items = []
        auto_accepted = []
        auto_rejected = []

        for pred in predictions:
            decision = self.evaluate_prediction(pred)
            decisions.append(decision)

            if decision['action'] == 'auto_accept':
                auto_accepted.append(pred)
            elif decision['action'] == 'require_review':
                review_items.append((pred, decision))
            else:
                auto_rejected.append(pred)

        return {
            'decisions': decisions,
            'auto_accepted': auto_accepted,
            'review_items': review_items,
            'auto_rejected': auto_rejected,
            'summary': {
                'auto_accept_rate': len(auto_accepted) / len(predictions),
                'review_rate': len(review_items) / len(predictions),
                'auto_reject_rate': len(auto_rejected) / len(predictions)
            }
        }

# API Integration
@router.post("/api/v1/tools/relationships/predict-batch-with-uncertainty")
async def predict_relationships_batch(request: RelationshipBatchRequest):
    """Batch predict relationships with uncertainty and decisions."""
    decision_system = UncertaintyAwareDecisionSystem()

    predictions = []
    for rel_request in request.relationships:
        # Get prediction with uncertainty
        pred = await tool_service.predict_relationship_with_uncertainty(
            rel_request.head,
            rel_request.tail,
            rel_request.context
        )
        predictions.append(pred)

    # Make decisions
    decisions = decision_system.batch_decision_making(predictions)

    # Auto-accept
    for pred, decision in zip(predictions, decisions['auto_accepted']):
        await graph_service.create_relationship(
            from_id=pred['head'],
            to_id=pred['tail'],
            rel_type=pred['type'],
            properties={
                'confidence': pred['confidence'],
                'uncertainty': pred.get('uncertainty', 0),
                'source': 'auto_predicted',
                'decision': 'auto_accept',
                'timestamp': datetime.now()
            }
        )

    return {
        'success': True,
        'predictions': predictions,
        'decisions': decisions,
        'auto_accepted': len(decisions['auto_accepted']),
        'review_needed': len(decisions['review_items']),
        'auto_rejected': len(decisions['auto_rejected'])
    }
```

**Expected Impact**:
- 60-80% automation rate (vs 0% currently)
- Human review only for uncertain cases (20-40%)
- Cost reduction of 70-90% in data entry
- Maintain or improve data quality

---

### 3. Continuous Learning & Monitoring (P3)

**Why**: Self-improving system with drift detection

**Implementation**:
```python
# backend/app/science/monitoring/continuous_learning.py

class ContinuousLearningSystem:
    """Monitor and continuously improve models."""

    def __init__(self, models, kg):
        self.models = models  # dict of all models
        self.kg = kg
        self.metrics_history = []
        self.drift_detectors = {}

        # Initialize drift detectors
        for model_name in models.keys():
            self.drift_detectors[model_name] = {
                'performance': [],
                'uncertainty': [],
                'data_distribution': []
            }

    def monitor_model_performance(self, model_name, metrics):
        """Track model performance over time."""
        entry = {
            'timestamp': datetime.now(),
            'model': model_name,
            'metrics': metrics
        }

        self.metrics_history.append(entry)
        self.drift_detectors[model_name]['performance'].append(metrics)

        # Check for drift
        drift_alert = self.check_drift(model_name)

        return drift_alert

    def check_drift(self, model_name, window=100):
        """Check for concept drift in model performance."""
        recent = self.drift_detectors[model_name]['performance'][-window:]

        if len(recent) < window // 2:
            return None  # Not enough data

        # Calculate statistics
        metrics = list(recent[0].keys())
        drift_detected = False
        drift_details = {}

        for metric in metrics:
            values = [m.get(metric, 0) for m in recent]

            # Check for significant change (simplified)
            baseline = np.mean(values[:window//2])
            recent_mean = np.mean(values[window//2:])

            change = abs(recent_mean - baseline) / (baseline + 1e-8)

            if change > 0.2:  # 20% change threshold
                drift_detected = True
                drift_details[metric] = {
                    'baseline': baseline,
                    'recent': recent_mean,
                    'change': change
                }

        if drift_detected:
            return {
                'alert': 'concept_drift_detected',
                'model': model_name,
                'details': drift_details,
                'recommendation': 'retrain_model'
            }

        return None

    def retrain_trigger(self, model_name, conditions=None):
        """Determine if model should be retrained."""
        if conditions is None:
            conditions = {
                'min_samples': 100,  # Minimum new samples
                'max_drift': 0.15,   # Maximum allowed drift
                'schedule': 'weekly'  # Retraining schedule
            }

        # Check data availability
        new_data_count = self.get_new_feedback_count(model_name)

        if new_data_count < conditions['min_samples']:
            return False, f"Insufficient new data: {new_data_count}/{conditions['min_samples']}"

        # Check drift
        drift = self.check_drift(model_name)
        if drift:
            drift_magnitude = max(d['change'] for d in drift['details'].values())
            if drift_magnitude > conditions['max_drift']:
                return True, f"Significant drift detected: {drift_magnitude:.2%}"

        # Check schedule
        if conditions['schedule'] == 'weekly':
            last_retrain = self.get_last_retrain_time(model_name)
            if last_retrain and (datetime.now() - last_retrain).days >= 7:
                return True, "Weekly retraining schedule reached"

        return False, "No retraining needed"

    def retrain_all_models(self):
        """Trigger retraining for all models as needed."""
        results = {}

        for model_name, model in self.models.items():
            should_retrain, reason = self.retrain_trigger(model_name)

            if should_retrain:
                print(f"Retraining {model_name}: {reason}")

                # Collect training data
                train_data = self.collect_training_data(model_name)

                # Retrain
                start_time = datetime.now()
                new_model = self.retrain_model(model, train_data)
                training_time = (datetime.now() - start_time).total_seconds()

                # Evaluate
                eval_results = self.evaluate_model(new_model, train_data['val'])

                # Deploy if improved
                if self.should_deploy(model_name, new_model, eval_results):
                    self.deploy_model(model_name, new_model)
                    results[model_name] = {
                        'status': 'retrained_and_deployed',
                        'training_time': training_time,
                        'evaluation': eval_results
                    }
                else:
                    results[model_name] = {
                        'status': 'retrained_but_not_deployed',
                        'reason': 'no_improvement',
                        'evaluation': eval_results
                    }
            else:
                results[model_name] = {
                    'status': 'not_retrained',
                    'reason': reason
                }

        return results

    def generate_monitoring_report(self):
        """Generate comprehensive monitoring report."""
        report = {
            'timestamp': datetime.now(),
            'models': {},
            'overall_health': 'healthy',
            'alerts': []
        }

        for model_name in self.models.keys():
            # Calculate key metrics
            recent_perf = self.drift_detectors[model_name]['performance'][-10:]
            if recent_perf:
                avg_perf = {k: np.mean([m.get(k, 0) for m in recent_perf])
                           for k in recent_perf[0].keys()}

                # Check calibration
                ece = avg_perf.get('ece', 0)
                health = 'healthy' if ece < 0.05 else 'needs_attention' if ece < 0.1 else 'unhealthy'

                report['models'][model_name] = {
                    'performance': avg_perf,
                    'health': health,
                    'n_samples': len(self.drift_detectors[model_name]['performance'])
                }

                if health != 'healthy':
                    report['alerts'].append({
                        'model': model_name,
                        'issue': f'Poor calibration (ECE={ece:.3f})',
                        'recommendation': 'retrain with calibration'
                    })

        # Overall health
        unhealthy = sum(1 for m in report['models'].values() if m['health'] != 'healthy')
        if unhealthy > len(self.models) // 2:
            report['overall_health'] = 'unhealthy'
        elif unhealthy > 0:
            report['overall_health'] = 'degraded'

        return report

# API for monitoring
@router.get("/api/v1/monitoring/health")
async def get_system_health():
    """Get overall system health dashboard."""
    monitor = ContinuousLearningSystem(
        models={
            'ner': ner_model,
            'relation': relation_model,
            'embeddings': embedding_model
        },
        kg=graph_service
    )

    report = monitor.generate_monitoring_report()

    return report

@router.post("/api/v1/monitoring/retrain")
async def trigger_retraining(model_name: Optional[str] = None):
    """Trigger model retraining."""
    monitor = ContinuousLearningSystem(
        models={
            'ner': ner_model,
            'relation': relation_model,
            'embeddings': embedding_model
        },
        kg=graph_service
    )

    if model_name:
        results = {model_name: monitor.retrain_all_models().get(model_name, {})}
    else:
        results = monitor.retrain_all_models()

    return {
        'success': True,
        'results': results
    }
```

**Expected Impact**:
- Automatic model improvement over time
- Early detection of performance degradation
- Continuous adaptation to user behavior
- Reduced manual monitoring effort

---

## Implementation Checklist

### Week 1: Foundation
- [ ] Install required ML libraries
- [ ] Set up directory structure
- [ ] Implement Beta uncertainty tracker
- [ ] Add uncertainty to API responses
- [ ] Create user feedback collection UI

### Week 2: Quick Wins
- [ ] Implement confidence calibration
- [ ] Add multi-source similarity
- [ ] A/B test against current system
- [ ] Monitor initial results

### Month 1: ML Enhancement
- [ ] Collect training data (500+ examples)
- [ ] Fine-tune BERT for NER
- [ ] Fine-tune BERT for relations
- [ ] Deploy HNSW vector index
- [ ] Implement MC dropout uncertainty

### Month 2: Graph Intelligence
- [ ] Train ComplEx embeddings
- [ ] Implement graph traversal algorithms
- [ ] Add community detection
- [ ] Build active learning pipeline
- [ ] Set up continuous retraining

### Month 3: Refinement
- [ ] Optimize decision thresholds
- [ ] Add uncertainty visualization
- [ ] Implement drift detection
- [ ] Scale to production workload
- [ ] Document algorithms for users

### Month 4-6: Advanced Features
- [ ] Bayesian TransE for KG embeddings
- [ ] Automated decision system
- [ ] Causal inference capabilities
- [ ] Research publications
- [ ] Community contributions

---

## Success Metrics

### Technical Metrics
- **Calibration**: ECE < 0.05 (well-calibrated)
- **Performance**: 85-90% F1-score on extraction tasks
- **Speed**: <100ms query latency for similarity
- **Scale**: Handle 100K+ entities
- **Accuracy**: 85-90% Hits@10 for link prediction

### User Metrics
- **Acceptance Rate**: >70% for automated suggestions
- **User Satisfaction**: >4.5/5 for relationship recommendations
- **Time Savings**: 50-80% reduction in manual data entry
- **Review Rate**: <40% requiring human review
- **Trust Score**: >4/5 for uncertainty estimates

### Business Metrics
- **Cost Reduction**: 70-90% in data entry costs
- **Graph Growth**: 5-10× increase in entities/relationships
- **Quality**: >90% precision on automated additions
- **Maintenance**: 50% reduction in manual oversight

---

## Common Pitfalls & Solutions

### Pitfall 1: Insufficient Training Data
**Symptom**: Models perform poorly, high uncertainty
**Solution**: Start with rule-based, collect corrections, fine-tune gradually

### Pitfall 2: Overfitting to Initial Data
**Symptom**: Model doesn't generalize to new data
**Solution**: Use regularization, cross-validation, active learning

### Pitfall 3: Poor Calibration
**Symptom**: 80% confidence means 60% accuracy
**Solution**: Implement temperature scaling, monitor ECE, retrain regularly

### Pitfall 4: Computational Bottlenecks
**Symptom**: Slow queries, high memory usage
**Solution**: Use HNSW, batch processing, GPU acceleration

### Pitfall 5: User Resistance
**Symptom**: Users don't trust automated suggestions
**Solution**: Show uncertainty, explain decisions, gradual rollout

---

## Resource Requirements

### Compute
- **Minimum**: CPU (4 cores), 8GB RAM
- **Recommended**: GPU (8GB VRAM), 16GB RAM
- **Production**: GPU (16GB+ VRAM), 32GB RAM, multiple instances

### Storage
- **Embeddings**: 1-10GB (depending on entity count)
- **Training Data**: 100MB-1GB
- **Models**: 500MB-2GB

### Time Investment
- **Research & Setup**: 1-2 weeks
- **Initial Implementation**: 2-4 weeks
- **Production Deployment**: 4-8 weeks
- **Continuous Improvement**: Ongoing (few hours/week)

### Personnel
- **ML Engineer**: 50% time for 2 months, then 20% time
- **Backend Developer**: 30% time for 2 months, then 10% time
- **Domain Expert**: 10% time for labeling/validation

---

## Next Steps

### Immediate (This Week)
1. Read [Uncertainty Quantification](./uncertainty_quantification.md) - Start here
2. Implement Beta uncertainty tracker
3. Set up user feedback collection

### Short-term (Next 2 Weeks)
1. Implement confidence calibration
2. Add multi-source similarity
3. A/B test improvements

### Medium-term (Month 1-2)
1. Fine-tune BERT models
2. Deploy HNSW index
3. Implement active learning

### Long-term (Month 3+)
1. Build complete uncertainty pipeline
2. Add graph intelligence features
3. Deploy production system

---

## Questions & Support

For questions about specific algorithms:
- Check individual research documents in this directory
- Review implementation code comments
- Consult academic sources linked in documents

For implementation guidance:
- Follow the phase-by-phase roadmap
- Use the priority matrix for planning
- Start with P0 items first

For performance issues:
- Check [Graph Traversal Algorithms](./graph_traversal_algorithms.md) for optimization
- Use HNSW for similarity search
- Implement caching for frequent queries

---

**Document Version**: 1.0
**Last Updated**: 2026-01-25
**Status**: All research complete, ready for implementation