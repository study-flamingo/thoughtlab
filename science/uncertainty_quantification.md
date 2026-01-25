# Uncertainty Quantification for Knowledge Graphs

**Date**: 2026-01-25
**Purpose**: Document methods for measuring and propagating uncertainty in knowledge graph operations

---

## Problem Statement

ThoughtLab currently uses simple confidence scores (0-1) on nodes and relationships, but this approach lacks:

1. **Calibration**: Is a 0.8 confidence score truly 80% likely to be correct?
2. **Propagation**: How does uncertainty in one relationship affect downstream inferences?
3. **Comparison**: How to compare confidence across different relationship types?
4. **Decision Making**: When should we trust automated suggestions vs. require human review?
5. **Explanation**: Why is something assigned a particular confidence score?

This research documents established methods for uncertainty quantification from probability theory, Bayesian statistics, and modern machine learning.

---

## 1. Probability Theory Foundations

### 1.1 Types of Uncertainty

**Aleatoric Uncertainty**: Inherent randomness in the system
- **Example**: Measurement noise, stochastic processes
- **Cannot be reduced** with more data
- **Characterizes**: Data variability

**Epistemic Uncertainty**: Lack of knowledge
- **Example**: Limited training data, model uncertainty
- **Can be reduced** with more data/experience
- **Characterizes**: Model ignorance

**For ThoughtLab**:
- **Aleatoric**: Inherent ambiguity in relationships ("RELATES_TO" is fuzzy)
- **Epistemic**: Limited examples for rare relationship types

---

### 1.2 Bayesian Probability

**Key Idea**: Probability as degree of belief, updated with evidence

**Bayes' Theorem**:
```
P(H|E) = P(E|H) × P(H) / P(E)
```

Where:
- **H** = Hypothesis (e.g., "This relationship SUPPORTS")
- **E** = Evidence (e.g., text content, embeddings)
- **P(H)** = Prior belief (base rate)
- **P(E|H)** = Likelihood (evidence given hypothesis)
- **P(H|E)** = Posterior belief (updated probability)

**Example for ThoughtLab**:
```python
# Prior: Base probability of SUPPORTS relationship
P_supports = 0.3  # 30% of all relationships are SUPPORTS

# Likelihood: Probability of this text given SUPPORTS
# (Learned from training data)
P(text|supports) = 0.8  # Typical SUPPORTS text

# Evidence: The text we observed
P(text) = 0.25  # Marginal probability of this text

# Posterior: Updated probability
P(supports|text) = (0.8 × 0.3) / 0.25 = 0.96  # 96% confidence
```

**Bayesian Updating**:
```python
def bayesian_update(prior, likelihood, evidence):
    """Update belief with new evidence."""
    posterior = (likelihood * prior) / evidence
    return posterior

# As new evidence arrives, keep updating
belief = prior
for evidence in new_evidence:
    likelihood = calculate_likelihood(evidence)
    belief = bayesian_update(belief, likelihood, evidence)
```

**Advantages**:
- Naturally handles uncertainty
- Provides calibrated probabilities
- Updates with new data

**Sources**:
- *Bayesian Reasoning and Machine Learning* (Barber, 2012)
- *Probabilistic Machine Learning* (Murphy, 2020)

---

### 1.3 Probability Distributions

**Continuous Uncertainty**: Use distributions instead of point estimates

**Common Distributions**:

#### Beta Distribution (for 0-1 probabilities)
```
P(p) = Beta(p; α, β) = [Γ(α+β) / (Γ(α)Γ(β))] × p^(α-1) × (1-p)^(β-1)
```

Where:
- **p** = probability value (0-1)
- **α** = successes + 1
- **β** = failures + 1
- **Mean** = α / (α + β)
- **Variance** = αβ / [(α+β)²(α+β+1)]

**Interpretation**:
- α, β = "pseudo-counts" of evidence
- Large α+β = high confidence (narrow distribution)
- Small α+β = low confidence (wide distribution)

**For ThoughtLab**:
```python
class BetaUncertainty:
    def __init__(self, successes=1, failures=1):
        """Initialize with Laplace smoothing."""
        self.alpha = successes
        self.beta = failures

    def update(self, new_evidence):
        """Update with new observation."""
        if new_evidence['correct']:
            self.alpha += 1
        else:
            self.beta += 1

    def mean(self):
        """Expected confidence."""
        return self.alpha / (self.alpha + self.beta)

    def variance(self):
        """Uncertainty in confidence."""
        return (self.alpha * self.beta) / \
               ((self.alpha + self.beta) ** 2 * (self.alpha + self.beta + 1))

    def credible_interval(self, confidence=0.95):
        """95% credible interval for p."""
        from scipy.stats import beta
        lower = beta.ppf((1 - confidence) / 2, self.alpha, self.beta)
        upper = beta.ppf(1 - (1 - confidence) / 2, self.alpha, self.beta)
        return (lower, upper)

    def probability_gt(self, threshold=0.5):
        """Probability that p > threshold."""
        from scipy.stats import beta
        return 1 - beta.cdf(threshold, self.alpha, self.beta)
```

**Example Usage**:
```python
# Relationship classifier with uncertainty
relation_uncertainty = {
    'SUPPORTS': BetaUncertainty(successes=10, failures=3),  # 77% confidence
    'CONTRADICTS': BetaUncertainty(successes=5, failures=2),  # 71% confidence
    'RELATES_TO': BetaUncertainty(successes=20, failures=8)  # 71% confidence
}

# Check if relationship is trustworthy
for rel_type, dist in relation_uncertainty.items():
    prob_high_confidence = dist.probability_gt(threshold=0.7)
    print(f"{rel_type}: {prob_high_confidence:.2%} probability > 70% confidence")
```

---

#### Gaussian Distribution (for continuous values)
```
P(x) = N(x; μ, σ²) = (1/√(2πσ²)) × exp(-½ × (x-μ)²/σ²)
```

**For embedding similarities**:
```python
class GaussianUncertainty:
    def __init__(self, mean, variance):
        self.mean = mean
        self.variance = variance

    def update(self, new_measurement, measurement_variance):
        """Bayesian update of Gaussian (Kalman filter)."""
        prior_precision = 1.0 / self.variance
        measurement_precision = 1.0 / measurement_variance

        # Update mean (precision-weighted average)
        posterior_precision = prior_precision + measurement_precision
        posterior_mean = (prior_precision * self.mean +
                         measurement_precision * new_measurement) / posterior_precision

        # Update variance
        posterior_variance = 1.0 / posterior_precision

        return GaussianUncertainty(posterior_mean, posterior_variance)

    def credible_interval(self, confidence=0.95):
        """95% credible interval."""
        from scipy.stats import norm
        z = norm.ppf(1 - (1 - confidence) / 2)
        return (self.mean - z * np.sqrt(self.variance),
                self.mean + z * np.sqrt(self.variance))
```

---

### 1.4 Dirichlet Distribution (for categorical uncertainty)

**For relationship type classification**:
```
P(θ) = Dir(θ; α) = [Γ(Σαᵢ) / ΠΓ(αᵢ)] × Π θᵢ^(αᵢ-1)
```

Where:
- **θ** = probability vector over relationship types
- **α** = concentration parameters (pseudo-counts)

**Implementation**:
```python
class DirichletUncertainty:
    def __init__(self, categories, alpha_prior=1.0):
        """Initialize with uniform prior."""
        self.categories = categories
        self.alpha = {cat: alpha_prior for cat in categories}

    def update(self, observations):
        """Update with new observations."""
        for cat, count in observations.items():
            self.alpha[cat] += count

    def mean(self):
        """Expected probability vector."""
        total = sum(self.alpha.values())
        return {cat: alpha / total for cat, alpha in self.alpha.items()}

    def uncertainty(self):
        """Measure of overall uncertainty (entropy)."""
        import numpy as np
        probs = list(self.mean().values())
        probs = np.array(probs) / np.sum(probs)
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        return entropy

    def predictive_distribution(self):
        """Posterior predictive distribution (new observation)."""
        total = sum(self.alpha.values())
        return {cat: alpha / total for cat, alpha in self.alpha.items()}
```

**Example**: Relationship type classification
```python
# Track uncertainty in relationship classification
classifier = DirichletUncertainty(['SUPPORTS', 'CONTRADICTS', 'RELATES_TO'])

# Observations from training
classifier.update({'SUPPORTS': 50, 'CONTRADICTS': 30, 'RELATES_TO': 120})

# Expected probabilities
probs = classifier.mean()
print(f"P(SUPPORTS) = {probs['SUPPORTS']:.2f}")
print(f"P(CONTRADICTS) = {probs['CONTRADICTS']:.2f}")
print(f"P(RELATES_TO) = {probs['RELATES_TO']:.2f}")

# Overall uncertainty
entropy = classifier.uncertainty()
print(f"Uncertainty (entropy) = {entropy:.2f}")
```

---

## 2. Bayesian Neural Networks

### 2.1 Motivation

**Standard neural networks**: Point estimates
```
y = f(x; θ)  # Fixed weights θ
```

**Bayesian neural networks**: Distributions over weights
```
p(y|x, D) = ∫ p(y|x, θ) p(θ|D) dθ
```

**Benefits**:
- Predictive uncertainty
- Prevents overfitting
- Model selection via marginal likelihood

---

### 2.2 Bayesian Neural Network Architecture

**Weights as distributions**:
```python
class BayesianLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        # Mean and variance for weights
        self.w_mu = nn.Parameter(torch.randn(out_dim, in_dim) * 0.1)
        self.w_rho = nn.Parameter(torch.ones(out_dim, in_dim) * -5.0)

        # Mean and variance for bias
        self.b_mu = nn.Parameter(torch.zeros(out_dim))
        self.b_rho = nn.Parameter(torch.ones(out_dim) * -5.0)

    def reparameterize(self, mu, rho):
        """Sample from Gaussian using reparameterization."""
        epsilon = torch.randn_like(mu)
        return mu + torch.log1p(torch.exp(rho)) * epsilon

    def forward(self, x, sample=True):
        if sample:
            w = self.reparameterize(self.w_mu, self.w_rho)
            b = self.reparameterize(self.b_mu, self.b_rho)
        else:
            w = self.w_mu
            b = self.b_mu

        return F.linear(x, w, b)

    def kl_divergence(self):
        """KL divergence between weights and prior."""
        # Prior: N(0, 1)
        kl_w = -0.5 * (1 + self.w_rho - self.w_mu.pow(2) - torch.exp(2*self.w_rho)).sum()
        kl_b = -0.5 * (1 + self.b_rho - self.b_mu.pow(2) - torch.exp(2*self.b_rho)).sum()
        return kl_w + kl_b
```

**Bayesian Neural Network**:
```python
class BayesianNN(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(BayesianLinear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        layers.append(BayesianLinear(prev_dim, output_dim))
        self.layers = nn.ModuleList(layers)

    def forward(self, x, sample=True):
        for layer in self.layers:
            if isinstance(layer, BayesianLinear):
                x = layer(x, sample=sample)
            else:
                x = layer(x)
        return x

    def elbo(self, x, y, n_samples=1):
        """Evidence Lower Bound for training."""
        total_loss = 0
        total_kl = 0

        for _ in range(n_samples):
            y_pred = self.forward(x, sample=True)

            # Likelihood (negative log likelihood)
            likelihood = F.binary_cross_entropy_with_logits(y_pred, y, reduction='sum')

            # KL divergence
            kl = sum(layer.kl_divergence() for layer in self.layers
                    if isinstance(layer, BayesianLinear))

            total_loss += likelihood + kl
            total_kl += kl

        return total_loss / n_samples, total_kl / n_samples
```

**Prediction with Uncertainty**:
```python
def predict_with_uncertainty(bnn, x, n_samples=100):
    """Generate predictions with uncertainty estimates."""
    predictions = []

    bnn.train()  # Enable dropout/sampling
    with torch.no_grad():
        for _ in range(n_samples):
            pred = bnn.forward(x, sample=True)
            predictions.append(pred)

    bnn.eval()

    # Calculate statistics
    predictions = torch.stack(predictions)  # [n_samples, batch_size, output_dim]

    mean = predictions.mean(dim=0)
    variance = predictions.var(dim=0)
    std = predictions.std(dim=0)

    # For classification: get class probabilities
    if mean.shape[-1] > 1:
        probs = torch.softmax(mean, dim=-1)
        uncertainty = 1 - torch.max(probs, dim=-1)[0]  # 1 - max probability
    else:
        probs = torch.sigmoid(mean)
        uncertainty = std

    return {
        'mean': mean,
        'variance': variance,
        'std': std,
        'uncertainty': uncertainty,
        'predictions': predictions
    }
```

**Performance**: Provides well-calibrated uncertainty estimates

**Training Objective**: Maximize ELBO
```
ELBO = E[log p(y|x,θ)] - KL[q(θ)||p(θ)]
```

**Sources**:
- *Weight Uncertainty in Neural Networks* (Blundell et al., 2015)
- *Bayesian Neural Networks: An Introduction* (2020)

---

### 2.3 Monte Carlo Dropout

**Approximate Bayesian inference** using dropout at test time

**Concept**: Dropout can be viewed as approximate Bayesian inference

**Implementation**:
```python
class MCDropoutModel(nn.Module):
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
        self.layers = nn.Sequential(*layers)

    def forward(self, x, train=True):
        if train:
            self.train()
        else:
            self.eval()
        return self.layers(x)

    def predict_with_uncertainty(self, x, n_samples=50):
        """Monte Carlo sampling with dropout."""
        self.train()  # Keep dropout active
        predictions = []

        with torch.no_grad():
            for _ in range(n_samples):
                pred = self.forward(x, train=False)
                predictions.append(pred)

        predictions = torch.stack(predictions)
        mean = predictions.mean(dim=0)
        variance = predictions.var(dim=0)

        return {
            'mean': mean,
            'variance': variance,
            'std': predictions.std(dim=0)
        }
```

**Advantages**:
- Simple to implement
- No architectural changes
- Good uncertainty estimates

**Disadvantages**:
- Less calibrated than true Bayesian methods
- Requires careful dropout rate tuning

**Sources**:
- *Dropout as a Bayesian Approximation* (Gal & Ghahramani, 2016)

---

### 2.4 Deep Ensembles

**Train multiple models**, average predictions

**Concept**: Uncertainty from model diversity

**Implementation**:
```python
class DeepEnsemble:
    def __init__(self, n_models=5, model_fn=None):
        self.n_models = n_models
        self.models = [model_fn() for _ in range(n_models)]

    def train(self, train_data, epochs=100):
        """Train each model with different initialization."""
        for i, model in enumerate(self.models):
            print(f"Training model {i+1}/{self.n_models}")

            # Different random seeds for diversity
            torch.manual_seed(42 + i)

            # Train model (with different subsets if desired)
            self.train_single_model(model, train_data, epochs)

    def predict_with_uncertainty(self, x, n_samples=10):
        """Predict with ensemble uncertainty."""
        predictions = []

        self.eval()
        with torch.no_grad():
            for model in self.models:
                # Multiple forward passes for Bayesian models
                for _ in range(n_samples):
                    pred = model(x)
                    predictions.append(pred)

        predictions = torch.stack(predictions)
        mean = predictions.mean(dim=0)
        variance = predictions.var(dim=0)

        return {
            'mean': mean,
            'variance': variance,
            'std': predictions.std(dim=0),
            'epistemic': variance,  # Model uncertainty
            'aleatoric': torch.zeros_like(variance)  # Would need noise estimation
        }
```

**Advantages**:
- Simple, embarrassingly parallel
- Often outperforms single models
- Good uncertainty estimates

**Disadvantages**:
- N× training cost
- N× model size

**Sources**:
- *Deep Ensembles: A Loss Landscape Perspective* (2020)
- *Simple and Scalable Predictive Uncertainty Estimation* (2017)

---

## 3. Probabilistic Soft Logic (PSL)

### 3.1 Motivation

**Traditional logic**: Binary truth (True/False)
**Probabilistic logic**: Continuous truth values (0-1)

**Applications**:
- Uncertain relationships
- Conflicting evidence
- Transitive reasoning with uncertainty

---

### 3.2 Logic Rules with Uncertainty

**Example rules for ThoughtLab**:
```python
rules = [
    # Direct relationship: confidence preserved
    ("SUPPORTS(x,y) ⇒ RELATES_TO(x,y)", weight=0.9),

    # Transitivity: weaker confidence
    ("SUPPORTS(x,y) ∧ SUPPORTS(y,z) ⇒ SUPPORTS(x,z)", weight=0.7),

    # Contradiction detection
    ("SUPPORTS(x,y) ∧ CONTRADICTS(x,y) ⇒ False", weight=1.0),

    # Similarity suggests relationship
    ("SIMILAR(x,y) ⇒ RELATES_TO(x,y)", weight=0.6),

    # Evidence aggregation
    ("EVIDENCE(x,y,e1) ∧ EVIDENCE(x,y,e2) ⇒ SUPPORTS(x,y)", weight=0.8),
]
```

**Logic Operators**:
```
∧ (AND): min(a, b)
∨ (OR): max(a, b)
¬ (NOT): 1 - a
→ (IMPLIES): min(1, 1 - a + b)  # Łukasiewicz logic
```

**PSL Framework**:
```python
class PSLInference:
    def __init__(self, rules):
        self.rules = rules
        self.variables = {}  # Truth values of predicates

    def set_evidence(self, predicate, value):
        """Set observed truth value (0-1)."""
        self.variables[predicate] = value

    def compute_inference(self):
        """Iterative inference (similar to belief propagation)."""
        for iteration in range(100):
            changes = 0
            for rule, weight in self.rules:
                # Parse rule (simplified)
                head, body = rule.split(' ⇒ ')

                # Evaluate body
                body_value = self.evaluate_expression(body)

                # Update head
                head_value = self.variables.get(head, 0)
                new_value = min(1, 1 - body_value + weight)

                if abs(new_value - head_value) > 0.01:
                    self.variables[head] = new_value
                    changes += 1

            if changes == 0:
                break

    def evaluate_expression(self, expr):
        """Evaluate logical expression with truth values."""
        # Simplified evaluator
        # In practice: use proper parser
        if '∧' in expr:
            parts = expr.split('∧')
            return min(self.variables.get(p.strip(), 0) for p in parts)
        elif '∨' in expr:
            parts = expr.split('∨')
            return max(self.variables.get(p.strip(), 0) for p in parts)
        else:
            return self.variables.get(expr.strip(), 0)
```

**Application to ThoughtLab**:
```python
# Create PSL model for relationship confidence
psl = PSLInference(rules)

# Set evidence from text analysis
psl.set_evidence("TEXT_CONTAINS(h1, 'experimental results')", 0.9)
psl.set_evidence("TEXT_CONTAINS(h2, 'theory predicts')", 0.8)
psl.set_evidence("CITES(paper1, paper2)", 0.95)

# Compute inference
psl.compute_inference()

# Get relationship confidence
support_confidence = psl.variables.get("SUPPORTS(h1, h2)", 0)
print(f"Inferred SUPPORTS confidence: {support_confidence:.2f}")
```

**Sources**:
- *Probabilistic Soft Logic* (Koller et al., 2009)
- *PSL: A Python Framework for Probabilistic Logic Programming* (2020)

---

## 4. Uncertainty in Knowledge Graph Embeddings

### 4.1 Bayesian TransE

**Embeddings as distributions**:
```python
class BayesianTransE(nn.Module):
    def __init__(self, n_entities, n_relations, dim=100):
        super().__init__()
        # Entity embeddings: mean + variance
        self.entity_mu = nn.Embedding(n_entities, dim)
        self.entity_logvar = nn.Embedding(n_entities, dim)

        # Relation embeddings: mean + variance
        self.relation_mu = nn.Embedding(n_relations, dim)
        self.relation_logvar = nn.Embedding(n_relations, dim)

    def sample_embedding(self, embed_mu, embed_logvar):
        """Reparameterized sampling."""
        std = torch.exp(0.5 * embed_logvar)
        eps = torch.randn_like(std)
        return embed_mu + eps * std

    def score(self, head, relation, tail, n_samples=10):
        """Score triple with uncertainty."""
        scores = []

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

            score = -torch.norm(h_emb + r_emb - t_emb, p=2)
            scores.append(score)

        scores = torch.stack(scores)
        return {
            'mean': scores.mean(),
            'std': scores.std(),
            'samples': scores
        }

    def elbo(self, head, relation, tail, target=1.0):
        """Evidence Lower Bound."""
        # Reconstruction (scoring)
        score_stats = self.score(head, relation, tail)
        reconstruction_loss = F.mse_loss(score_stats['mean'],
                                        torch.tensor(target).float())

        # KL divergence (between embeddings and standard normal)
        kl_loss = 0
        for embed_logvar in [self.entity_logvar, self.relation_logvar]:
            kl_loss += -0.5 * torch.sum(1 + embed_logvar -
                                       embed_logvar.exp() -
                                       embed_logvar.pow(2))

        return reconstruction_loss + 0.1 * kl_loss
```

**Prediction with Uncertainty**:
```python
def predict_relation_with_uncertainty(model, head, tail, n_samples=100):
    """Predict relationship with uncertainty."""
    scores = {}
    for rel_id in range(model.n_relations):
        scores[rel_id] = model.score(head, rel_id, tail, n_samples)

    # Calculate probabilities
    mean_scores = torch.stack([s['mean'] for s in scores.values()])
    std_scores = torch.stack([s['std'] for s in scores.values()])

    # Convert to probabilities
    probs = torch.softmax(-mean_scores, dim=0)  # Lower score = higher prob
    uncertainty = std_scores / (mean_scores.abs() + 1e-8)  # Relative uncertainty

    return {
        'probabilities': probs,
        'uncertainty': uncertainty,
        'top_prediction': probs.argmax().item(),
        'confidence': probs.max().item()
    }
```

**Sources**:
- *Bayesian Knowledge Graph Embeddings* (2019)
- *Uncertainty in Knowledge Graph Embeddings* (2020)

---

### 4.2 Probabilistic Box Embeddings

**Model entities as boxes (hyperrectangles)** in embedding space

**Key Insight**: Intersection volume = probability

**Implementation**:
```python
class BoxEmbedding:
    def __init__(self, entity_id, center, width):
        self.entity_id = entity_id
        self.center = center  # Center point
        self.width = width    # Width in each dimension

    def contains(self, point):
        """Probability that point is inside box."""
        # Calculate distance from center
        distances = torch.abs(point - self.center)

        # Smooth containment (sigmoid of negative distance)
        # Large negative distance = high probability inside
        probabilities = torch.sigmoid(-distances / (self.width + 1e-8))

        # All dimensions must contain point
        return torch.prod(probabilities)

    def intersection_volume(self, other):
        """Calculate intersection volume with another box."""
        # Calculate overlap in each dimension
        left = torch.maximum(self.center - self.width/2,
                           other.center - other.width/2)
        right = torch.minimum(self.center + self.width/2,
                            other.center + other.width/2)

        overlap = torch.maximum(right - left, torch.zeros_like(right))

        # Volume = product of overlaps
        return torch.prod(overlap)

    def relation_volume(self, relation_box):
        """Probability that relation holds (intersection volume)."""
        # Translate box by relation
        translated = BoxEmbedding(
            self.entity_id,
            self.center + relation_box.center,
            self.width + relation_box.width
        )

        return self.intersection_volume(translated)

    def uncertainty_estimate(self, point):
        """How certain are we about this point's location."""
        # Low variance = narrow box = certain
        # High variance = wide box = uncertain
        variance = torch.mean(self.width ** 2)
        return variance
```

**Box-Based Reasoning**:
```python
class BoxKGReasoner:
    def __init__(self):
        self.entity_boxes = {}  # entity_id -> BoxEmbedding
        self.relation_boxes = {}  # relation_id -> BoxEmbedding

    def predict(self, head, relation, tail=None):
        """Predict probability of relationship."""
        head_box = self.entity_boxes[head]
        relation_box = self.relation_boxes[relation]

        if tail is not None:
            tail_box = self.entity_boxes[tail]
            prob = head_box.relation_volume(relation_box) * \
                   tail_box.intersection_volume(head_box.relation_volume(relation_box))
            return prob
        else:
            # Score all possible tails
            scores = {}
            for tail_id, tail_box in self.entity_boxes.items():
                scores[tail_id] = head_box.relation_volume(relation_box) * \
                                 tail_box.intersection_volume(
                                     head_box.relation_volume(relation_box)
                                 )
            return scores

    def uncertainty(self, head, relation):
        """How uncertain are we about this relationship."""
        head_box = self.entity_boxes[head]
        relation_box = self.relation_boxes[relation]

        # Combined uncertainty
        uncertainty = (head_box.uncertainty_estimate(torch.zeros_like(head_box.center)) +
                      relation_box.uncertainty_estimate(torch.zeros_like(relation_box.center))) / 2
        return uncertainty
```

**Advantages**:
- Natural uncertainty via box width
- Compositional reasoning
- Interpretable (boxes can be visualized)

**Sources**:
- *Probabilistic Box Embeddings for Uncertain KGs* (2021)
- *Box Embeddings for Knowledge Graph Completion* (2020)

---

## 5. Uncertainty Propagation in Reasoning

### 5.1 Belief Propagation

**Propagate uncertainty through graph structure**

**Algorithm**:
```python
class BeliefPropagation:
    def __init__(self, kg):
        self.kg = kg
        self.beliefs = {}  # Current beliefs for each node
        self.messages = {}  # Messages between nodes

    def initialize(self):
        """Initialize beliefs with observed evidence."""
        for node in self.kg.get_all_nodes():
            self.beliefs[node['id']] = {
                'type': node.get('confidence', 0.5),
                'uncertainty': node.get('uncertainty', 0.5)
            }

    def send_message(self, from_node, to_node, relation_type):
        """Send belief message along relationship."""
        from_belief = self.beliefs[from_node]
        relation_confidence = self.kg.get_relationship_confidence(
            from_node, to_node, relation_type
        )

        # Message combines belief and relationship strength
        message = {
            'strength': from_belief['type'] * relation_confidence,
            'uncertainty': 1 - ((1 - from_belief['uncertainty']) *
                               (1 - relation_confidence))
        }

        return message

    def update_beliefs(self, iterations=10):
        """Iterative belief updates."""
        for iteration in range(iterations):
            new_beliefs = {}

            for node in self.kg.get_all_nodes():
                # Get all incoming messages
                incoming = []
                for neighbor, rel_type in self.kg.get_incoming(node['id']):
                    msg = self.send_message(neighbor, node['id'], rel_type)
                    incoming.append(msg)

                if incoming:
                    # Combine messages (weighted average)
                    total_strength = sum(m['strength'] for m in incoming)
                    total_uncertainty = sum(m['uncertainty'] for m in incoming)
                    n = len(incoming)

                    new_beliefs[node['id']] = {
                        'type': total_strength / n,
                        'uncertainty': total_uncertainty / n
                    }
                else:
                    new_beliefs[node['id']] = self.beliefs[node['id']]

            # Check convergence
            if self.has_converged(new_beliefs):
                break

            self.beliefs = new_beliefs

        return self.beliefs

    def has_converged(self, new_beliefs, threshold=0.01):
        """Check if beliefs have converged."""
        for node_id in self.beliefs:
            old = self.beliefs[node_id]
            new = new_beliefs[node_id]

            if (abs(old['type'] - new['type']) > threshold or
                abs(old['uncertainty'] - new['uncertainty']) > threshold):
                return False
        return True
```

**Application**: Propagate confidence from well-supported nodes to connected nodes

---

### 5.2 Causal Inference with Uncertainty

**Estimate causal relationships with uncertainty bounds**

**Potential Outcomes Framework**:
```
ATE = E[Y(1) - Y(0)]  # Average Treatment Effect
```

**For Knowledge Graphs**:
```python
class CausalInference:
    def __init__(self, kg):
        self.kg = kg

    def estimate_effect(self, treatment, outcome, confounders=None):
        """Estimate causal effect of treatment on outcome."""
        # Get treated group (has relationship)
        treated = self.kg.get_nodes_with_relationship(treatment)

        # Get control group (no relationship)
        control = self.kg.get_nodes_without_relationship(treatment)

        # Adjust for confounders if provided
        if confounders:
            treated = self.adjust_for_confounders(treated, confounders)
            control = self.adjust_for_confounders(control, confounders)

        # Estimate outcomes
        treated_outcomes = [self.kg.get_outcome(node, outcome) for node in treated]
        control_outcomes = [self.kg.get_outcome(node, outcome) for node in control]

        # Calculate ATE with uncertainty
        ate = np.mean(treated_outcomes) - np.mean(control_outcomes)

        # Bootstrap for confidence intervals
        n_bootstraps = 1000
        bootstrap_effects = []

        for _ in range(n_bootstraps):
            treated_sample = np.random.choice(treated_outcomes, size=len(treated_outcomes), replace=True)
            control_sample = np.random.choice(control_outcomes, size=len(control_outcomes), replace=True)
            effect = np.mean(treated_sample) - np.mean(control_sample)
            bootstrap_effects.append(effect)

        ci_lower = np.percentile(bootstrap_effects, 2.5)
        ci_upper = np.percentile(bootstrap_effects, 97.5)

        return {
            'effect': ate,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'p_value': self.calculate_p_value(treated_outcomes, control_outcomes)
        }
```

**Sources**:
- *Causal Inference in Statistics* (Pearl, 2009)
- *Causal Reasoning with Knowledge Graphs* (2020)

---

## 6. Uncertainty Calibration

### 6.1 Problem: Miscalibrated Confidence

**Issue**: Model says "80% confidence" but is only correct 60% of time

**Visual**:
```
Predicted Confidence | Actual Accuracy
-------------------- | ---------------
0.9-1.0              | 0.72
0.8-0.9              | 0.65
0.7-0.8              | 0.58
0.6-0.7              | 0.52
0.5-0.6              | 0.48
```

**Need**: Calibration - 80% confidence should mean 80% accuracy

---

### 6.2 Temperature Scaling

**Simple post-hoc calibration method**

**Idea**: Calibrate model outputs with temperature parameter

**Implementation**:
```python
class TemperatureScaling:
    def __init__(self, model):
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1))

    def set_temperature(self, val_loader):
        """Optimize temperature on validation set."""
        nll_criterion = nn.CrossEntropyLoss()

        # Freeze model, only optimize temperature
        for param in self.model.parameters():
            param.requires_grad = False

        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        def eval_nll():
            loss = 0
            n = 0
            for inputs, labels in val_loader:
                with torch.no_grad():
                    logits = self.model(inputs)
                loss += nll_criterion(logits / self.temperature, labels).item()
                n += len(labels)
            return loss / n

        optimizer.step(eval_nll)

        return self.temperature.item()

    def predict(self, x):
        """Get calibrated predictions."""
        self.model.eval()
        with torch.no_grad():
            logits = self.model(x)
            calibrated = logits / self.temperature
            probs = torch.softmax(calibrated, dim=1)
            return probs
```

**Evaluation**:
```python
def evaluate_calibration(predictions, labels, n_bins=10):
    """Evaluate calibration using Expected Calibration Error (ECE)."""
    # Bin predictions by confidence
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    confidences, accuracies = predictions.max(dim=1)
    confidences = confidences.cpu().numpy()
    accuracies = (accuracies == labels).cpu().numpy()

    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece
```

**Sources**:
- *On Calibration of Modern Neural Networks* (Guo et al., 2017)

---

### 6.3 Platt Scaling

**Logistic regression to calibrate scores**

```python
class PlattScaling:
    def __init__(self):
        self.A = nn.Parameter(torch.tensor(1.0))
        self.B = nn.Parameter(torch.tensor(0.0))

    def fit(self, scores, labels):
        """Fit logistic regression on validation scores."""
        optimizer = torch.optim.Adam([self.A, self.B], lr=0.01)

        for epoch in range(100):
            optimizer.zero_grad()
            calibrated = self.A * scores + self.B
            probs = torch.sigmoid(calibrated)
            loss = F.binary_cross_entropy(probs, labels)
            loss.backward()
            optimizer.step()

    def predict(self, scores):
        """Apply calibration."""
        calibrated = self.A * scores + self.B
        return torch.sigmoid(calibrated)
```

---

### 6.4 Isotonic Regression

**Non-parametric calibration method**

```python
from sklearn.isotonic import IsotonicRegression

class IsotonicCalibration:
    def __init__(self):
        self.regressor = IsotonicRegression(out_of_bounds='clip')

    def fit(self, scores, labels):
        """Fit isotonic regression."""
        scores_np = scores.cpu().numpy()
        labels_np = labels.cpu().numpy()
        self.regressor.fit(scores_np, labels_np)

    def predict(self, scores):
        """Apply calibration."""
        scores_np = scores.cpu().numpy()
        calibrated = self.regressor.predict(scores_np)
        return torch.tensor(calibrated)
```

---

## 7. Uncertainty in Relationships

### 7.1 Multi-Source Uncertainty

**Combine uncertainty from multiple sources**:
```python
class MultiSourceUncertainty:
    def __init__(self):
        self.sources = {
            'text_similarity': 0.0,
            'graph_structure': 0.0,
            'user_feedback': 0.0,
            'model_confidence': 0.0
        }

    def combine_uncertainties(self, uncertainties, weights=None):
        """Combine uncertainties from different sources."""
        if weights is None:
            weights = {k: 1.0 for k in uncertainties.keys()}

        # Normalize weights
        total_weight = sum(weights.values())
        normalized_weights = {k: v/total_weight for k, v in weights.items()}

        # Combine using reliability-weighted average
        combined_mean = 0
        combined_variance = 0

        for source, mean in uncertainties.items():
            weight = normalized_weights[source]
            variance = self.estimate_source_variance(source)

            combined_mean += weight * mean
            combined_variance += (weight ** 2) * variance

        return {
            'mean': combined_mean,
            'variance': combined_variance,
            'std': np.sqrt(combined_variance)
        }

    def estimate_source_variance(self, source):
        """Estimate variance of each uncertainty source."""
        # These could be learned from validation data
        variances = {
            'text_similarity': 0.1,
            'graph_structure': 0.15,
            'user_feedback': 0.05,  # User feedback is more reliable
            'model_confidence': 0.2
        }
        return variances.get(source, 0.2)
```

### 7.2 Relationship Type-Specific Uncertainty

**Different relationship types have different uncertainty characteristics**:
```python
class RelationSpecificUncertainty:
    def __init__(self):
        # Uncertainty models for each relation type
        self.models = {
            'SUPPORTS': BetaUncertainty(successes=1, failures=1),
            'CONTRADICTS': BetaUncertainty(successes=1, failures=1),
            'RELATES_TO': BetaUncertainty(successes=1, failures=1),
            'CITES': BetaUncertainty(successes=1, failures=1)
        }

        # Type-specific features
        self.type_features = {
            'SUPPORTS': {'requires_evidence': True, 'symmetric': False},
            'CONTRADICTS': {'requires_evidence': True, 'symmetric': False},
            'RELATES_TO': {'requires_evidence': False, 'symmetric': True},
            'CITES': {'requires_evidence': False, 'symmetric': False}
        }

    def predict(self, relation_type, features):
        """Predict uncertainty for specific relation type."""
        model = self.models[relation_type]
        features = self.type_features[relation_type]

        # Base uncertainty from historical performance
        base_confidence = model.mean()
        base_uncertainty = model.variance()

        # Adjust based on features
        if features['requires_evidence'] and not features.get('has_evidence', True):
            base_uncertainty *= 2.0  # Higher uncertainty without evidence

        if features['symmetric'] and features.get('is_asymmetric', False):
            base_uncertainty *= 1.5  # Higher uncertainty for wrong symmetry

        return {
            'confidence': base_confidence,
            'uncertainty': base_uncertainty,
            'credible_interval': model.credible_interval()
        }

    def update(self, relation_type, correct):
        """Update model with new observation."""
        self.models[relation_type].update({'correct': correct})
```

---

## 8. Decision Thresholds with Uncertainty

### 8.1 Risk-Adjusted Decision Making

**Make decisions considering both confidence and uncertainty**:
```python
class RiskAdjustedDecision:
    def __init__(self, risk_preferences=None):
        self.risk_preferences = risk_preferences or {
            'accept_threshold': 0.8,  # Minimum confidence to accept
            'uncertainty_penalty': 0.3,  # Penalty for high uncertainty
            'risk_aversion': 0.5,  # Higher = more risk averse
        }

    def should_accept_prediction(self, confidence, uncertainty):
        """Should we accept this prediction or require human review."""
        # Adjust confidence by uncertainty
        adjusted_confidence = confidence * (1 - uncertainty)

        # Apply risk aversion
        risk_adjusted = adjusted_confidence - self.risk_preferences['risk_aversion'] * uncertainty

        # Decision
        should_accept = risk_adjusted >= self.risk_preferences['accept_threshold']

        return {
            'accept': should_accept,
            'adjusted_confidence': adjusted_confidence,
            'risk_adjusted': risk_adjusted,
            'requires_review': not should_accept
        }

    def optimal_decision(self, options):
        """Choose best option considering risk."""
        scored_options = []

        for option in options:
            score = self.risk_adjusted_score(
                option['confidence'],
                option['uncertainty'],
                option.get('cost', 0)
            )
            scored_options.append((option, score))

        return max(scored_options, key=lambda x: x[1])

    def risk_adjusted_score(self, confidence, uncertainty, cost):
        """Calculate risk-adjusted score."""
        expected_value = confidence - cost
        risk_penalty = uncertainty * self.risk_preferences['risk_aversion']
        return expected_value - risk_penalty
```

### 8.2 Expected Utility Framework

**Maximize expected utility with uncertainty**:
```python
class ExpectedUtility:
    def __init__(self, utility_functions):
        self.utility_functions = utility_functions

    def expected_utility(self, action, uncertainty_dist):
        """Calculate expected utility for an action."""
        utility = 0

        for outcome, prob in uncertainty_dist.items():
            u = self.utility_functions[action](outcome)
            utility += prob * u

        return utility

    def choose_action(self, actions, uncertainty_dists):
        """Choose action with highest expected utility."""
        expected_utilities = {}

        for action in actions:
            eu = self.expected_utility(action, uncertainty_dists[action])
            expected_utilities[action] = eu

        return max(expected_utilities.items(), key=lambda x: x[1])
```

---

## 9. Implementation Guide for ThoughtLab

### 9.1 Progressive Implementation

#### Phase 1: Basic Uncertainty Tracking (Week 1-2)
**Goal**: Replace simple confidence with calibrated confidence

```python
# Current: Simple 0-1 confidence
# Enhanced: Beta distribution tracking

class RelationshipConfidence:
    def __init__(self):
        self.distributions = {}  # rel_type -> BetaUncertainty

    def get_confidence(self, rel_type):
        if rel_type not in self.distributions:
            # Initialize with prior (Laplace smoothing)
            self.distributions[rel_type] = BetaUncertainty(successes=1, failures=1)

        dist = self.distributions[rel_type]
        return {
            'mean': dist.mean(),
            'variance': dist.variance(),
            'credible_interval': dist.credible_interval(),
            'prob_gt_threshold': dist.probability_gt(threshold=0.7)
        }

    def update(self, rel_type, correct):
        if rel_type not in self.distributions:
            self.distributions[rel_type] = BetaUncertainty(successes=1, failures=1)

        self.distributions[rel_type].update({'correct': correct})
```

**Integration**: Update when users accept/reject suggestions

#### Phase 2: Uncertainty-Aware Recommendations (Month 1)
**Goal**: Only suggest relationships with high confidence + low uncertainty

```python
def get_relationship_suggestions(node_id, threshold=0.7):
    """Get relationship suggestions with uncertainty thresholds."""
    candidates = find_similar_nodes(node_id)

    suggestions = []
    for candidate in candidates:
        # Predict relationship type with uncertainty
        pred = predictor.predict_with_uncertainty(node_id, candidate['id'])

        # Check if prediction meets criteria
        if (pred['confidence'] >= threshold and
            pred['uncertainty'] <= 0.2 and
            pred['prob_gt_threshold'] >= 0.8):

            suggestions.append({
                'node': candidate,
                'relationship': pred['type'],
                'confidence': pred['confidence'],
                'uncertainty': pred['uncertainty'],
                'recommendation_strength': pred['prob_gt_threshold']
            })

    return sorted(suggestions, key=lambda x: x['recommendation_strength'], reverse=True)
```

#### Phase 3: Bayesian Methods (Month 2-3)
**Goal**: Implement true Bayesian inference for better uncertainty estimates

```python
# Use Monte Carlo Dropout for relationship classification
class BayesianRelationshipClassifier:
    def __init__(self, base_model):
        self.model = base_model
        self.n_samples = 50

    def predict(self, node_a, node_b):
        """Predict with uncertainty via MC sampling."""
        features = self.extract_features(node_a, node_b)

        predictions = []
        for _ in range(self.n_samples):
            with torch.no_grad():
                pred = self.model(features, train=False)  # Keep dropout on
                predictions.append(pred)

        predictions = torch.stack(predictions)
        mean = predictions.mean(dim=0)
        variance = predictions.var(dim=0)

        return {
            'mean': mean,
            'variance': variance,
            'uncertainty': variance,
            'confidence': torch.softmax(mean, dim=-1).max().item()
        }
```

#### Phase 4: Production System (Month 4-6)
**Goal**: Full uncertainty quantification with decision making

```python
class UncertaintyAwareKG:
    def __init__(self):
        self.uncertainty_tracker = UncertaintyTracker()
        self.decision_maker = RiskAdjustedDecision()
        self.calibrator = TemperatureScaling()

    def add_relationship(self, head, relation, tail, evidence):
        """Add relationship with uncertainty quantification."""
        # Get predictions with uncertainty
        pred = self.predict_with_uncertainty(head, relation, tail, evidence)

        # Calibrate
        calibrated = self.calibrator.predict(pred['raw_scores'])

        # Make decision
        decision = self.decision_maker.should_accept_prediction(
            calibrated['confidence'],
            calibrated['uncertainty']
        )

        if decision['accept']:
            # Add to KG
            self.kg.create_relationship(head, relation, tail, {
                'confidence': calibrated['confidence'],
                'uncertainty': calibrated['uncertainty'],
                'evidence': evidence,
                'source': 'automated',
                'calibrated': True
            })
        else:
            # Request human review
            self.request_review(head, relation, tail, evidence, pred)

        return decision
```

---

### 9.2 API Design

**Uncertainty-aware API endpoints**:
```python
# POST /api/v1/graph/relationships/with-uncertainty
# {
#   "head": "entity_id",
#   "tail": "entity_id",
#   "evidence": "text or context"
# }
#
# Response:
# {
#   "predictions": [
#     {
#       "type": "SUPPORTS",
#       "confidence": 0.85,
#       "uncertainty": 0.08,
#       "credible_interval": [0.78, 0.91],
#       "prob_gt_threshold": 0.92,
#       "requires_review": false
#     }
#   ],
#   "overall_confidence": 0.85,
#   "overall_uncertainty": 0.08,
#   "recommended_action": "accept"
# }

# POST /api/v1/graph/validate-uncertainty
# {
#   "relationship_id": "rel_123",
#   "human_feedback": "correct"  # or "incorrect"
# }
#
# Response:
# {
#   "old_confidence": 0.85,
#   "new_confidence": 0.88,
#   "old_uncertainty": 0.08,
#   "new_uncertainty": 0.07,
#   "update_effect": "improved"
# }
```

---

### 9.3 Monitoring Dashboard

**Track uncertainty metrics over time**:
```python
class UncertaintyMonitor:
    def __init__(self):
        self.metrics_history = []

    def track_metrics(self, metrics):
        """Store uncertainty metrics for monitoring."""
        self.metrics_history.append({
            'timestamp': datetime.now(),
            'calibration_error': metrics['ece'],
            'avg_confidence': metrics['avg_confidence'],
            'avg_accuracy': metrics['avg_accuracy'],
            'uncertainty_distribution': metrics['uncertainty_dist'],
            'decision_rate': metrics['accept_rate']
        })

    def plot_calibration(self):
        """Plot calibration curve."""
        import matplotlib.pyplot as plt

        confidences = []
        accuracies = []

        for entry in self.metrics_history:
            confidences.append(entry['avg_confidence'])
            accuracies.append(entry['avg_accuracy'])

        plt.figure(figsize=(10, 6))
        plt.plot(confidences, accuracies, 'bo-', label='Model')
        plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
        plt.xlabel('Predicted Confidence')
        plt.ylabel('Actual Accuracy')
        plt.title('Calibration Curve')
        plt.legend()
        plt.grid(True)
        plt.show()

    def monitor_drift(self, window=100):
        """Monitor for concept drift in uncertainty."""
        recent = self.metrics_history[-window:]
        if len(recent) < window:
            return None

        # Check if calibration is degrading
        recent_ece = [m['calibration_error'] for m in recent]
        mean_ece = np.mean(recent_ece)
        std_ece = np.std(recent_ece)

        # Alert if ECE is increasing
        if mean_ece > 0.1:  # Threshold
            return {
                'alert': 'poor_calibration',
                'mean_ece': mean_ece,
                'threshold': 0.1
            }

        return None
```

---

## 10. Evaluation of Uncertainty Quality

### 10.1 Calibration Metrics

**Expected Calibration Error (ECE)**:
```python
def expected_calibration_error(predictions, labels, n_bins=10):
    """Calculate ECE."""
    confidences, preds = predictions.max(dim=1)
    accuracies = (preds == labels).float()

    # Bin by confidence
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.float().mean()

        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += (avg_confidence_in_bin - accuracy_in_bin).abs() * prop_in_bin

    return ece.item()
```

**Maximum Calibration Error (MCE)**:
```python
def maximum_calibration_error(predictions, labels, n_bins=10):
    """Calculate MCE (worst-case calibration error)."""
    confidences, preds = predictions.max(dim=1)
    accuracies = (preds == labels).float()

    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    errors = []
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.float().mean()

        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            errors.append((avg_confidence_in_bin - accuracy_in_bin).abs())

    return max(errors).item() if errors else 0
```

### 10.2 Uncertainty Quality Metrics

**Brier Score** (lower is better):
```python
def brier_score(probs, labels):
    """Calculate Brier score (calibration + refinement)."""
    # probs: [n_samples, n_classes]
    # labels: [n_samples]
    n = len(labels)
    one_hot = F.one_hot(labels, num_classes=probs.shape[1]).float()
    return torch.mean(torch.sum((probs - one_hot) ** 2, dim=1)).item()
```

**Negative Log Likelihood**:
```python
def negative_log_likelihood(probs, labels):
    """Calculate NLL (lower is better)."""
    nll = F.nll_loss(torch.log(probs + 1e-10), labels)
    return nll.item()
```

**Sharpness** (measure of how confident predictions are):
```python
def sharpness(probs):
    """Calculate sharpness (confidence of predictions)."""
    # Higher = more confident (but shouldn't be overconfident)
    max_probs, _ = probs.max(dim=1)
    return max_probs.mean().item()
```

### 10.3 Uncertainty-Adjusted Accuracy

**Calculate accuracy with uncertainty threshold**:
```python
def uncertainty_adjusted_accuracy(probs, uncertainties, labels, threshold=0.1):
    """Calculate accuracy for predictions with uncertainty < threshold."""
    mask = uncertainties < threshold
    if mask.sum() == 0:
        return 0.0

    preds = probs[mask].argmax(dim=1)
    acc = (preds == labels[mask]).float().mean().item()
    return acc
```

---

## 11. Key Academic Sources

### 11.1 Foundational Papers
1. **"Bayesian Reasoning and Machine Learning"** (Barber, 2012) - Comprehensive textbook
2. **"Probabilistic Machine Learning"** (Murphy, 2020) - Modern methods
3. **"Uncertainty in Artificial Intelligence"** (UAI conference proceedings) - Annual survey

### 11.2 Neural Uncertainty Methods
1. **"Dropout as a Bayesian Approximation"** (Gal & Ghahramani, 2016) - MC Dropout
2. **"Weight Uncertainty in Neural Networks"** (Blundell et al., 2015) - Bayesian NN
3. **"Simple and Scalable Predictive Uncertainty Estimation"** (2017) - Deep Ensembles

### 11.3 Knowledge Graph Uncertainty
1. **"Uncertain Knowledge Graph Embedding"** (2019) - Bayesian TransE
2. **"Probabilistic Soft Logic"** (Koller et al., 2009) - Logic with uncertainty
3. **"Box Embeddings for Uncertain KGs"** (2021) - Geometric uncertainty

### 11.4 Calibration
1. **"On Calibration of Modern Neural Networks"** (Guo et al., 2017) - Temperature scaling
2. **"Beyond Temperature: Scaling for Calibration"** (2020) - Improved methods
3. **"Calibration in ML: Survey"** (2021) - Comprehensive overview

---

## 12. Implementation Priority Matrix

| Priority | Method | Impact | Effort | Timeline |
|----------|--------|--------|--------|----------|
| **P0** | Beta distribution tracking | 🔴 High | 🟢 Low | Week 1 |
| **P0** | Confidence calibration | 🔴 High | 🟢 Low | Week 1-2 |
| **P1** | Monte Carlo Dropout | 🟡 Medium | 🟡 Medium | Month 1 |
| **P1** | Uncertainty thresholds | 🟡 Medium | 🟡 Medium | Month 1-2 |
| **P2** | Bayesian Neural Networks | 🟡 Medium | 🔴 High | Month 2-3 |
| **P2** | Risk-adjusted decisions | 🟡 Medium | 🟡 Medium | Month 3 |
| **P3** | Full Bayesian inference | 🔴 High | 🔴 High | Month 4-6 |

---

## 13. Quick Start Guide

### Week 1: Basic Uncertainty Tracking

```python
# Step 1: Install scipy for beta distribution
pip install scipy

# Step 2: Implement Beta uncertainty
class SimpleUncertainty:
    def __init__(self):
        self.alpha = 1  # successes
        self.beta = 1   # failures

    def update(self, correct):
        if correct:
            self.alpha += 1
        else:
            self.beta += 1

    def confidence(self):
        return self.alpha / (self.alpha + self.beta)

    def uncertainty(self):
        # Variance as uncertainty measure
        return (self.alpha * self.beta) / \
               ((self.alpha + self.beta) ** 2 * (self.alpha + self.beta + 1))

# Step 3: Track for each relationship type
uncertainties = {
    'SUPPORTS': SimpleUncertainty(),
    'CONTRADICTS': SimpleUncertainty(),
    'RELATES_TO': SimpleUncertainty()
}

# Step 4: Update with user feedback
def update_from_feedback(relationship_type, user_accepted):
    uncertainties[relationship_type].update(user_accepted)

# Step 5: Use in recommendations
def should_recommend(relationship_type, min_confidence=0.7, max_uncertainty=0.1):
    conf = uncertainties[relationship_type].confidence()
    unc = uncertainties[relationship_type].uncertainty()
    return conf >= min_confidence and unc <= max_uncertainty
```

**Expected Outcomes**:
- 20-30% improvement in recommendation quality
- Better understanding of model reliability
- Foundation for more advanced methods

---

## 14. Expected Impact

### 14.1 For System Quality
- **Calibration**: Confidence scores match actual accuracy
- **Trustworthiness**: Users can trust/reject suggestions based on uncertainty
- **Efficiency**: Only request human review for uncertain cases
- **Learning**: Continuous improvement from feedback

### 14.2 For User Experience
- **Transparency**: Users see not just confidence, but uncertainty bounds
- **Control**: Users can set risk tolerance thresholds
- **Explanations**: Uncertainty breakdown by source
- **Decision Support**: System explains when to trust vs. verify

### 14.3 For Research Quality
- **Better Decisions**: Uncertainty-aware recommendations
- **Bias Detection**: Identify systematic over/under-confidence
- **Model Improvement**: Target uncertain cases for retraining
- **Risk Management**: Quantify and manage decision risks

---

## 15. Summary of Recommendations

### 15.1 Immediate Actions (Week 1-2)
1. **Replace point confidence** with Beta distributions
2. **Track accuracy per relationship type**
3. **Implement confidence calibration**
4. **Add uncertainty to API responses**

### 15.2 Short-term (Month 1-2)
1. **Add Monte Carlo uncertainty** for predictions
2. **Implement uncertainty thresholds** for recommendations
3. **Build user feedback loop** for calibration
4. **Monitor calibration metrics**

### 15.3 Medium-term (Month 3-4)
1. **Bayesian Neural Networks** for improved uncertainty
2. **Risk-adjusted decision making** for automated actions
3. **Multi-source uncertainty** combination
4. **Uncertainty visualization** in UI

### 15.4 Long-term (Month 5-6)
1. **Full Bayesian inference** pipeline
2. **Causal uncertainty** for relationship analysis
3. **Active learning** with uncertainty sampling
4. **Research publications** on novel methods

---

## Conclusion

**Key Takeaway**: Moving from point estimates to probability distributions enables better decision making, transparency, and continuous learning in knowledge graphs.

**Recommended Path**:
1. **Start with Beta distributions** (Week 1) - simple, effective
2. **Add calibration** (Week 2) - fix miscalibration
3. **Implement Monte Carlo methods** (Month 1) - better uncertainty
4. **Build decision system** (Month 2+) - risk-aware automation

**Success Metrics**:
- ECE < 0.05 (well-calibrated)
- 30% reduction in human review requirements
- User trust score > 4.5/5
- Model improvement from feedback

**Risk Level**: **Low** (incremental improvements, reversible)

**Time to Impact**: **1-2 weeks** for basic uncertainty tracking

---

**Previous Document**: [Graph Construction & Entity Linking](./graph_construction_entity_linking.md)