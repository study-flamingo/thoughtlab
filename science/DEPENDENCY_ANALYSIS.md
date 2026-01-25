# ThoughtLab Dependency Analysis & Gap Assessment

**Date**: 2026-01-25
**Purpose**: Identify existing dependencies vs required tools for scientific algorithms

---

## 📊 Current Dependency Inventory

### Backend (Python)
```
✅ INSTALLED:
├─ Neo4j Driver: 6.0.3 ✓ (Recent, excellent)
├─ NumPy: 2.3.5 ✓ (Essential for ML)
├─ SciPy: 1.16.3 ✓ (Statistical functions)
├─ FastAPI: 0.121.0+ ✓ (Web framework)
├─ Uvicorn: 0.32.0+ ✓ (ASGI server)
├─ Redis: 5.0.0+ ✓ (Caching)
├─ Pydantic: 2.6.0+ ✓ (Validation)
├─ httpx: 0.26.0+ ✓ (HTTP client)
├─ pytest: 8.0.0+ ✓ (Testing)
└─ python-dotenv: 1.0.0+ ✓ (Config)

❌ NOT INSTALLED:
├─ PyTorch: ❌ (Required for ML)
├─ LangChain: ❌ (AI framework)
├─ Transformers: ❌ (BERT models)
├─ PyTorch Geometric: ❌ (GNNs)
├─ scikit-learn: ❌ (ML utilities)
├─ spacy: ❌ (NLP/NER)
├─ scipy: ✅ (Already installed!)
├─ scikit-learn: ❌ (ML utilities)
├─ wandb: ❌ (Experiment tracking)
├─ optuna: ❌ (Hyperparameter optimization)
└─ pyro-ppl: ❌ (Bayesian inference)
```

### Frontend (Node.js)
```
✅ INSTALLED:
├─ React: 18.3.1 ✓ (UI framework)
├─ TypeScript: 5.7.2 ✓ (Type safety)
├─ Vite: 5.4.21 ✓ (Build tool)
├─ Cytoscape: 3.31.0 ✓ (Graph visualization)
├─ TanStack Query: 5.62.11 ✓ (Server state)
├─ Axios: 1.7.9 ✓ (HTTP client)
├─ Tailwind CSS: 3.4.17 ✓ (Styling)
└─ Vitest: 2.1.8 ✓ (Testing)

❌ NOT INSTALLED (for ML viz):
├─ TensorFlow.js: ❌ (Browser ML - optional)
├─ D3.js: ❌ (Advanced visualization - optional)
└─ Plotly.js: ❌ (Charts - optional)
```

---

## 🎯 Gap Analysis by Phase

### Phase 1: Foundation (Week 1-2)
**Required for HNSW indexing**:
- ✅ Neo4j 5.13+ (Have 6.0.3) - **READY**
- ✅ Python 3.11+ (Have 3.13) - **READY**
- ✅ NumPy (Have 2.3.5) - **READY**

**Required for Beta uncertainty**:
- ✅ Python math - **READY**
- ✅ SciPy (Have 1.16.3) - **READY** (for Beta distribution)
- ✅ NumPy (Have 2.3.5) - **READY**

**Required for Neo4j GDS**:
- ✅ Neo4j 5.13+ Enterprise (Have 6.0.3) - **READY**
- ⚠️ GDS Plugin - Need to check
- ✅ Neo4j driver (Have 6.0.3) - **READY**

**Required for calibration**:
- ✅ NumPy (Have 2.3.5) - **READY**
- ✅ SciPy (Have 1.16.3) - **READY**
- ⚠️ scikit-learn (for temperature scaling) - **MISSING**

**Phase 1 Status**: ✅ **90% READY**
- Only missing: scikit-learn (easy install)
- GDS plugin check needed

---

### Phase 2: Intelligence (Month 1)
**Required for BERT fine-tuning**:
- ❌ PyTorch - **MISSING** (Major)
- ❌ Transformers (HuggingFace) - **MISSING** (Major)
- ❌ scikit-learn - **MISSING** (Easy)
- ✅ NumPy - **READY**
- ⚠️ GPU recommended (RTX 3090 or similar) - **CHECK**

**Required for ComplEx embeddings**:
- ❌ PyTorch - **MISSING** (Major)
- ❌ scikit-learn - **MISSING** (Easy)
- ✅ NumPy - **READY**
- ✅ SciPy - **READY**

**Required for A/B testing**:
- ⚠️ wandb (optional) or MLflow - **MISSING** (Easy)
- ✅ Existing: Manual tracking possible

**Phase 2 Status**: ⚠️ **40% READY**
- Missing: PyTorch, Transformers (core ML libraries)
- Need: GPU for training
- Easy wins: scikit-learn, wandb

---

### Phase 3: Advanced (Month 2-3)
**Required for RGCN**:
- ❌ PyTorch - **MISSING**
- ❌ PyTorch Geometric - **MISSING**
- ❌ scikit-learn - **MISSING**
- ✅ NumPy - **READY**
- ⚠️ GPU recommended - **CHECK**

**Required for Active Learning**:
- ❌ PyTorch - **MISSING**
- ❌ scikit-learn - **MISSING**
- ⚠️ wandb/optuna - **MISSING**

**Required for Bayesian methods**:
- ❌ Pyro/Pyro-ppl (Bayesian) - **MISSING**
- ❌ PyTorch - **MISSING**
- ✅ SciPy - **READY**

**Phase 3 Status**: 🚧 **30% READY**
- Missing: PyTorch, PyTorch Geometric, Pyro
- Need: GPU resources
- Complex: Bayesian inference setup

---

### Phase 4: Production (Month 4-6)
**Required for scale**:
- ⚠️ Distributed training - **MISSING** (infrastructure)
- ⚠️ MLOps tools - **MISSING** (wandb, MLflow)
- ✅ Existing: Basic infrastructure in place

**Required for research**:
- ❌ Advanced libraries - **MISSING**
- ⚠️ Publication tools - **MISSING**
- ✅ Existing: Strong foundation

**Phase 4 Status**: 🚧 **50% READY**
- Need: Infrastructure scaling
- Need: MLOps pipeline
- Need: Research tools

---

## 📋 Installation Priority Matrix

### P0: Phase 1 (Week 1) - Critical
| Package | Why | Install Command | Priority |
|---------|-----|-----------------|----------|
| **scikit-learn** | Calibration, ML utilities | `uv add scikit-learn` | 🔴 Critical |
| **pytorch** | Foundation for all ML | `uv add torch --extra-index-url` | 🔴 Critical |
| **transformers** | BERT models for NER/RE | `uv add transformers` | 🔴 Critical |

### P1: Phase 2 (Month 1) - Important
| Package | Why | Install Command | Priority |
|---------|-----|-----------------|----------|
| **torch-geometric** | GNNs (RGCN) | `uv add torch-geometric` | 🟡 High |
| **wandb** | Experiment tracking | `uv add wandb` | 🟡 Medium |
| **spacy** | Alternative NER | `uv add spacy` | 🟢 Low |

### P2: Phase 3 (Month 2-3) - Advanced
| Package | Why | Install Command | Priority |
|---------|-----|-----------------|----------|
| **pyro-ppl** | Bayesian inference | `uv add pyro-ppl` | 🟡 Medium |
| **optuna** | Hyperparameter opt | `uv add optuna` | 🟡 Medium |
| **dgl** | Alternative GNN lib | `uv add dgl` | 🟢 Low |

### P3: Phase 4 (Month 4-6) - Optimization
| Package | Why | Install Command | Priority |
|---------|-----|-----------------|----------|
| **mlflow** | MLOps pipeline | `uv add mlflow` | 🟡 Medium |
| **ray** | Distributed training | `uv add ray` | 🟢 Low |
| **knockknock** | Training alerts | `uv add knockknock` | 🟢 Low |

---

## 🔧 Installation Commands

### For Phase 1 (Week 1)
```bash
# Activate virtual environment
cd backend
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate  # Windows

# Install critical packages (5 minutes)
uv add scikit-learn
uv add pytorch --extra-index-url https://download.pytorch.org/whl/cpu  # CPU version
uv add transformers

# Verify installations
python -c "import sklearn; print('scikit-learn:', sklearn.__version__)"
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "from transformers import pipeline; print('Transformers: OK')"
```

### For Phase 2 (Month 1)
```bash
# Additional packages
uv add torch-geometric
uv add wandb
uv add optuna

# Verify
python -c "import torch_geometric; print('PyG:', torch_geometric.__version__)"
```

### For Phase 3 (Month 2-3)
```bash
# Advanced packages
uv add pyro-ppl
uv add dgl -f https://data.dgl.ai/wheels/torch2.2/cpu/repo.html

# Verify
python -c "import pyro; print('Pyro: OK')"
```

---

## 💻 Hardware Requirements

### Current Setup Check
```bash
# Check CPU
python -c "import multiprocessing; print(f'CPU Cores: {multiprocessing.cpu_count()}')"

# Check RAM
import psutil
print(f"RAM: {psutil.virtual_memory().total / 1e9:.1f} GB")
```

### Phase 1 Requirements
- **CPU**: 4+ cores (likely OK)
- **RAM**: 8+ GB (likely OK)
- **GPU**: Not required
- **Storage**: 50 GB (likely OK)

### Phase 2 Requirements
- **CPU**: 8+ cores (check)
- **RAM**: 16+ GB (check)
- **GPU**: Recommended (RTX 3090 or similar)
- **Storage**: 100 GB (check)

### Phase 3 Requirements
- **CPU**: 16+ cores (may need upgrade)
- **RAM**: 32+ GB (may need upgrade)
- **GPU**: Required (2× RTX 3090 or A100)
- **Storage**: 200 GB (check)

### Phase 4 Requirements
- **CPU**: 32+ cores (cloud recommended)
- **RAM**: 64+ GB (cloud recommended)
- **GPU**: 4× A100 (cloud recommended)
- **Storage**: 500 GB+ (cloud recommended)

---

## 🎯 Quick Wins (Minimal Install)

### Option 1: Minimal Phase 1 (2 hours)
```bash
# Just what's needed for Week 1
uv add scikit-learn
uv add pytorch --extra-index-url https://download.pytorch.org/whl/cpu

# Total: 2 packages, ~2GB download
```
**Impact**: 90% of Phase 1 value

### Option 2: Phase 1 + 2 Basics (4 hours)
```bash
# Phase 1 + BERT fine-tuning basics
uv add scikit-learn
uv add pytorch --extra-index-url https://download.pytorch.org/whl/cpu
uv add transformers
uv add torch-geometric

# Total: 4 packages, ~5GB download
```
**Impact**: 90% of Phase 1 + 80% of Phase 2 value

### Option 3: Full Phase 1-2 (6 hours)
```bash
# All Phase 1-2 requirements
uv add scikit-learn
uv add pytorch --extra-index-url https://download.pytorch.org/whl/cpu
uv add transformers
uv add torch-geometric
uv add wandb
uv add optuna
uv add pyro-ppl

# Total: 7 packages, ~6GB download
```
**Impact**: 100% of Phase 1-2 value + Phase 3 basics

---

## 🔍 Neo4j GDS Check

### Check if GDS Plugin is Available
```bash
# Run in Neo4j Browser
RETURN gds.version();

# Or via Cypher shell
cypher-shell "RETURN gds.version();"
```

### If Not Available, Enable It
```bash
# Update docker-compose.yml
services:
  neo4j:
    image: neo4j:5.13.0-enterprise  # or 6.0.0-enterprise
    environment:
      - NEO4J_ACCEPT_LICENSE_AGREEMENT=yes
      - NEO4J_PLUGINS=["graph-data-science"]  # Add this line
```

### Verify Installation
```bash
# After restart
cypher-shell "RETURN gds.version();"

# Should return: 2.x.x
```

---

## 🚀 Recommended Installation Order

### Today (30 minutes)
1. ✅ Check Neo4j version (done: 6.0.3)
2. ✅ Check Neo4j GDS plugin
3. ⚠️ Install scikit-learn
4. ⚠️ Install PyTorch (CPU version first)

### This Week (4 hours)
1. ⚠️ Install transformers
2. ⚠️ Install torch-geometric
3. ⚠️ Test with Phase 1 algorithms
4. ⚠️ Set up experiment tracking

### Month 1 (1-2 days)
1. ⚠️ Install full ML stack
2. ⚠️ Test BERT fine-tuning
3. ⚠️ Set up GPU environment
4. ⚠️ Run first ML experiments

---

## 💰 Cost Estimates

### Software Costs (All Free/Open Source)
- **PyTorch**: Free
- **Transformers**: Free
- **scikit-learn**: Free
- **PyTorch Geometric**: Free
- **wandb**: Free tier available
- **MLflow**: Free/Open Source

### Hardware Costs (If Upgrading)
```
Local Development:
- GPU (RTX 3090): $1,500-2,000 (one-time)
- RAM upgrade (16→32 GB): $100-200
- Total: $1,600-2,200

Cloud (Alternative):
- AWS g4dn.xlarge (T4 GPU): $0.526/hr
- Google Colab Pro: $9.99/month
- RunPod.io: $0.20-0.80/hr
```

### Team Time Costs
```
Installation & Setup:
- Phase 1: 4 hours × $100/hr = $400
- Phase 2: 16 hours × $100/hr = $1,600
- Phase 3: 40 hours × $100/hr = $4,000
- Total Setup: ~$6,000

Ongoing (6 months):
- 20% FTE ML Engineer: ~$60,000
- Infrastructure: $2,000-5,000
```

---

## 🎓 Learning Requirements

### Team Skills Assessment
**Current** (likely):
- ✅ Python programming
- ✅ FastAPI/REST APIs
- ✅ Neo4j/Cypher
- ✅ Basic statistics
- ✅ Git/version control

**Required for ML** (to learn):
- ⚠️ PyTorch/TensorFlow
- ⚠️ Deep learning fundamentals
- ⚠️ Graph neural networks
- ⚠️ Bayesian statistics
- ⚠️ MLOps practices

### Learning Timeline
```
Week 1-2: PyTorch basics (8 hours)
Month 1: GNNs + Transformers (20 hours)
Month 2: Bayesian methods (16 hours)
Month 3: MLOps + Advanced topics (20 hours)
Total: ~64 hours per team member
```

---

## 📊 Summary: Current vs Required

### What We Have ✅
```
Core Infrastructure:
✓ Neo4j 6.0.3 (excellent)
✓ FastAPI/Backend
✓ React/Frontend
✓ Redis cache
✓ Basic ML: NumPy, SciPy

Mathematical Foundations:
✓ Linear algebra (NumPy)
✓ Statistics (SciPy)
✓ Probability (SciPy)

Development Tools:
✓ Testing (pytest)
✓ Type checking (Pydantic)
✓ Build tools (uv)
```

### What We Need ⚠️
```
Machine Learning:
✗ PyTorch (core requirement)
✗ Transformers (BERT models)
✗ scikit-learn (utilities)
✗ PyTorch Geometric (GNNs)

Advanced Methods:
✗ Bayesian inference (Pyro)
✗ Hyperparameter optimization (optuna)
✗ Experiment tracking (wandb/MLflow)

Hardware:
? GPU for training (RTX 3090 or cloud)
? More RAM (16-32 GB recommended)
? Storage (50-200 GB depending on phase)

Team Knowledge:
? Deep learning fundamentals
? Graph neural networks
? Bayesian statistics
? MLOps practices
```

### Installation Difficulty
- **Easy** (5 minutes): scikit-learn, wandb
- **Medium** (30 minutes): PyTorch, transformers
- **Hard** (1-2 hours): PyTorch Geometric, CUDA setup
- **Complex** (1 day): Full MLOps pipeline

---

## 🎯 Immediate Action Plan

### Today (1 hour)
1. [ ] Check Neo4j GDS plugin availability
2. [ ] Install scikit-learn: `uv add scikit-learn`
3. [ ] Install PyTorch CPU: `uv add torch --extra-index-url https://download.pytorch.org/whl/cpu`
4. [ ] Test installations work
5. [ ] Plan GPU acquisition if needed

### This Week (4 hours)
1. [ ] Install transformers: `uv add transformers`
2. [ ] Test with a simple BERT model
3. [ ] Set up wandb account for tracking
4. [ ] Install Phase 2 dependencies
5. [ ] Run Phase 1 algorithms to test

### Month 1 (1-2 days)
1. [ ] Assess GPU needs based on testing
2. [ ] Purchase GPU or set up cloud account
3. [ ] Install PyTorch Geometric
4. [ ] Run BERT fine-tuning test
5. [ ] Set up experiment tracking
6. [ ] Plan Phase 3 dependencies

---

## 📞 Getting Help

### Installation Issues
- **PyTorch**: Check https://pytorch.org/get-started/locally/
- **PyTorch Geometric**: https://pytorch-geometric.readthedocs.io/en/latest/install/
- **Transformers**: https://huggingface.co/docs/transformers/installation
- **Neo4j GDS**: https://neo4j.com/docs/graph-data-science/current/installation/

### Team Support
- Schedule pair programming for installation
- Use team learning sessions
- Share installation scripts
- Document common issues

### External Resources
- PyTorch tutorials: https://pytorch.org/tutorials/
- HuggingFace course: https://huggingface.co/learn
- Neo4j GDS docs: https://neo4j.com/docs/graph-data-science/
- Papers from research documents

---

## ✅ Success Criteria

### Installation Complete When:
1. ✅ scikit-learn imports without error
2. ✅ PyTorch basic operations work
3. ✅ Transformers can load a model
4. ✅ Neo4j GDS algorithms run
5. ✅ Phase 1 code executes successfully

### Phase 1 Ready When:
1. ✅ HNSW indexing deployed
2. ✅ Beta uncertainty tracking works
3. ✅ Neo4j GDS provides analytics
4. ✅ Calibration improves confidence scores

### Phase 2 Ready When:
1. ✅ BERT can be fine-tuned on sample data
2. ✅ ComplEx embeddings train successfully
3. ✅ User feedback loop integrated
4. ✅ A/B testing framework operational

---

**Next Step**: Check Neo4j GDS plugin availability, then install scikit-learn and PyTorch

**Estimated Time to Phase 1**: 1-2 hours (install + test)
**Estimated Time to Phase 2**: 1-2 days (install + basic tests)
**Total Time to Full Stack**: 1 week (install + validation)

---
**Document Version**: 1.0
**Last Updated**: 2026-01-25
**Status**: Gap Analysis Complete
**Next**: Installation & Testing