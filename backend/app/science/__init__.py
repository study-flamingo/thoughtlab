"""
ThoughtLab Scientific Foundation Module

Implements research-backed algorithms for:
- Relationship confidence scoring
- Semantic similarity search
- Graph traversal and analytics
- Uncertainty quantification
- Knowledge graph embeddings
"""

from .uncertainty import BetaUncertainty, UncertaintyTracker
from .similarity import HNSWSimilarity, MultiMetricSimilarity
from .calibration import TemperatureScaling, ConfidenceCalibrator
from .graph_algorithms import GraphAnalytics

__all__ = [
    "BetaUncertainty",
    "UncertaintyTracker",
    "HNSWSimilarity",
    "MultiMetricSimilarity",
    "TemperatureScaling",
    "ConfidenceCalibrator",
    "GraphAnalytics",
]