"""
Uncertainty Quantification Module

Implements Bayesian uncertainty tracking using Beta distributions for relationship
confidence scores. Based on research in uncertainty_quantification.md

Key Concepts:
- Beta distribution: P(successes, failures) = Beta(α, β)
- Confidence: mean = α / (α + β)
- Uncertainty: variance = αβ / ((α+β)²(α+β+1))
- Credible interval: Bayesian confidence bounds
"""

import math
import json
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from datetime import datetime, UTC


@dataclass
class BetaUncertainty:
    """
    Beta distribution for tracking confidence in binary outcomes.

    Implements Bayesian uncertainty tracking where:
    - α (alpha): Number of successful predictions
    - β (beta): Number of failed predictions

    Start with α=1, β=1 for maximum uncertainty (uniform prior).
    Update with each outcome for Bayesian updating.
    """

    successes: int = 1  # α
    failures: int = 1   # β

    def update(self, correct: bool) -> None:
        """
        Update distribution with new outcome.

        Args:
            correct: Whether the prediction was correct
        """
        if correct:
            self.successes += 1
        else:
            self.failures += 1

    def mean(self) -> float:
        """
        Expected confidence (posterior mean).

        Returns:
            Confidence score between 0 and 1
        """
        total = self.successes + self.failures
        return self.successes / total if total > 0 else 0.5

    def variance(self) -> float:
        """
        Uncertainty (posterior variance).

        Returns:
            Variance between 0 and 0.25 (max for Beta(1,1))
        """
        total = self.successes + self.failures
        if total <= 1:
            return 0.25  # Maximum uncertainty for uniform prior

        alpha = self.successes
        beta = self.failures
        return (alpha * beta) / ((total ** 2) * (total + 1))

    def std_dev(self) -> float:
        """Standard deviation of the distribution."""
        return math.sqrt(self.variance())

    def credible_interval(self, confidence: float = 0.95) -> Tuple[float, float]:
        """
        Bayesian credible interval.

        For Beta distribution, approximate using normal approximation
        for simplicity. For exact intervals, would need to use quantile
        functions from scipy.

        Args:
            confidence: Confidence level (e.g., 0.95 for 95% interval)

        Returns:
            (lower_bound, upper_bound) tuple
        """
        # Use normal approximation for credible interval
        # This is reasonable for α, β > 10
        if self.successes + self.failures <= 10:
            # Return conservative interval for small samples
            return (0.0, 1.0)

        import math
        z_score = 1.96  # ~95% confidence

        mean = self.mean()
        std = self.std_dev()

        lower = max(0.0, mean - z_score * std)
        upper = min(1.0, mean + z_score * std)

        return (lower, upper)

    def probability_greater_than(self, threshold: float) -> float:
        """
        Probability that true confidence > threshold.

        Uses incomplete beta function for exact calculation.
        For threshold <= mean, approximates as > 0.5.

        Args:
            threshold: Threshold between 0 and 1

        Returns:
            Probability between 0 and 1
        """
        # For simplicity, use normal approximation
        # In production, use scipy.stats.betainc for exact calculation
        mean = self.mean()
        std = self.std_dev()

        if std == 0:
            return 1.0 if mean > threshold else 0.0

        # P(X > threshold) = 1 - Φ((threshold - mean) / std)
        from math import erf, sqrt
        z = (threshold - mean) / (std * sqrt(2))
        return 0.5 * (1 + erf(-z / sqrt(2)))

    def is_reliable(self, confidence_threshold: float = 0.8,
                   uncertainty_threshold: float = 0.1,
                   min_samples: int = 5) -> bool:
        """
        Determine if this uncertainty estimate is reliable enough
        for automated decision making.

        Args:
            confidence_threshold: Minimum required confidence
            uncertainty_threshold: Maximum allowed uncertainty
            min_samples: Minimum number of observations

        Returns:
            True if reliable for automated decisions
        """
        if (self.successes + self.failures) < min_samples:
            return False

        mean = self.mean()
        variance = self.variance()

        return mean >= confidence_threshold and variance <= uncertainty_threshold

    def to_dict(self) -> Dict[str, int]:
        """Serialize to dictionary for storage."""
        return {
            "successes": self.successes,
            "failures": self.failures
        }

    @classmethod
    def from_dict(cls, data: Dict[str, int]) -> 'BetaUncertainty':
        """Deserialize from dictionary."""
        return cls(successes=data["successes"], failures=data["failures"])

    def __str__(self) -> str:
        return (f"BetaUncertainty(α={self.successes}, β={self.failures}, "
                f"conf={self.mean():.3f}, unc={self.variance():.3f})")

    def __repr__(self) -> str:
        return str(self)


class UncertaintyTracker:
    """
    Tracks uncertainty for multiple relationship types.

    Maintains separate Beta distributions for each relationship type
    to enable type-aware confidence scoring.
    """

    def __init__(self, relationship_types: Optional[List[str]] = None):
        """
        Initialize tracker.

        Args:
            relationship_types: List of relationship types to track.
                              If None, tracks all types.
        """
        self.distributions: Dict[str, BetaUncertainty] = {}
        self.relationship_types = relationship_types
        self.total_updates = 0

        if relationship_types:
            for rel_type in relationship_types:
                self.distributions[rel_type] = BetaUncertainty()

    def update(self, relationship_type: str, correct: bool) -> None:
        """
        Update uncertainty for a specific relationship type.

        Args:
            relationship_type: Type of relationship
            correct: Whether the prediction was correct
        """
        # Normalize relationship type
        rel_type = relationship_type.upper().replace(" ", "_")

        # Create distribution if it doesn't exist
        if rel_type not in self.distributions:
            self.distributions[rel_type] = BetaUncertainty()

        # Update the distribution
        self.distributions[rel_type].update(correct)
        self.total_updates += 1

    def get_confidence(self, relationship_type: str) -> float:
        """
        Get current confidence for a relationship type.

        Args:
            relationship_type: Type of relationship

        Returns:
            Confidence score between 0 and 1
        """
        rel_type = relationship_type.upper().replace(" ", "_")

        if rel_type not in self.distributions:
            # Unknown relationship type - use conservative estimate
            return 0.5

        return self.distributions[rel_type].mean()

    def get_uncertainty(self, relationship_type: str) -> float:
        """
        Get current uncertainty for a relationship type.

        Args:
            relationship_type: Type of relationship

        Returns:
            Uncertainty (variance) between 0 and 0.25
        """
        rel_type = relationship_type.upper().replace(" ", "_")

        if rel_type not in self.distributions:
            return 0.25  # Maximum uncertainty for unknown type

        return self.distributions[rel_type].variance()

    def should_suggest(self, relationship_type: str,
                      confidence_threshold: float = 0.8,
                      uncertainty_threshold: float = 0.1,
                      prob_threshold: float = 0.8) -> bool:
        """
        Determine if automated suggestions should be made for this relationship type.

        Implements the decision framework from our research:
        Only suggest when: conf > 0.8 AND unc < 0.1 AND prob_gt > 0.8

        Args:
            relationship_type: Type of relationship
            confidence_threshold: Minimum confidence threshold
            uncertainty_threshold: Maximum uncertainty threshold
            prob_threshold: Minimum probability > threshold

        Returns:
            True if suggestions should be made
        """
        rel_type = relationship_type.upper().replace(" ", "_")

        if rel_type not in self.distributions:
            return False

        dist = self.distributions[rel_type]

        # Check all three conditions
        if dist.mean() < confidence_threshold:
            return False

        if dist.variance() > uncertainty_threshold:
            return False

        if dist.probability_greater_than(confidence_threshold) < prob_threshold:
            return False

        return True

    def get_all_relationship_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Get statistics for all tracked relationship types.

        Returns:
            Dictionary mapping relationship type to stats
        """
        stats = {}
        for rel_type, dist in self.distributions.items():
            stats[rel_type] = {
                "successes": dist.successes,
                "failures": dist.failures,
                "confidence": dist.mean(),
                "uncertainty": dist.variance(),
                "std_dev": dist.std_dev(),
                "credible_interval_lower": dist.credible_interval()[0],
                "credible_interval_upper": dist.credible_interval()[1],
            }
        return stats

    def get_overall_stats(self) -> Dict[str, float]:
        """
        Get overall statistics across all relationship types.

        Returns:
            Dictionary with overall metrics
        """
        if not self.distributions:
            return {
                "total_samples": 0,
                "avg_confidence": 0.5,
                "avg_uncertainty": 0.25,
                "reliable_types": 0,
                "total_types": 0,
            }

        total_samples = sum(d.successes + d.failures for d in self.distributions.values())
        avg_confidence = sum(d.mean() for d in self.distributions.values()) / len(self.distributions)
        avg_uncertainty = sum(d.variance() for d in self.distributions.values()) / len(self.distributions)
        reliable_count = sum(1 for d in self.distributions.values() if d.is_reliable())

        return {
            "total_samples": total_samples,
            "avg_confidence": avg_confidence,
            "avg_uncertainty": avg_uncertainty,
            "reliable_types": reliable_count,
            "total_types": len(self.distributions),
        }

    def save(self, filepath: str) -> None:
        """Save tracker state to file."""
        data = {
            "relationship_types": self.relationship_types,
            "distributions": {
                rel_type: dist.to_dict()
                for rel_type, dist in self.distributions.items()
            },
            "total_updates": self.total_updates,
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> 'UncertaintyTracker':
        """Load tracker state from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)

        tracker = cls(data.get("relationship_types"))
        tracker.distributions = {
            rel_type: BetaUncertainty.from_dict(dist_data)
            for rel_type, dist_data in data["distributions"].items()
        }
        tracker.total_updates = data["total_updates"]

        return tracker

    def __str__(self) -> str:
        stats = self.get_overall_stats()
        return (f"UncertaintyTracker(samples={stats['total_samples']}, "
                f"types={stats['total_types']}, "
                f"reliable={stats['reliable_types']}, "
                f"avg_conf={stats['avg_confidence']:.3f})")


# Convenience function for quick testing
def test_beta_uncertainty():
    """Test the Beta uncertainty implementation."""
    print("Testing BetaUncertainty...")

    # Test basic functionality
    dist = BetaUncertainty(successes=10, failures=3)
    print(f"Initial: {dist}")
    print(f"Mean: {dist.mean():.3f}")
    print(f"Variance: {dist.variance():.3f}")
    print(f"Credible interval: {dist.credible_interval()}")
    print(f"P(conf > 0.8): {dist.probability_greater_than(0.8):.3f}")

    # Test updating
    dist.update(correct=True)
    dist.update(correct=True)
    dist.update(correct=False)
    print(f"\nAfter 3 updates: {dist}")

    # Test reliability
    print(f"\nIs reliable (conf>0.8, unc<0.1): {dist.is_reliable()}")

    print("\n" + "="*50)
    print("Testing UncertaintyTracker...")

    # Test tracker
    tracker = UncertaintyTracker(["SUPPORTS", "CONTRADICTS", "RELATES_TO"])

    # Simulate updates
    tracker.update("SUPPORTS", correct=True)  # 2 successes, 1 failure (initial 1,1)
    tracker.update("SUPPORTS", correct=True)
    tracker.update("SUPPORTS", correct=False)

    tracker.update("CONTRADICTS", correct=False)
    tracker.update("CONTRADICTS", correct=False)

    tracker.update("RELATES_TO", correct=True)
    tracker.update("RELATES_TO", correct=True)
    tracker.update("RELATES_TO", correct=True)

    print(f"Tracker: {tracker}")
    print(f"\nAll stats:")
    for rel_type, stats in tracker.get_all_relationship_stats().items():
        print(f"  {rel_type}: conf={stats['confidence']:.3f}, unc={stats['uncertainty']:.3f}")

    print(f"\nShould suggest SUPPORTS? {tracker.should_suggest('SUPPORTS')}")
    print(f"Should suggest CONTRADICTS? {tracker.should_suggest('CONTRADICTS')}")

    print(f"\nOverall stats: {tracker.get_overall_stats()}")


if __name__ == "__main__":
    test_beta_uncertainty()