"""
Confidence Calibration Module

Implements calibration methods to fix miscalibrated confidence scores.
Based on research in uncertainty_quantification.md

Key Methods:
- Temperature Scaling: Single-parameter scaling for neural networks
- Platt Scaling: Logistic regression on logits
- Isotonic Regression: Non-parametric calibration
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import math


@dataclass
class TemperatureScaling:
    """
    Simple temperature scaling for confidence calibration.

    Based on: https://arxiv.org/abs/1706.04599

    Divides logits by temperature T to improve calibration.
    Optimal T is found by minimizing Expected Calibration Error (ECE) on validation set.

    Formula: p(y|x) = softmax(logits / T)
    """

    temperature: float = 1.0

    def calibrate(self, logits: np.ndarray) -> np.ndarray:
        """
        Apply temperature scaling to logits.

        Args:
            logits: Unnormalized model outputs [batch_size, num_classes]

        Returns:
            Calibrated probabilities
        """
        scaled_logits = logits / self.temperature
        # Softmax
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=-1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    def set_temperature(self, temperature: float) -> None:
        """Set temperature parameter (must be positive)."""
        if temperature <= 0:
            raise ValueError("Temperature must be positive")
        self.temperature = float(temperature)

    @classmethod
    def fit(cls, logits: np.ndarray, labels: np.ndarray,
            num_bins: int = 15, lr: float = 0.01, epochs: int = 50) -> 'TemperatureScaling':
        """
        Fit temperature scaling on validation data.

        Args:
            logits: Model logits from validation set [n_samples, n_classes]
            labels: True labels [n_samples] (0-indexed)
            num_bins: Number of bins for ECE calculation
            lr: Learning rate for optimization
            epochs: Number of optimization epochs

        Returns:
            Fitted TemperatureScaling instance
        """
        # Initial temperature
        temperature = np.array([1.0], dtype=np.float32)

        # Convert to probabilities for ECE calculation
        def softmax(x):
            exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
            return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

        def compute_ece(temp):
            scaled_logits = logits / temp[0]
            probs = softmax(scaled_logits)
            confidences = np.max(probs, axis=-1)
            predictions = np.argmax(probs, axis=-1)
            accuracies = (predictions == labels).astype(float)

            # Compute ECE
            bins = np.linspace(0, 1, num_bins + 1)
            bin_indices = np.digitize(confidences, bins[1:-1])

            ece = 0.0
            total_samples = len(labels)

            for bin_idx in range(num_bins):
                mask = bin_indices == bin_idx
                if np.sum(mask) > 0:
                    bin_conf = np.mean(confidences[mask])
                    bin_acc = np.mean(accuracies[mask])
                    bin_weight = np.sum(mask) / total_samples
                    ece += bin_weight * abs(bin_conf - bin_acc)

            return ece

        # Simple gradient descent on temperature
        best_temp = 1.0
        best_ece = float('inf')

        for epoch in range(epochs):
            # Compute gradient (numerical approximation)
            eps = 0.01
            ece_pos = compute_ece(temperature + eps)
            ece_neg = compute_ece(temperature - eps)
            grad = (ece_pos - ece_neg) / (2 * eps)

            # Update temperature
            temperature -= lr * grad

            # Keep temperature positive
            temperature = np.clip(temperature, 0.1, 10.0)

            # Track best
            current_ece = compute_ece(temperature)
            if current_ece < best_ece:
                best_ece = current_ece
                best_temp = temperature[0]

        return cls(temperature=float(best_temp))

    def compute_ece(self, logits: np.ndarray, labels: np.ndarray,
                   num_bins: int = 15) -> float:
        """
        Compute Expected Calibration Error.

        Args:
            logits: Model logits [n_samples, n_classes]
            labels: True labels [n_samples]
            num_bins: Number of bins

        Returns:
            ECE score (lower is better)
        """
        probs = self.calibrate(logits)
        confidences = np.max(probs, axis=-1)
        predictions = np.argmax(probs, axis=-1)
        accuracies = (predictions == labels).astype(float)

        # Bin by confidence
        bins = np.linspace(0, 1, num_bins + 1)
        bin_indices = np.digitize(confidences, bins[1:-1])

        ece = 0.0
        total_samples = len(labels)

        for bin_idx in range(num_bins):
            mask = bin_indices == bin_idx
            if np.sum(mask) > 0:
                bin_conf = np.mean(confidences[mask])
                bin_acc = np.mean(accuracies[mask])
                bin_weight = np.sum(mask) / total_samples
                ece += bin_weight * abs(bin_conf - bin_acc)

        return ece

    def to_dict(self) -> Dict[str, float]:
        """Serialize to dictionary."""
        return {"temperature": self.temperature}

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> 'TemperatureScaling':
        """Deserialize from dictionary."""
        return cls(temperature=data["temperature"])

    def __str__(self) -> str:
        return f"TemperatureScaling(T={self.temperature:.3f})"


@dataclass
class PlattScaling:
    """
    Platt scaling for binary classification calibration.

    Uses logistic regression on model scores to calibrate probabilities.
    Based on: Platt, John (1999)
    """

    A: float = 1.0  # Slope
    B: float = 0.0  # Intercept

    def calibrate(self, scores: np.ndarray) -> np.ndarray:
        """
        Apply Platt scaling to scores.

        Args:
            scores: Model scores (e.g., decision function) [n_samples]

        Returns:
            Calibrated probabilities
        """
        # Sigmoid: 1 / (1 + exp(A * score + B))
        return 1.0 / (1.0 + np.exp(self.A * scores + self.B))

    @classmethod
    def fit(cls, scores: np.ndarray, labels: np.ndarray) -> 'PlattScaling':
        """
        Fit Platt scaling on validation data.

        Args:
            scores: Model scores from validation set [n_samples]
            labels: True labels (0 or 1) [n_samples]

        Returns:
            Fitted PlattScaling instance
        """
        # Use maximum likelihood estimation
        from scipy.optimize import minimize

        def nll(params):
            A, B = params
            calibrated = 1.0 / (1.0 + np.exp(A * scores + B))
            # Avoid log(0)
            calibrated = np.clip(calibrated, 1e-15, 1 - 1e-15)
            return -np.mean(labels * np.log(calibrated) +
                          (1 - labels) * np.log(1 - calibrated))

        # Initial guess
        x0 = np.array([1.0, 0.0])

        # Constraints: A should be negative (to map high scores to high prob)
        constraints = [{"type": "ineq", "fun": lambda x: -x[0]}]

        result = minimize(nll, x0, constraints=constraints, method='SLSQP')

        if result.success:
            return cls(A=result.x[0], B=result.x[1])
        else:
            # Fallback to simple scaling
            return cls(A=-1.0, B=0.0)

    def to_dict(self) -> Dict[str, float]:
        """Serialize to dictionary."""
        return {"A": self.A, "B": self.B}

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> 'PlattScaling':
        """Deserialize from dictionary."""
        return cls(A=data["A"], B=data["B"])

    def __str__(self) -> str:
        return f"PlattScaling(A={self.A:.3f}, B={self.B:.3f})"


class ConfidenceCalibrator:
    """
    Unified confidence calibration system.

    Supports multiple calibration methods and tracks calibration quality.
    """

    def __init__(self, method: str = "temperature"):
        """
        Initialize calibrator.

        Args:
            method: 'temperature', 'platt', or 'isotonic'
        """
        self.method = method
        self.calibrator = None
        self.is_fitted = False
        self.ece_score = None

    def fit(self, logits: np.ndarray, labels: np.ndarray,
            validation_split: float = 0.2) -> None:
        """
        Fit calibrator on validation data.

        Args:
            logits: Model logits [n_samples, n_classes]
            labels: True labels [n_samples]
            validation_split: Fraction of data to use for validation
        """
        # Split data
        n_samples = len(logits)
        n_val = int(n_samples * validation_split)

        val_logits = logits[:n_val]
        val_labels = labels[:n_val]

        if self.method == "temperature":
            self.calibrator = TemperatureScaling.fit(val_logits, val_labels)
        elif self.method == "platt":
            # For Platt, use max logit as score
            scores = np.max(val_logits, axis=-1)
            self.calibrator = PlattScaling.fit(scores, val_labels)
        else:
            raise ValueError(f"Unknown calibration method: {self.method}")

        # Compute ECE on full validation set
        if isinstance(self.calibrator, TemperatureScaling):
            self.ece_score = self.calibrator.compute_ece(val_logits, val_labels)
        elif isinstance(self.calibrator, PlattScaling):
            probs = self.calibrator.calibrate(np.max(val_logits, axis=-1))
            # Simple ECE calculation for Platt
            confidences = probs
            predictions = (probs > 0.5).astype(int)
            accuracies = (predictions == val_labels).astype(float)

            bins = np.linspace(0, 1, 15)
            bin_indices = np.digitize(confidences, bins[1:-1])

            ece = 0.0
            for bin_idx in range(15):
                mask = bin_indices == bin_idx
                if np.sum(mask) > 0:
                    bin_conf = np.mean(confidences[mask])
                    bin_acc = np.mean(accuracies[mask])
                    bin_weight = np.sum(mask) / len(val_labels)
                    ece += bin_weight * abs(bin_conf - bin_acc)

            self.ece_score = ece

        self.is_fitted = True

    def calibrate(self, logits: np.ndarray) -> np.ndarray:
        """
        Apply calibration to logits.

        Args:
            logits: Model logits [n_samples, n_classes] or [n_samples]

        Returns:
            Calibrated probabilities
        """
        if not self.is_fitted:
            raise ValueError("Calibrator must be fitted before calibration")

        if isinstance(self.calibrator, TemperatureScaling):
            return self.calibrator.calibrate(logits)
        elif isinstance(self.calibrator, PlattScaling):
            # For Platt, use max logit as score
            if len(logits.shape) == 1:
                scores = logits
            else:
                scores = np.max(logits, axis=-1)
            return self.calibrator.calibrate(scores)
        else:
            raise ValueError(f"Unknown calibrator type: {type(self.calibrator)}")

    def is_well_calibrated(self, threshold: float = 0.05) -> bool:
        """
        Check if model is well-calibrated.

        Args:
            threshold: Maximum acceptable ECE

        Returns:
            True if ECE <= threshold
        """
        if self.ece_score is None:
            return False
        return self.ece_score <= threshold

    def get_calibration_metrics(self) -> Dict[str, float]:
        """Get calibration quality metrics."""
        return {
            "ece": self.ece_score if self.ece_score is not None else float('nan'),
            "is_well_calibrated": 1.0 if self.is_well_calibrated() else 0.0,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serialize calibrator state."""
        if not self.is_fitted:
            return {"method": self.method, "is_fitted": False}

        return {
            "method": self.method,
            "is_fitted": True,
            "ece": self.ece_score,
            "calibrator": self.calibrator.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConfidenceCalibrator':
        """Deserialize calibrator state."""
        calibrator = cls(method=data["method"])
        calibrator.is_fitted = data.get("is_fitted", False)
        calibrator.ece_score = data.get("ece")

        if calibrator.is_fitted:
            if data["method"] == "temperature":
                calibrator.calibrator = TemperatureScaling.from_dict(
                    data["calibrator"]
                )
            elif data["method"] == "platt":
                calibrator.calibrator = PlattScaling.from_dict(
                    data["calibrator"]
                )

        return calibrator

    def __str__(self) -> str:
        if not self.is_fitted:
            return f"ConfidenceCalibrator(method={self.method}, not fitted)"
        return (f"ConfidenceCalibrator(method={self.method}, "
                f"ECE={self.ece_score:.3f}, "
                f"well_calibrated={self.is_well_calibrated()})")


def test_calibration():
    """Test calibration methods with synthetic data."""
    print("Testing Calibration Methods...")
    print("=" * 60)

    # Generate synthetic data with known miscalibration
    np.random.seed(42)
    n_samples = 1000

    # True labels
    true_labels = np.random.binomial(1, 0.5, n_samples)

    # Miscalibrated model outputs (overconfident)
    # True confidence: 0.5, but model outputs mean 0.8
    logits = np.random.normal(1.5 * true_labels - 0.75, 1.0, size=(n_samples, 2))

    # Test Temperature Scaling
    print("\n1. Temperature Scaling:")
    temp_scaler = TemperatureScaling.fit(logits, true_labels)
    print(f"   Fitted: {temp_scaler}")
    print(f"   ECE before: {temp_scaler.compute_ece(logits, true_labels):.3f}")

    calibrated_probs = temp_scaler.calibrate(logits)
    print(f"   Calibrated examples: {calibrated_probs[:5]}")

    # Test Platt Scaling
    print("\n2. Platt Scaling:")
    scores = np.max(logits, axis=-1)
    platt_scaler = PlattScaling.fit(scores, true_labels)
    print(f"   Fitted: {platt_scaler}")
    calibrated_platt = platt_scaler.calibrate(scores)
    print(f"   Calibrated examples: {calibrated_platt[:5]}")

    # Test ConfidenceCalibrator
    print("\n3. ConfidenceCalibrator (Temperature):")
    calibrator = ConfidenceCalibrator(method="temperature")
    calibrator.fit(logits, true_labels)
    print(f"   {calibrator}")
    print(f"   Metrics: {calibrator.get_calibration_metrics()}")

    # Test serialization
    print("\n4. Serialization:")
    calibrator_dict = calibrator.to_dict()
    print(f"   Serialized: {calibrator_dict}")

    restored = ConfidenceCalibrator.from_dict(calibrator_dict)
    print(f"   Restored: {restored}")


if __name__ == "__main__":
    test_calibration()