"""
Tests for src/coaching/coaching_metrics.py
"""

import pytest
import numpy as np
import sys

sys.path.insert(0, '/Users/kevinwang/Documents/20Projects/fencingbuddy')

from src.coaching.coaching_metrics import (
    CoachingMetrics,
    FencingMetrics,
    extract_lunge_speed,
    extract_arm_extension,
    VELOCITY_START,
    VELOCITY_END,
    ARM_EXTENSION_START,
)


class TestFencingMetrics:
    """Test FencingMetrics dataclass."""

    def test_default_metrics(self):
        """Test default metrics are zeros."""
        metrics = FencingMetrics()
        assert metrics.lunge_speed == 0.0
        assert metrics.velocity_magnitude == 0.0
        assert metrics.arm_extension_ratio == 0.0
        assert metrics.is_attacking is False
        assert metrics.is_retreating is False

    def test_metric_assignment(self):
        """Test metrics can be assigned."""
        metrics = FencingMetrics(
            lunge_speed=20.0,
            arm_extension_ratio=0.85,
            is_attacking=True,
        )
        assert metrics.lunge_speed == 20.0
        assert metrics.arm_extension_ratio == 0.85
        assert metrics.is_attacking is True


class TestCoachingMetrics:
    """Test CoachingMetrics class."""

    def test_initialization(self):
        """Test CoachingMetrics initializes correctly."""
        cm = CoachingMetrics()
        assert cm.history_size == 5
        assert len(cm._velocity_history) == 0

    def test_custom_history_size(self):
        """Test CoachingMetrics with custom history size."""
        cm = CoachingMetrics(history_size=10)
        assert cm.history_size == 10

    def test_compute_fencer_metrics_shape(self):
        """Test compute_fencer_metrics returns FencingMetrics."""
        cm = CoachingMetrics()
        # Create mock 101-dim features
        features = np.random.rand(101).astype(np.float32)
        metrics = cm.compute_fencer_metrics(features)
        assert isinstance(metrics, FencingMetrics)

    def test_compute_fencer_metrics_with_opponent(self):
        """Test compute_fencer_metrics with opponent features."""
        cm = CoachingMetrics()
        son_features = np.random.rand(101).astype(np.float32)
        opp_features = np.random.rand(101).astype(np.float32)

        metrics = cm.compute_fencer_metrics(son_features, opp_features)
        assert isinstance(metrics, FencingMetrics)

    def test_velocity_history_tracking(self):
        """Test velocity history is tracked."""
        cm = CoachingMetrics(history_size=3)

        # Add multiple velocity vectors
        for _ in range(5):
            velocity = np.random.rand(24).astype(np.float32)
            cm._update_velocity_history(velocity)

        # Should only keep last 3
        assert len(cm._velocity_history) == 3

    def test_predictability_computation(self):
        """Test predictability score computation."""
        cm = CoachingMetrics()

        # Add similar velocities (high predictability)
        for _ in range(3):
            velocity = np.ones(24, dtype=np.float32) * 10.0
            cm._update_velocity_history(velocity)

        score = cm._compute_predictability()
        assert 0.0 <= score <= 1.0
        # Similar velocities should give high predictability
        assert score > 0.5

    def test_reset_clears_history(self):
        """Test reset clears velocity history."""
        cm = CoachingMetrics()

        velocity = np.random.rand(24).astype(np.float32)
        cm._update_velocity_history(velocity)
        assert len(cm._velocity_history) == 1

        cm.reset()
        assert len(cm._velocity_history) == 0


class TestExtractFunctions:
    """Test convenience extraction functions."""

    def test_extract_lunge_speed(self):
        """Test lunge speed extraction."""
        features = np.zeros(101, dtype=np.float32)
        # Set forward velocity values (indices 0, 2, 4, ..., 22 in velocity slice)
        # These correspond to original indices VELOCITY_START + [0, 2, 4, ..., 22]
        forward_indices = list(range(0, 24, 2))  # [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]
        for i in forward_indices:
            features[VELOCITY_START + i] = 10.0

        speed = extract_lunge_speed(features)
        assert speed == pytest.approx(10.0)

    def test_extract_arm_extension(self):
        """Test arm extension extraction."""
        features = np.zeros(101, dtype=np.float32)
        # Set arm extension values (index 44 = second arm extension value)
        features[ARM_EXTENSION_START + 1] = 0.85

        extension = extract_arm_extension(features)
        assert extension == pytest.approx(0.85)
