"""
Tests for src/recognition/action_classifier.py
"""

import pytest
import numpy as np
import sys

sys.path.insert(0, '/Users/kevinwang/Documents/20Projects/fencingbuddy')

from src.recognition.action_classifier import (
    ActionClassifier,
    ActionType,
    ActionResult,
    ATTACK_LUNGE_SPEED,
    ATTACK_ARM_EXTENSION,
)


class TestActionType:
    """Test ActionType enum."""

    def test_action_types_exist(self):
        """Test all expected action types exist."""
        expected = [
            "idle", "advance", "retreat", "attack", "attack_prep",
            "parry", "riposte", "counter_attack", "fleche", "recovery"
        ]
        actual = [a.value for a in ActionType]
        for e in expected:
            assert e in actual


class TestActionClassifier:
    """Test ActionClassifier class."""

    def test_initialization(self):
        """Test ActionClassifier initializes correctly."""
        classifier = ActionClassifier()
        assert classifier.history_size == 10
        assert len(classifier._action_history) == 0

    def test_custom_history_size(self):
        """Test ActionClassifier with custom history size."""
        classifier = ActionClassifier(history_size=5)
        assert classifier.history_size == 5

    def test_classify_returns_action_result(self):
        """Test classify returns ActionResult."""
        from src.coaching.coaching_metrics import FencingMetrics

        classifier = ActionClassifier()
        features = np.random.rand(2, 101).astype(np.float32)

        son_metrics = FencingMetrics(lunge_speed=5.0, arm_extension_ratio=0.5)
        opp_metrics = FencingMetrics(lunge_speed=3.0, arm_extension_ratio=0.5)

        result = classifier.classify(features, son_metrics, opp_metrics)
        assert isinstance(result, ActionResult)
        assert isinstance(result.son_action, ActionType)
        assert isinstance(result.opp_action, ActionType)

    def test_idle_detection(self):
        """Test IDLE action is detected for low movement."""
        from src.coaching.coaching_metrics import FencingMetrics

        classifier = ActionClassifier()
        features = np.random.rand(2, 101).astype(np.float32)

        # Low movement metrics
        son_metrics = FencingMetrics(
            lunge_speed=1.0,
            arm_extension_ratio=0.5,
            com_velocity_x=0.5,
        )
        opp_metrics = FencingMetrics(
            lunge_speed=1.0,
            arm_extension_ratio=0.5,
            com_velocity_x=0.5,
        )

        result = classifier.classify(features, son_metrics, opp_metrics)
        assert result.son_action == ActionType.IDLE

    def test_attack_detection(self):
        """Test ATTACK action is detected for high lunge speed + arm extension."""
        from src.coaching.coaching_metrics import FencingMetrics

        classifier = ActionClassifier()
        features = np.random.rand(2, 101).astype(np.float32)

        # Attack metrics: high lunge speed + arm extension
        son_metrics = FencingMetrics(
            lunge_speed=20.0,  # > ATTACK_LUNGE_SPEED (15)
            arm_extension_ratio=0.9,  # > ATTACK_ARM_EXTENSION (0.8)
            com_velocity_x=10.0,  # Forward motion
        )
        opp_metrics = FencingMetrics(
            lunge_speed=2.0,
            arm_extension_ratio=0.5,
        )

        result = classifier.classify(features, son_metrics, opp_metrics)
        assert result.son_action == ActionType.ATTACK

    def test_fleche_detection(self):
        """Test FLECHE action is detected."""
        from src.coaching.coaching_metrics import FencingMetrics

        classifier = ActionClassifier()
        features = np.random.rand(2, 101).astype(np.float32)

        # Fleche metrics: deep lean + high speed
        son_metrics = FencingMetrics(
            lunge_speed=30.0,
            torso_forward_lean=0.5,  # > FLECHE_TORSO_LEAN (0.4)
            arm_extension_ratio=0.9,
            com_velocity_x=15.0,
            is_attacking=True,
        )
        opp_metrics = FencingMetrics(lunge_speed=2.0)

        result = classifier.classify(features, son_metrics, opp_metrics)
        assert result.son_action == ActionType.FLECHE

    def test_parry_detection(self):
        """Test PARRY action is detected."""
        from src.coaching.coaching_metrics import FencingMetrics

        classifier = ActionClassifier()
        features = np.random.rand(2, 101).astype(np.float32)

        # Parry: opponent attacking + good blade position
        son_metrics = FencingMetrics(
            lunge_speed=2.0,
            arm_extension_ratio=0.8,  # Good blade position
            weapon_elbow_angle=1.0,  # Guard position
        )
        opp_metrics = FencingMetrics(
            lunge_speed=20.0,  # Attacking
            arm_extension_ratio=0.9,
            is_attacking=True,
        )

        result = classifier.classify(features, son_metrics, opp_metrics)
        assert result.son_action == ActionType.PARRY

    def test_retreat_detection(self):
        """Test RETREAT action is detected."""
        from src.coaching.coaching_metrics import FencingMetrics

        classifier = ActionClassifier()
        features = np.random.rand(2, 101).astype(np.float32)

        son_metrics = FencingMetrics(
            lunge_speed=-5.0,  # Negative = retreating
            is_retreating=True,
        )
        opp_metrics = FencingMetrics(lunge_speed=5.0)

        result = classifier.classify(features, son_metrics, opp_metrics)
        assert result.son_action == ActionType.RETREAT

    def test_action_history_tracking(self):
        """Test action history is tracked."""
        from src.coaching.coaching_metrics import FencingMetrics

        classifier = ActionClassifier(history_size=5)
        features = np.random.rand(2, 101).astype(np.float32)

        son_metrics = FencingMetrics(lunge_speed=5.0)
        opp_metrics = FencingMetrics(lunge_speed=3.0)

        # Classify multiple times
        for _ in range(3):
            classifier.classify(features, son_metrics, opp_metrics)

        assert len(classifier._action_history) == 3

    def test_get_action_sequence(self):
        """Test get_action_sequence returns list of actions."""
        from src.coaching.coaching_metrics import FencingMetrics

        classifier = ActionClassifier()
        features = np.random.rand(2, 101).astype(np.float32)

        son_metrics = FencingMetrics(lunge_speed=5.0)
        opp_metrics = FencingMetrics(lunge_speed=3.0)

        classifier.classify(features, son_metrics, opp_metrics)
        classifier.classify(features, son_metrics, opp_metrics)

        sequence = classifier.get_action_sequence()
        assert len(sequence) == 2
        assert all(isinstance(a, ActionType) for a in sequence)

    def test_reset(self):
        """Test reset clears state."""
        from src.coaching.coaching_metrics import FencingMetrics

        classifier = ActionClassifier()
        features = np.random.rand(2, 101).astype(np.float32)

        son_metrics = FencingMetrics(lunge_speed=5.0)
        opp_metrics = FencingMetrics(lunge_speed=3.0)

        classifier.classify(features, son_metrics, opp_metrics)
        assert len(classifier._action_history) == 1

        classifier.reset()
        assert len(classifier._action_history) == 0
        assert classifier._last_son_action is None


class TestActionResult:
    """Test ActionResult dataclass."""

    def test_action_result_fields(self):
        """Test ActionResult has all required fields."""
        result = ActionResult(
            action=ActionType.ATTACK,
            confidence=0.9,
            son_action=ActionType.ATTACK,
            opp_action=ActionType.IDLE,
            timestamp=1.5,
            notes="Test attack",
        )
        assert result.action == ActionType.ATTACK
        assert result.confidence == 0.9
        assert result.son_action == ActionType.ATTACK
        assert result.opp_action == ActionType.IDLE
        assert result.timestamp == 1.5
        assert result.notes == "Test attack"
