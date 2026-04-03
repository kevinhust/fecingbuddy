"""
FencerAI Coaching Metrics
=======================
Version: 2.0 | Last Updated: 2026-04-02

Extracts fencing-specific metrics from 101-dimensional feature vectors.
These metrics feed the coaching engine to generate real-time alerts.

Feature Index Reference (per ARCHITECTURE.md):
    0-23:   Static Geometry (12 keypoints × 2 coords)
    24-25:  Center of Mass (CoM)
    26-36:  Distance Features (11 dims)
    37-40:  Angles (4 dims)
    41-42:  Torso Orientation (2 dims)
    43-48:  Arm Extension (6 dims)
    49-72:  Velocity (24 dims, EMA smoothed)
    73-96:  Acceleration (24 dims, EMA smoothed)
    97-98:  CoM Velocity (2 dims)
    99:     CoM Acceleration (1 dim)
    100:    Audio Touch Flag
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List
import numpy as np

from src.utils.constants import (
    FEATURE_DIM,
    COCO_INDICES,
)


# =============================================================================
# Coaching Metrics Dataclass
# =============================================================================

@dataclass
class FencingMetrics:
    """Computed fencing metrics for a single fencer."""
    # Distance metrics
    distance_to_opponent: float = 0.0  # Inter-fencer pelvis distance (normalized)

    # Velocity metrics (from velocity indices 49-72)
    lunge_speed: float = 0.0  # Magnitude of forward velocity
    lateral_speed: float = 0.0  # Magnitude of lateral velocity
    velocity_magnitude: float = 0.0  # Total velocity magnitude

    # Acceleration metrics (from acceleration indices 73-96)
    acceleration_magnitude: float = 0.0  # Total acceleration magnitude
    is_decelerating: bool = False  # True if negative acceleration

    # Arm extension (indices 43-48)
    arm_extension_ratio: float = 0.0  # 0-1, full extension = 1.0
    weapon_arm_elevation: float = 0.0  # Angle relative to horizontal

    # Angular metrics (indices 37-40)
    front_knee_angle: float = 0.0
    back_knee_angle: float = 0.0
    weapon_elbow_angle: float = 0.0
    torso_lean_angle: float = 0.0

    # Torso orientation (indices 41-42)
    torso_forward lean: float = 0.0
    torso_lateral_tilt: float = 0.0

    # CoM metrics (indices 97-98)
    com_velocity_x: float = 0.0  # Forward/backward
    com_velocity_y: float = 0.0  # Up/down

    # Derived metrics
    is_retreating: bool = False  # True if moving backward
    is_attacking: bool = False  # True if lunge speed high and forward
    is_recovering: bool = False  # True if post-lunge deceleration
    predictability_score: float = 0.0  # 0-1, high = predictable

    # Opponent-aware metrics
    opponent_distance_change: float = 0.0  # Positive = closing
    relative_speed: float = 0.0  # Son speed - Opp speed


# =============================================================================
# Feature Index Constants for Coaching
# =============================================================================

# Velocity feature indices (49-72 = 24 dims for 12 keypoints × 2 coords)
VELOCITY_START = 49
VELOCITY_END = 73

# Acceleration feature indices (73-96 = 24 dims)
ACCELERATION_START = 73
ACCELERATION_END = 97

# Arm extension indices (43-48 = 6 dims)
ARM_EXTENSION_START = 43
ARM_EXTENSION_END = 49

# Angular indices (37-40 = 4 dims)
ANGLE_START = 37
ANGLE_END = 41

# Torso orientation indices (41-42 = 2 dims)
TORSO_START = 41
TORSO_END = 43

# CoM velocity indices (97-98 = 2 dims)
COM_VELOCITY_START = 97
COM_VELOCITY_END = 99


# =============================================================================
# Metric Thresholds
# =============================================================================

# Distance thresholds (normalized units)
DISTANCE_OPTIMAL_ATTACK = 0.15  # Optimal attack distance
DISTANCE_TOO_FAR = 0.30  # Too far to attack
DISTANCE_TOO_CLOSE = 0.05  # Too close (in-fighting)

# Velocity thresholds (normalized units per second)
LUNGE_SPEED_THRESHOLD = 15.0  # Minimum lunge speed
ATTACK_PREP_SPEED = 5.0  # Speed during preparation
RETREAT_SPEED_THRESHOLD = -3.0  # Negative = retreating

# Arm extension thresholds
ARM_EXTENSION_MIN = 0.70  # Below this = weak extension
ARM_EXTENSION_FULL = 0.90  # Full extension

# Acceleration thresholds
DECELERATION_THRESHOLD = -10.0  # Negative = decelerating
RECOVERY_ACCEL_THRESHOLD = 5.0  # Minimum recovery acceleration

# Predictability threshold
PREDICTABILITY_HIGH = 0.8  # Velocity variance > this = predictable

# Angle thresholds (radians)
TORSO_LEAN_DANGER = 0.5  # Leaning too far forward
KNEE_ANGLE_DEEP = 1.0  # Deep lunge (good for power)


# =============================================================================
# Metric Extraction Functions
# =============================================================================

class CoachingMetrics:
    """
    Extracts fencing-specific metrics from 101-dim feature vectors.

    Example:
        >>> cm = CoachingMetrics()
        >>> metrics = cm.compute_fencer_metrics(features[0])  # Son's features
        >>> print(f"Lunge speed: {metrics.lunge_speed:.2f}")
    """

    def __init__(self, history_size: int = 5):
        """
        Initialize coaching metrics extractor.

        Args:
            history_size: Number of past frames to use for trend analysis
        """
        self.history_size = history_size
        self._velocity_history: List[np.ndarray] = []

    def compute_fencer_metrics(
        self,
        features: np.ndarray,
        opponent_features: Optional[np.ndarray] = None,
    ) -> FencingMetrics:
        """
        Compute all metrics for a single fencer from 101-dim features.

        Args:
            features: 101-dimensional feature vector for one fencer
            opponent_features: Optional 101-dim vector for opponent

        Returns:
            FencingMetrics dataclass with all computed metrics
        """
        metrics = FencingMetrics()

        # Distance to opponent (index 26 - inter-fencer pelvis distance)
        if opponent_features is not None:
            metrics.distance_to_opponent = float(features[26])
            metrics.opponent_distance_change = self._compute_distance_change(
                features[26], opponent_features[26]
            )

        # Velocity features (49-72)
        velocity = features[VELOCITY_START:VELOCITY_END]
        metrics.velocity_magnitude = float(np.linalg.norm(velocity))

        # Lunge speed = forward velocity (use x-components of keypoints)
        # Keypoints in order: x0, y0, x1, y1, ... for 12 keypoints
        # Forward motion = increasing x for left fencer
        forward_velocity_indices = list(range(0, 24, 2))  # x-components
        lateral_velocity_indices = list(range(1, 24, 2))  # y-components

        metrics.lunge_speed = float(np.mean([velocity[i] for i in forward_velocity_indices]))
        metrics.lateral_speed = float(np.mean([velocity[i] for i in lateral_velocity_indices]))

        # CoM velocity (97-98)
        metrics.com_velocity_x = float(features[COM_VELOCITY_START])
        metrics.com_velocity_y = float(features[COM_VELOCITY_START + 1])

        # Acceleration features (73-96)
        acceleration = features[ACCELERATION_START:ACCELERATION_END]
        metrics.acceleration_magnitude = float(np.linalg.norm(acceleration))
        metrics.is_decelerating = metrics.acceleration_magnitude < DECELERATION_THRESHOLD

        # Arm extension features (43-48)
        arm_ext = features[ARM_EXTENSION_START:ARM_EXTENSION_END]
        if len(arm_ext) >= 6:
            metrics.arm_extension_ratio = float(arm_ext[1])  # Total reach ratio
            metrics.weapon_arm_elevation = float(arm_ext[5])  # Elevation angle

        # Angular features (37-40)
        angles = features[ANGLE_START:ANGLE_END]
        if len(angles) >= 4:
            metrics.front_knee_angle = float(angles[0])
            metrics.back_knee_angle = float(angles[1])
            metrics.weapon_elbow_angle = float(angles[2])
            metrics.torso_lean_angle = float(angles[3])

        # Torso orientation (41-42)
        torso = features[TORSO_START:TORSO_END]
        if len(torso) >= 2:
            metrics.torso_forward_lean = float(torso[0])
            metrics.torso_lateral_tilt = float(torso[1])

        # Derived metrics
        metrics.is_retreating = metrics.lunge_speed < RETREAT_SPEED_THRESHOLD
        metrics.is_attacking = (
            metrics.lunge_speed > LUNGE_SPEED_THRESHOLD
            and metrics.arm_extension_ratio > ARM_EXTENSION_MIN
        )
        metrics.is_recovering = metrics.is_decelerating and metrics.lunge_speed > 0

        # Update velocity history and compute predictability
        self._update_velocity_history(velocity)
        metrics.predictability_score = self._compute_predictability()

        # Relative speed
        if opponent_features is not None:
            opp_velocity = opponent_features[VELOCITY_START:VELOCITY_END]
            opp_lunge = float(np.mean([opp_velocity[i] for i in forward_velocity_indices]))
            metrics.relative_speed = metrics.lunge_speed - opp_lunge

        return metrics

    def compute_both_fencers_metrics(
        self,
        features: np.ndarray,
    ) -> Tuple[FencingMetrics, FencingMetrics, Optional[FencingMetrics]]:
        """
        Compute metrics for both fencers from (2, 101) feature matrix.

        Args:
            features: (2, 101) feature matrix

        Returns:
            Tuple of (son_metrics, opponent_metrics, relative_metrics)
        """
        son_features = features[0]
        opp_features = features[1]

        # Compute individual metrics
        son_metrics = self.compute_fencer_metrics(son_features, opp_features)
        opp_metrics = self.compute_fencer_metrics(opp_features, son_features)

        # Relative metrics
        relative = FencingMetrics()
        relative.opponent_distance_change = son_metrics.opponent_distance_change
        relative.relative_speed = son_metrics.relative_speed

        return son_metrics, opp_metrics, relative

    def _compute_distance_change(
        self,
        current_distance: float,
        previous_distance: float,
    ) -> float:
        """Compute change in distance to opponent."""
        return float(current_distance - previous_distance)

    def _update_velocity_history(self, velocity: np.ndarray) -> None:
        """Update velocity history for trend analysis."""
        self._velocity_history.append(velocity.copy())
        if len(self._velocity_history) > self.history_size:
            self._velocity_history.pop(0)

    def _compute_predictability(self) -> float:
        """
        Compute predictability score from velocity history.

        Returns:
            Float 0-1, where 1 = highly predictable (low variance)
        """
        if len(self._velocity_history) < 2:
            return 0.0

        # Compute variance across history
        stacked = np.stack(self._velocity_history)
        variance = float(np.var(stacked))

        # Normalize variance to 0-1 predictability score
        # Higher variance = less predictable = lower score
        max_variance = 100.0  # Calibrated for typical velocity variance
        predictability = 1.0 - min(variance / max_variance, 1.0)

        return predictability

    def reset(self) -> None:
        """Reset velocity history."""
        self._velocity_history.clear()


# =============================================================================
# Convenience Functions
# =============================================================================

def extract_lunge_speed(features: np.ndarray) -> float:
    """
    Extract lunge speed from feature vector.

    Args:
        features: 101-dimensional feature vector

    Returns:
        Lunge speed (forward velocity magnitude)
    """
    velocity = features[VELOCITY_START:VELOCITY_END]
    forward_indices = list(range(0, 24, 2))
    return float(np.mean([velocity[i] for i in forward_indices]))


def extract_arm_extension(features: np.ndarray) -> float:
    """
    Extract arm extension ratio from feature vector.

    Args:
        features: 101-dimensional feature vector

    Returns:
        Arm extension ratio (0-1, full extension = 1.0)
    """
    arm_ext = features[ARM_EXTENSION_START:ARM_EXTENSION_END]
    if len(arm_ext) >= 2:
        return float(arm_ext[1])
    return 0.0


def extract_recovery_speed(features: np.ndarray) -> float:
    """
    Extract recovery speed (post-action deceleration).

    Args:
        features: 101-dimensional feature vector

    Returns:
        Recovery speed (positive = good recovery)
    """
    acceleration = features[ACCELERATION_START:ACCELERATION_END]
    return float(np.linalg.norm(acceleration))
