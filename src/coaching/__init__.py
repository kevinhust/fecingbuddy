"""
FencerAI Coaching Module
=======================
Version: 2.0 | Last Updated: 2026-04-02

Real-time coaching engine for fencing analysis.
Computes metrics from 101-dim features and generates actionable alerts.

Example:
    from src.coaching import CoachingEngine, CoachingMetrics

    engine = CoachingEngine()
    metrics = engine.compute_metrics(features_2x101, frame_data)
    alerts = engine.evaluate(features_2x101, metrics)
"""

from src.coaching.coaching_metrics import CoachingMetrics
from src.coaching.coaching_engine import CoachingEngine

__all__ = ["CoachingMetrics", "CoachingEngine"]
