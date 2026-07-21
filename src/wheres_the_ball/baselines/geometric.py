"""Trivial geometric baselines (no learning).

Phase 0 has ground truth for the ball only (not players), so the applicable baseline is
CENTER_OF_FRAME, which captures the broadcast-camera bias (TV keeps the ball near the
middle of the shot). The player-based baselines (centroid, fastest player, Voronoi) need
player tracking and land with the formal dataset.
"""
from __future__ import annotations


def center_of_frame() -> tuple[float, float]:
    """Predict the ball at the image center — the bias any real system must beat."""
    return 0.5, 0.5
