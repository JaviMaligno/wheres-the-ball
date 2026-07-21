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


def centroid(players: list[dict]) -> tuple[float, float] | None:
    """B1 — centroid of the players' normalized positions."""
    if not players:
        return None
    return (sum(p["x"] for p in players) / len(players),
            sum(p["y"] for p in players) / len(players))


def nearest_player(point: tuple[float, float], players: list[dict]) -> int | None:
    """Index of the player closest to `point` (normalized coords). None if empty."""
    if not players:
        return None
    import math
    return min(range(len(players)),
              key=lambda i: math.hypot(players[i]["x"] - point[0], players[i]["y"] - point[1]))
