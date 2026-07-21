"""Classify the ball state at a frame, to stratify the evaluation set.

Derived from the ball trajectory + player positions (SoccerNet-Tracking GT).
Thresholds calibrated on the test split: ball speed px/frame p25=1, p50=4, p75=10,
p90=21 (25 fps, 1920px wide).

States (mirrors the Level-1 design):
  - possession   : ball slow, essentially at a player's feet
  - short_pass   : ball moving at moderate speed
  - long_pass    : ball fast (pass/clearance/shot in flight)
  - contested    : ball slow but surrounded by players of both teams (disputa)
"""
from __future__ import annotations

import math

from .soccernet_tracking import Clip

SPEED_SHORT = 2.0   # px/frame; below this the ball is basically static
SPEED_LONG = 12.0   # px/frame; above this the ball is in flight
CONGESTION_R = 100.0  # px radius around the ball to count nearby players
CONGESTION_MIN = 4    # players within R to consider it a disputa


def ball_speed(clip: Clip, frame: int) -> float | None:
    """Average px/frame speed around `frame` (needs neighbours with the ball)."""
    b = clip.ball_at(frame)
    if b is None:
        return None
    deltas = []
    for g in (frame - 1, frame + 1):
        n = clip.ball_at(g)
        if n:
            deltas.append(math.hypot(n[0] - b[0], n[1] - b[1]))
    if not deltas:
        return None
    return sum(deltas) / len(deltas)


def _congestion(clip: Clip, frame: int) -> tuple[int, int]:
    """(#players within R of the ball, #distinct teams among them)."""
    b = clip.ball_at(frame)
    if b is None:
        return 0, 0
    near_teams = set()
    n = 0
    for cx, cy, ent in clip.players_at(frame):
        if math.hypot(cx - b[0], cy - b[1]) <= CONGESTION_R:
            n += 1
            if ent.team:
                near_teams.add(ent.team)
    return n, len(near_teams)


def classify(clip: Clip, frame: int) -> str | None:
    """Return the ball state, or None if the ball is absent at `frame`."""
    speed = ball_speed(clip, frame)
    if speed is None:
        return None
    n_near, n_teams = _congestion(clip, frame)
    if speed < SPEED_LONG and n_near >= CONGESTION_MIN and n_teams >= 2:
        return "contested"
    if speed >= SPEED_LONG:
        return "long_pass"
    if speed >= SPEED_SHORT:
        return "short_pass"
    return "possession"
