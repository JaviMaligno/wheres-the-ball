"""Interpretable geometric features for Level 3 ("is ball inference geometry?").

Input: one frame as players[N, 5] = (x, y, vx, vy, team±1) in field coords [0,1]
(the Level-2 sample format). Output: a flat feature vector of interpretable
geometric quantities, each with a name — the point is that a human can read them.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import Delaunay

SPEED_MIN = 0.02  # field-fractions/s below which a player is "not moving"


def _convergence_point(pos, vel):
    """Least-squares point closest to all motion rays (players 'point' at the ball).

    Solve sum_i (I - d_i d_i^T)(p - x_i) = 0 for p, using only moving players.
    Falls back to the centroid when the system is degenerate.
    """
    speed = np.linalg.norm(vel, axis=1)
    mask = speed > SPEED_MIN
    if mask.sum() < 3:
        return pos.mean(axis=0)
    d = vel[mask] / np.linalg.norm(vel[mask], axis=1, keepdims=True)
    x = pos[mask]
    A = np.zeros((2, 2)); b = np.zeros(2)
    for di, xi in zip(d, x):
        P = np.eye(2) - np.outer(di, di)
        A += P; b += P @ xi
    try:
        p = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return pos.mean(axis=0)
    return np.clip(p, -0.2, 1.2)


def _team_contact(pos, team):
    """Midpoint of the shortest Delaunay edge joining opposing teams."""
    if len(pos) < 4 or len(set(team)) < 2:
        return pos.mean(axis=0)
    try:
        tri = Delaunay(pos)
    except Exception:
        return pos.mean(axis=0)
    best, best_d = None, np.inf
    for simplex in tri.simplices:
        for a in range(3):
            i, j = simplex[a], simplex[(a + 1) % 3]
            if team[i] != team[j]:
                dd = np.linalg.norm(pos[i] - pos[j])
                if dd < best_d:
                    best_d, best = dd, (pos[i] + pos[j]) / 2
    return best if best is not None else pos.mean(axis=0)


FEATURE_NAMES = [
    "centroid_x", "centroid_y",
    "vel_centroid_x", "vel_centroid_y",
    "fastest_x", "fastest_y",
    "densest_x", "densest_y",
    "converge_x", "converge_y",
    "team_contact_x", "team_contact_y",
    "home_centroid_x", "home_centroid_y",
    "away_centroid_x", "away_centroid_y",
    "spread_x", "spread_y",
    "mean_speed", "max_speed",
]


def frame_features(players: np.ndarray) -> np.ndarray:
    pos, vel, team = players[:, :2], players[:, 2:4], players[:, 4]
    speed = np.linalg.norm(vel, axis=1)
    w = speed + 1e-6
    vel_centroid = (pos * w[:, None]).sum(0) / w.sum()
    dens = [(np.linalg.norm(pos - p, axis=1) < 0.1).sum() for p in pos]
    home = pos[team > 0] if (team > 0).any() else pos
    away = pos[team < 0] if (team < 0).any() else pos
    feats = np.concatenate([
        pos.mean(0),
        vel_centroid,
        pos[int(np.argmax(speed))],
        pos[int(np.argmax(dens))],
        _convergence_point(pos, vel),
        _team_contact(pos, team),
        home.mean(0), away.mean(0),
        pos.std(0),
        [speed.mean(), speed.max()],
    ])
    return feats.astype(np.float32)
