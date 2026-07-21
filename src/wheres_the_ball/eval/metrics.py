"""Localization error metrics for Phase 0."""
from __future__ import annotations

import math
from statistics import median


def euclidean_norm(pred: tuple[float, float], gt: tuple[float, float]) -> float:
    """Euclidean distance in normalized image coordinates ([0,1] on each axis)."""
    return math.hypot(pred[0] - gt[0], pred[1] - gt[1])


def summarize(errors: list[float]) -> dict[str, float]:
    """Median + IQR (distribution has long tails; mean alone is misleading)."""
    if not errors:
        return {"n": 0, "median": float("nan"), "q1": float("nan"), "q3": float("nan")}
    s = sorted(errors)
    n = len(s)

    def q(p: float) -> float:
        idx = p * (n - 1)
        lo, hi = int(math.floor(idx)), int(math.ceil(idx))
        if lo == hi:
            return s[lo]
        return s[lo] + (s[hi] - s[lo]) * (idx - lo)

    return {"n": n, "median": median(s), "q1": q(0.25), "q3": q(0.75), "mean": sum(s) / n}
