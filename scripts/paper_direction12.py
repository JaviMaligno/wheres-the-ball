"""M5 — is the ball's DIRECTION of travel readable, at 12-match scale?

The direction-in-flight result was only cross-validated on 2 Metrica games. Here we scale
it to leave-one-match-out over the 12 cached soccer matches. Ball velocity is recovered
from consecutive (temporally ordered, strided) ball positions in the cache; "in flight" =
top-quartile ball speed per match. We train a GBM to predict the ball's direction unit
vector from player features, and report median angular error vs the 90-degree chance
baseline and a constant-mean-direction baseline.

Usage: uv run python scripts/paper_direction12.py
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from paper_deep_tuned import load_cached_matches

OUT = pathlib.Path("results/nivel3")


def ball_dir_speed(Y):
    """Temporal diff of ball positions → per-frame velocity (unit dir + speed)."""
    v = np.diff(Y, axis=0, prepend=Y[:1])
    v[0] = v[1] if len(v) > 1 else v[0]
    spd = np.linalg.norm(v, axis=1)
    d = v / (spd[:, None] + 1e-9)
    return d, spd


def ang(a, b):
    return np.degrees(np.arccos(np.clip((a * b).sum(1), -1, 1)))


def main() -> None:
    matches = load_cached_matches()
    names = list(matches)
    for n in names:
        d, spd = ball_dir_speed(matches[n]["Y"])
        matches[n]["bdir"] = d; matches[n]["bspd"] = spd
        matches[n]["flight_thr"] = np.quantile(spd, 0.75)

    learned, base_mean, within45 = [], [], []
    for held in names:
        tr = [n for n in names if n != held]
        Xtr = np.concatenate([matches[n]["X"] for n in tr])
        Dtr = np.concatenate([matches[n]["bdir"] for n in tr])
        mx = HistGradientBoostingRegressor(max_iter=300, random_state=0).fit(Xtr, Dtr[:, 0])
        my = HistGradientBoostingRegressor(max_iter=300, random_state=0).fit(Xtr, Dtr[:, 1])
        d = matches[held]
        fl = d["bspd"] >= d["flight_thr"]
        if fl.sum() < 10:
            continue
        pred = np.stack([mx.predict(d["X"][fl]), my.predict(d["X"][fl])], 1)
        pred /= np.linalg.norm(pred, axis=1, keepdims=True) + 1e-9
        gt = d["bdir"][fl]
        e = ang(pred, gt)
        learned.append(np.median(e))
        within45.append(float((e < 45).mean()))
        md = Dtr.mean(0); md /= np.linalg.norm(md) + 1e-9
        base_mean.append(np.median(ang(np.tile(md, (fl.sum(), 1)), gt)))
    learned, base_mean, within45 = np.array(learned), np.array(base_mean), np.array(within45)
    print(f"Direction in flight, LOMO across {len(learned)} matches (median angular error):")
    print(f"  learned from players : {learned.mean():.1f}°±{learned.std():.1f}")
    print(f"  constant-mean baseline: {base_mean.mean():.1f}°")
    print(f"  chance (random dir)   : 90.0°")
    print(f"  within-45° cone       : {within45.mean()*100:.0f}% ± {within45.std()*100:.0f}%")
    print(f"  beats chance in {int((learned < 90).sum())}/{len(learned)} matches")
    (OUT / "paper_direction12.json").write_text(json.dumps(
        {"learned_deg": float(learned.mean()), "learned_std": float(learned.std()),
         "base_mean_deg": float(base_mean.mean()), "within45": float(within45.mean()),
         "n": int(len(learned))}, indent=2))
    print(f"Saved {OUT/'paper_direction12.json'}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "scripts")
    main()
