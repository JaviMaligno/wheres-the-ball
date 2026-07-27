"""Paper scale-up — the stillness blind spot across 12 soccer matches.

Level-3.5 was cross-validated only g1<->g2. Here: pool Metrica + 10 SkillCorner, split
each match's frames into STILL vs MOVING by ball speed (finite-difference between
consecutive sampled frames, a proxy that cleanly separates dead balls from live ones),
and run the velocity ablation (full vs positions-only geo-GBM) leave-one-match-out.

Claim: the still ball is harder, and velocity's contribution collapses when the ball is
still — should hold across matches, not just the two sample games.

Usage: uv run python scripts/paper_scale_stillness.py
"""
from __future__ import annotations

import glob
import json
import pathlib

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from wheres_the_ball.data.field_tracking import (load_metrica_trajectories,
                                                 load_skillcorner_trajectories)
from wheres_the_ball.features.geometry import frame_features, FEATURE_NAMES

OUT = pathlib.Path("results/nivel3")
CUR = [16, 17, 18, 19, 20]
_VEL = {"vel_centroid", "fastest", "converge", "mean_speed", "max_speed"}
POS_COLS = [i for i, n in enumerate(FEATURE_NAMES)
            if n.replace("_x", "").replace("_y", "") not in _VEL]


def load_match(loader, path):
    traj = list(loader(path))
    if not traj:
        return None
    fr = [(p[:, CUR], b) for p, b in traj]
    X = np.stack([frame_features(p) for p, _ in fr]); Y = np.stack([b for _, b in fr])
    # ball speed proxy: |Δ ball| between consecutive sampled frames (normalized units/step)
    balls = Y
    d = np.linalg.norm(np.diff(balls, axis=0, prepend=balls[:1]), axis=1)
    d[0] = d[1] if len(d) > 1 else 0.0
    return {"X": X, "Y": Y, "spd": d}


def gbm(X, Y):
    return (HistGradientBoostingRegressor(max_iter=300, random_state=0).fit(X, Y[:, 0]),
            HistGradientBoostingRegressor(max_iter=300, random_state=0).fit(X, Y[:, 1]))


def gerrs(m, X, Y):
    return np.linalg.norm(np.stack([m[0].predict(X), m[1].predict(X)], 1) - Y, axis=1)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    base = pathlib.Path("data/sample-data/data")
    matches = {}
    print("Loading 12 soccer matches…")
    for g in sorted(glob.glob(str(base / "Sample_Game_*"))):
        try:
            d = load_match(load_metrica_trajectories, g)
        except Exception:
            continue
        if d:
            matches["metrica/" + pathlib.Path(g).name.replace("Sample_Game_", "g")] = d
    for m in sorted(glob.glob("data/opendata/data/matches/*")):
        try:
            d = load_match(load_skillcorner_trajectories, m)
        except Exception:
            continue
        if d:
            matches["skillcorner/" + pathlib.Path(m).name] = d
    names = list(matches)
    # a per-match "still" threshold: bottom quartile of that match's ball-speed proxy
    for n in names:
        matches[n]["still_thr"] = np.quantile(matches[n]["spd"], 0.25)
    print(f"{len(names)} matches")

    rows = []
    print(f"\n{'held-out':26}{'still full':>11}{'still pos':>11}{'move full':>11}{'move pos':>10}")
    for held in names:
        tr = [n for n in names if n != held]
        Xtr = np.concatenate([matches[n]["X"] for n in tr]); Ytr = np.concatenate([matches[n]["Y"] for n in tr])
        mf = gbm(Xtr, Ytr); mp = gbm(Xtr[:, POS_COLS], Ytr)
        d = matches[held]
        ef = gerrs(mf, d["X"], d["Y"]); ep = gerrs(mp, d["X"][:, POS_COLS], d["Y"])
        still = d["spd"] <= d["still_thr"]
        r = {"match": held,
             "still_full": float(np.median(ef[still])), "still_pos": float(np.median(ep[still])),
             "move_full": float(np.median(ef[~still])), "move_pos": float(np.median(ep[~still]))}
        rows.append(r)
        print(f"{held:26}{r['still_full']:>11.4f}{r['still_pos']:>11.4f}{r['move_full']:>11.4f}{r['move_pos']:>10.4f}")

    sf = np.array([r["still_full"] for r in rows]); sp = np.array([r["still_pos"] for r in rows])
    mf_ = np.array([r["move_full"] for r in rows]); mp_ = np.array([r["move_pos"] for r in rows])
    print(f"\nAcross {len(rows)} matches (median ± std):")
    print(f"  STILL  full {sf.mean():.4f}±{sf.std():.4f}   pos-only {sp.mean():.4f}   velocity helps {(sp-sf).mean():+.4f}")
    print(f"  MOVING full {mf_.mean():.4f}±{mf_.std():.4f}   pos-only {mp_.mean():.4f}   velocity helps {(mp_-mf_).mean():+.4f}")
    print(f"  still harder than moving in {int((sf>mf_).sum())}/{len(rows)} matches")
    print(f"  velocity helps LESS when still in {int(((sp-sf)<(mp_-mf_)).sum())}/{len(rows)} matches")
    (OUT / "paper_scale_stillness.json").write_text(json.dumps(rows, indent=2))
    print(f"Saved {OUT/'paper_scale_stillness.json'}")


if __name__ == "__main__":
    main()
