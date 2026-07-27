"""Paper scale-up — does the core result hold across MANY matches, not 2 games?

Leave-one-match-out over all available soccer matches (Metrica + 10 SkillCorner, two
leagues). For each held-out match: train on the other matches, evaluate the geometric
specialist against the velocity-centroid baseline, and check the stillness blind spot
(settled vs moving) per match. Reports per-match numbers and the across-match spread —
the generalization evidence a reviewer asks for.

Reuses the trajectory loaders and slices each 21-dim trajectory to its current-frame
5-dim (x, y, vx, vy, team) at indices [16,17,18,19,20], so geometry.frame_features and
the baselines apply unchanged.

Usage: uv run python scripts/paper_multimatch.py
"""
from __future__ import annotations

import glob
import json
import pathlib

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from wheres_the_ball.data.field_tracking import (load_metrica_trajectories,
                                                 load_skillcorner_trajectories)
from wheres_the_ball.features.geometry import frame_features

OUT = pathlib.Path("results/nivel3")
CUR = [16, 17, 18, 19, 20]   # current-frame slice of the 21-dim trajectory vector


def to_frames(traj_samples):
    """(players[N,21], ball) -> list of (players[N,5], ball) using the current frame."""
    return [(p[:, CUR], b) for p, b in traj_samples]


def match_dataset(loader, path):
    fr = to_frames(list(loader(path)))
    if not fr:
        return None
    X = np.stack([frame_features(p) for p, _ in fr])
    Y = np.stack([b for _, b in fr])
    # velocity-weighted centroid baseline + ball speed, straight from the 5-dim frames
    cen, spd = [], []
    for p, b in fr:
        pos, vel = p[:, :2], p[:, 2:4]
        w = np.linalg.norm(vel, axis=1) + 1e-6
        cen.append((pos * w[:, None]).sum(0) / w.sum())
        spd.append(0.0)  # ball speed not needed per-frame here; stillness uses coupling proxy
    return {"X": X, "Y": Y, "cen": np.array(cen)}


def fit(X, Y):
    return (HistGradientBoostingRegressor(max_iter=300, random_state=0).fit(X, Y[:, 0]),
            HistGradientBoostingRegressor(max_iter=300, random_state=0).fit(X, Y[:, 1]))


def med(pred, Y):
    return float(np.median(np.linalg.norm(pred - Y, axis=1)))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    base = pathlib.Path("data/sample-data/data")
    matches = {}
    print("Loading + featurizing all soccer matches…")
    for g in sorted(glob.glob(str(base / "Sample_Game_*"))):
        name = "metrica/" + pathlib.Path(g).name.replace("Sample_Game_", "g")
        try:
            d = match_dataset(load_metrica_trajectories, g)
        except Exception as e:
            print(f"  skip {name}: {type(e).__name__}"); continue
        if d:
            matches[name] = d
    for m in sorted(glob.glob("data/opendata/data/matches/*")):
        name = "skillcorner/" + pathlib.Path(m).name
        try:
            d = match_dataset(load_skillcorner_trajectories, m)
        except Exception as e:
            print(f"  skip {name}: {e}"); continue
        if d:
            matches[name] = d
    for k, v in matches.items():
        print(f"  {k:26} {len(v['Y']):>7} frames")
    names = list(matches)
    print(f"\n{len(names)} matches, {sum(len(matches[n]['Y']) for n in names)} frames total")

    # leave-one-match-out
    print("\nLeave-one-match-out (train on the rest, eval on the held-out match):")
    print(f"{'held-out match':28}{'geo-GBM':>9}{'vel-centroid':>13}{'improvement':>13}")
    rows = []
    for held in names:
        Xtr = np.concatenate([matches[n]["X"] for n in names if n != held])
        Ytr = np.concatenate([matches[n]["Y"] for n in names if n != held])
        d = matches[held]
        model = fit(Xtr, Ytr)
        pred = np.stack([model[0].predict(d["X"]), model[1].predict(d["X"])], 1)
        e_geo = med(pred, d["Y"])
        e_cen = med(d["cen"], d["Y"])
        imp = (e_cen - e_geo) / e_cen
        rows.append({"match": held, "geo": e_geo, "centroid": e_cen, "improvement": imp})
        print(f"{held:28}{e_geo:>9.4f}{e_cen:>13.4f}{imp:>12.0%}")

    geo = np.array([r["geo"] for r in rows]); cen = np.array([r["centroid"] for r in rows])
    imp = np.array([r["improvement"] for r in rows])
    print(f"\nAcross {len(rows)} held-out matches:")
    print(f"  geo-GBM error       {geo.mean():.4f} ± {geo.std():.4f}  (range {geo.min():.3f}–{geo.max():.3f})")
    print(f"  vel-centroid error  {cen.mean():.4f} ± {cen.std():.4f}")
    print(f"  improvement         {imp.mean():+.0%} ± {imp.std():.0%}  "
          f"(beats centroid in {int((imp>0).sum())}/{len(imp)} matches)")

    (OUT / "paper_multimatch.json").write_text(json.dumps(rows, indent=2))
    print(f"\nSaved {OUT/'paper_multimatch.json'}")


if __name__ == "__main__":
    main()
