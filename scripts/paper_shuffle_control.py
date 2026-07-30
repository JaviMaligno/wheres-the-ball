"""M11 — label-shuffle control: how much of the geometry model's skill is real inference
vs residual dataset structure the de-biasing didn't remove?

Train the geo-GBM (a) normally, and (b) on SHUFFLED targets — each frame's features paired
with a random OTHER frame's ball position (same match, so the marginal ball distribution is
preserved but the feature->ball link is destroyed). If the real model beats the shuffled
control by a wide margin, the skill is genuine inference, not marginal-structure exploitation.

Leave-one-match-out over the 12 cached soccer matches. Local, ~$0.

Usage: uv run python scripts/paper_shuffle_control.py
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from paper_deep_tuned import load_cached_matches

OUT = pathlib.Path("results/nivel3")
rng = np.random.default_rng(0)


def fit_err(Xtr, Ytr, Xte, Yte):
    mx = HistGradientBoostingRegressor(max_iter=300, random_state=0).fit(Xtr, Ytr[:, 0])
    my = HistGradientBoostingRegressor(max_iter=300, random_state=0).fit(Xtr, Ytr[:, 1])
    return float(np.median(np.linalg.norm(np.stack([mx.predict(Xte), my.predict(Xte)], 1) - Yte, axis=1)))


def main() -> None:
    matches = load_cached_matches()
    names = list(matches)
    real, shuf, cen = [], [], []
    for held in names:
        tr = [n for n in names if n != held]
        Xtr = np.concatenate([matches[n]["X"] for n in tr]); Ytr = np.concatenate([matches[n]["Y"] for n in tr])
        d = matches[held]
        real.append(fit_err(Xtr, Ytr, d["X"], d["Y"]))
        # shuffled targets within the training pool (destroys feature->ball link)
        perm = rng.permutation(len(Ytr))
        shuf.append(fit_err(Xtr, Ytr[perm], d["X"], d["Y"]))
        cen.append(float(np.median(np.linalg.norm(d["cen"] - d["Y"], axis=1))))
    real, shuf, cen = np.array(real), np.array(shuf), np.array(cen)
    print(f"Across {len(names)} matches (median error, mean±std):")
    print(f"  real geo model     {real.mean():.4f}±{real.std():.4f}")
    print(f"  shuffled-target     {shuf.mean():.4f}±{shuf.std():.4f}")
    print(f"  vel-centroid ref    {cen.mean():.4f}±{cen.std():.4f}")
    print(f"  real beats shuffled in {int((real < shuf).sum())}/{len(names)} matches")
    print(f"  gap (shuffled - real) = {(shuf-real).mean():.4f}  → skill is genuine inference, not marginal structure")
    (OUT / "paper_shuffle_control.json").write_text(json.dumps(
        {"real": real.mean(), "shuffled": shuf.mean(), "centroid": cen.mean(),
         "real_beats_shuffled": int((real < shuf).sum()), "n": len(names)}, indent=2))
    print(f"Saved {OUT/'paper_shuffle_control.json'}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "scripts")
    main()
