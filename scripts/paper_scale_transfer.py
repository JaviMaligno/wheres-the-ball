"""Paper scale-up — cross-sport transfer asymmetry & the velocity-scale mechanism.

Re-establishes the Level-2 headline at scale: 12 soccer matches (Metrica + SkillCorner,
two leagues) vs 6 NBA SportVU games. Trains the geometric specialist on one sport
(pooled) and evaluates zero-shot on the other, for FULL features and POSITIONS-ONLY.

Claim to test at scale: the zero-shot asymmetry (basketball transfers to soccer, soccer
doesn't return the favour) lives in the velocity channel — in ratio over in-domain
error it should be large with full features and vanish with positions only.

Usage: uv run python scripts/paper_scale_transfer.py
"""
from __future__ import annotations

import glob
import json
import pathlib

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from wheres_the_ball.data.field_tracking import (load_metrica_trajectories,
                                                 load_skillcorner_trajectories,
                                                 load_sportvu_trajectories)
from wheres_the_ball.features.geometry import frame_features, FEATURE_NAMES

OUT = pathlib.Path("results/nivel3")
CUR = [16, 17, 18, 19, 20]
_VEL = {"vel_centroid", "fastest", "converge", "mean_speed", "max_speed"}
POS_COLS = [i for i, n in enumerate(FEATURE_NAMES)
            if n.replace("_x", "").replace("_y", "") not in _VEL]


def load_match(loader, path):
    fr = [(p[:, CUR], b) for p, b in loader(path)]
    if not fr:
        return None
    return (np.stack([frame_features(p) for p, _ in fr]), np.stack([b for _, b in fr]))


def gbm(X, Y):
    return (HistGradientBoostingRegressor(max_iter=300, random_state=0).fit(X, Y[:, 0]),
            HistGradientBoostingRegressor(max_iter=300, random_state=0).fit(X, Y[:, 1]))


def gerr(m, X, Y):
    return float(np.median(np.linalg.norm(np.stack([m[0].predict(X), m[1].predict(X)], 1) - Y, axis=1)))


def load_sport(specs):
    out = []
    for loader, path in specs:
        try:
            d = load_match(loader, path)
        except Exception:
            continue
        if d:
            out.append(d)
    return out


def in_domain(matches, cols):
    """Leave-one-match-out median error, averaged across held-out matches."""
    errs = []
    for i in range(len(matches)):
        Xtr = np.concatenate([matches[j][0][:, cols] for j in range(len(matches)) if j != i])
        Ytr = np.concatenate([matches[j][1] for j in range(len(matches)) if j != i])
        errs.append(gerr(gbm(Xtr, Ytr), matches[i][0][:, cols], matches[i][1]))
    return float(np.mean(errs))


def zero_shot(src, tgt, cols):
    Xtr = np.concatenate([m[0][:, cols] for m in src]); Ytr = np.concatenate([m[1] for m in src])
    Xte = np.concatenate([m[0][:, cols] for m in tgt]); Yte = np.concatenate([m[1] for m in tgt])
    return gerr(gbm(Xtr, Ytr), Xte, Yte)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    base = pathlib.Path("data/sample-data/data")
    soccer_specs = ([(load_metrica_trajectories, g) for g in sorted(glob.glob(str(base / "Sample_Game_*")))]
                    + [(load_skillcorner_trajectories, m) for m in sorted(glob.glob("data/opendata/data/matches/*"))])
    basket_specs = [(load_sportvu_trajectories, glob.glob(f"{d}/*.json")[0])
                    for d in sorted(glob.glob("data/sportvu/*")) if glob.glob(f"{d}/*.json")]
    print("Loading soccer…"); soccer = load_sport(soccer_specs)
    print("Loading basketball…"); basket = load_sport(basket_specs)
    print(f"soccer {len(soccer)} matches / {sum(len(m[1]) for m in soccer)} frames · "
          f"basket {len(basket)} games / {sum(len(m[1]) for m in basket)} frames")

    res = {}
    for tag, cols in [("full", list(range(len(FEATURE_NAMES)))), ("positions-only", POS_COLS)]:
        ids = in_domain(soccer, cols); idb = in_domain(basket, cols)
        s2b = zero_shot(soccer, basket, cols); b2s = zero_shot(basket, soccer, cols)
        # ratio over in-domain difficulty of the *target* sport
        r_s2b, r_b2s = s2b / idb, b2s / ids
        print(f"\n=== {tag} ===")
        print(f"  in-domain: soccer {ids:.4f}  basket {idb:.4f}")
        print(f"  zero-shot: soccer→basket {s2b:.4f} ({r_s2b:.2f}× in-domain)  "
              f"basket→soccer {b2s:.4f} ({r_b2s:.2f}× in-domain)")
        print(f"  asymmetry in ratio (s→b minus b→s): {r_s2b - r_b2s:+.2f}")
        res[tag] = {"id_soccer": ids, "id_basket": idb, "s2b": s2b, "b2s": b2s,
                    "ratio_s2b": r_s2b, "ratio_b2s": r_b2s, "asym_ratio": r_s2b - r_b2s}

    print(f"\nAsymmetry (ratio) — full: {res['full']['asym_ratio']:+.2f}   "
          f"positions-only: {res['positions-only']['asym_ratio']:+.2f}")
    print("→ if it shrinks/vanishes without velocity, the asymmetry lives in the velocity channel.")
    (OUT / "paper_scale_transfer.json").write_text(json.dumps(res, indent=2))
    print(f"Saved {OUT/'paper_scale_transfer.json'}")


if __name__ == "__main__":
    main()
