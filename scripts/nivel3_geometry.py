"""Level-3 Fase 0 — do ~10 interpretable geometric features recover the deep model?

Gradient boosting on the interpretable feature vector vs the Level-2 DeepSets, on
the same splits (soccer: train Metrica g1, eval g2; basketball: train pool games,
eval the 2 eval games). Reports gap recovery:

    recovery = (err_centroid - err_geo) / (err_centroid - err_deep)

plus per-feature importance (which geometry carries it) and the cross-sport zero-shot
of the geometric model (does interpretable geometry transfer like basketball did?).

Usage: uv run python scripts/nivel3_geometry.py
"""
from __future__ import annotations

import glob
import json
import pathlib

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from wheres_the_ball.data.field_tracking import load_metrica_game, load_sportvu_game
from wheres_the_ball.features.geometry import frame_features, FEATURE_NAMES

OUT = pathlib.Path("results/nivel3")


def build(samples):
    X = np.stack([frame_features(p) for p, _ in samples])
    Y = np.stack([b for _, b in samples])
    return X, Y


def fit(X, Y):
    mx = HistGradientBoostingRegressor(max_iter=300, random_state=0).fit(X, Y[:, 0])
    my = HistGradientBoostingRegressor(max_iter=300, random_state=0).fit(X, Y[:, 1])
    return mx, my


def med_err(models, X, Y):
    mx, my = models
    pred = np.stack([mx.predict(X), my.predict(X)], axis=1)
    return float(np.median(np.linalg.norm(pred - Y, axis=1)))


def perm_importance(models, X, Y, k=5):
    """Permutation importance on the paired feature columns (x,y together)."""
    base = med_err(models, X, Y)
    rng = np.random.default_rng(0)
    out = {}
    pairs = [(i, i + 1) if FEATURE_NAMES[i].endswith("_x") else (i,)
             for i in range(len(FEATURE_NAMES))
             if not FEATURE_NAMES[i].endswith("_y")]
    for cols in pairs:
        deltas = []
        for _ in range(k):
            Xp = X.copy()
            idx = rng.permutation(len(X))
            for c in cols:
                Xp[:, c] = X[idx, c]
            deltas.append(med_err(models, Xp, Y) - base)
        name = FEATURE_NAMES[cols[0]].replace("_x", "")
        out[name] = float(np.mean(deltas))
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    base = pathlib.Path("data/sample-data/data")
    print("Loading + featurizing…")
    Xs_tr, Ys_tr = build(list(load_metrica_game(base / "Sample_Game_1")))
    Xs_te, Ys_te = build(list(load_metrica_game(base / "Sample_Game_2")))
    eval_games = [glob.glob("data/sportvu/game1/*.json")[0],
                  glob.glob("data/sportvu/01.02.2016.BKN.at.BOS/*.json")[0]]
    pool_games = [glob.glob(f"data/sportvu/{d}/*.json")[0] for d in
                  ("01.01.2016.DAL.at.MIA", "01.01.2016.NYK.at.CHI",
                   "01.02.2016.MIL.at.MIN", "01.02.2016.PHX.at.SAC")]
    Xb_tr, Yb_tr = build([s for g in pool_games for s in load_sportvu_game(g)])
    Xb_te, Yb_te = build([s for g in eval_games for s in load_sportvu_game(g)])
    print(f"soccer train/test: {len(Xs_tr)}/{len(Xs_te)}  basket train/test: {len(Xb_tr)}/{len(Xb_te)}")

    res = {}
    soccer = fit(Xs_tr, Ys_tr)
    basket = fit(Xb_tr, Yb_tr)

    e_s = med_err(soccer, Xs_te, Ys_te)
    e_b = med_err(basket, Xb_te, Yb_te)
    # references from Level 2 (same splits)
    refs = {"soccer": {"centroid": 0.231, "vel_centroid": 0.204, "deep": 0.101},
            "basket": {"centroid": 0.227, "vel_centroid": 0.216, "deep": 0.158}}
    for name, e, (Xte, Yte), model in [("soccer", e_s, (Xs_te, Ys_te), soccer),
                                       ("basket", e_b, (Xb_te, Yb_te), basket)]:
        r = refs[name]
        rec = (r["centroid"] - e) / (r["centroid"] - r["deep"])
        print(f"\n{name}: geo-GBM={e:.4f}  (centroide {r['centroid']} · deep {r['deep']})"
              f"  → recupera {rec:.0%} del gap deep-sobre-centroide")
        res[name] = {"geo": e, "recovery": rec}

    # cross-sport zero-shot of the interpretable model
    zs_sb = med_err(soccer, Xb_te, Yb_te)
    zs_bs = med_err(basket, Xs_te, Ys_te)
    print(f"\nzero-shot geo-GBM: fútbol→basket {zs_sb:.4f}  ·  basket→fútbol {zs_bs:.4f}")
    print("(deep era: 0.347 y 0.174-0.177; centroide sin entrenar: 0.227 / 0.231)")
    res["zero_shot"] = {"soccer_to_basket": zs_sb, "basket_to_soccer": zs_bs}

    print("\nImportancia por permutación (fútbol, Δ error mediano):")
    imp = perm_importance(soccer, Xs_te, Ys_te)
    for k, v in imp.items():
        print(f"  {k:16} {v:+.4f}")
    res["importance_soccer"] = imp

    (OUT / "geometry.json").write_text(json.dumps(res, indent=2))
    print(f"\nSaved {OUT/'geometry.json'}")


if __name__ == "__main__":
    main()
