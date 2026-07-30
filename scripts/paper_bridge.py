"""B3 — the VLM<->specialist bridge on ONE shared set (the n=260 image benchmark).

The paper's Fig 1 juxtaposed pixel-space VLMs against a field-coordinate specialist that
never competed on the same items. Here we train a specialist on the SAME 260 items and the
SAME input the task exposes (player image-space positions; no velocity is available from a
single broadcast frame), via k-fold cross-validation, and score it in the far bin exactly
like the VLMs (win-rate vs camera center + median error). This makes the head-to-head real.

Positions-only geometric features (velocity features are undefined single-frame in image
space): centroid, per-team centroids, densest cluster, team spread, closest opposing pair.

Usage: uv run python scripts/paper_bridge.py
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
from scipy.spatial import Delaunay
from sklearn.ensemble import HistGradientBoostingRegressor

OUT = pathlib.Path("results/fase1")
rng = np.random.default_rng(0)


def pos_features(players):
    pos = np.array([[p["x"], p["y"]] for p in players], float)
    team = np.array([1.0 if p.get("team") == "left" else -1.0 for p in players])
    home = pos[team > 0] if (team > 0).any() else pos
    away = pos[team < 0] if (team < 0).any() else pos
    dens = [(np.linalg.norm(pos - q, axis=1) < 0.1).sum() for q in pos]
    # closest opposing pair midpoint (Delaunay)
    contact = pos.mean(0)
    if len(pos) >= 4 and (team > 0).any() and (team < 0).any():
        try:
            tri = Delaunay(pos); best = np.inf
            for s in tri.simplices:
                for a in range(3):
                    i, j = s[a], s[(a + 1) % 3]
                    if team[i] != team[j]:
                        d = np.linalg.norm(pos[i] - pos[j])
                        if d < best:
                            best, contact = d, (pos[i] + pos[j]) / 2
        except Exception:
            pass
    return np.concatenate([pos.mean(0), home.mean(0), away.mean(0), pos.std(0),
                           pos[int(np.argmax(dens))], contact])


def main() -> None:
    man = json.loads((OUT / "manifest.json").read_text())["items"]
    pred = json.loads((OUT / "predictions.json").read_text())
    X = np.stack([pos_features(it["players"]) for it in man])
    Y = np.stack([[it["gt"]["x"], it["gt"]["y"]] for it in man])
    far = np.array([it["center_bin"] == "far" for it in man])
    ids = [it["id"] for it in man]

    # k-fold CV specialist (positions-only), out-of-fold predictions
    k = 5
    fold = rng.integers(0, k, len(man))
    oof = np.zeros_like(Y)
    for f in range(k):
        tr, te = fold != f, fold == f
        mx = HistGradientBoostingRegressor(max_iter=300, random_state=0).fit(X[tr], Y[tr, 0])
        my = HistGradientBoostingRegressor(max_iter=300, random_state=0).fit(X[tr], Y[tr, 1])
        oof[te, 0] = mx.predict(X[te]); oof[te, 1] = my.predict(X[te])

    def winrate(pred_xy, mask):
        gt = Y[mask]; pp = pred_xy[mask]; cen = np.array([0.5, 0.5])
        w = np.linalg.norm(pp - gt, axis=1) < np.linalg.norm(cen - gt, axis=1)
        return float(w.mean()), float(np.median(np.linalg.norm(pp - gt, axis=1)))

    def boot_ci(pred_xy, mask, n=5000):
        gt = Y[mask]; pp = pred_xy[mask]; cen = np.array([0.5, 0.5])
        w = (np.linalg.norm(pp - gt, axis=1) < np.linalg.norm(cen - gt, axis=1)).astype(float)
        s = [w[rng.integers(0, len(w), len(w))].mean() for _ in range(n)]
        return float(np.percentile(s, 2.5)) * 100, float(np.percentile(s, 97.5)) * 100

    print(f"n far = {int(far.sum())}")
    wr, med = winrate(oof, far); lo, hi = boot_ci(oof, far)
    print(f"\nSpecialist (positions-only, 5-fold CV) FAR: win {wr*100:.0f}% [{lo:.0f},{hi:.0f}] med {med:.3f}")

    # VLMs on the same far items, for side-by-side
    for key, name in [("gpt", "GPT-5.4"), ("llama4", "Llama-4"), ("claude_opus", "Opus 4.8"), ("claude", "Sonnet 4.6")]:
        pv = []; msk = []
        for i, it in enumerate(man):
            r = pred.get(it["id"], {}).get(key)
            if far[i] and isinstance(r, dict) and "x" in r:
                pv.append([r["x"], r["y"]]); msk.append(i)
        if len(pv) < 5:
            print(f"{name:9} FAR: n={len(pv)} (skip)"); continue
        pv = np.array(pv); gt = Y[msk]; cen = np.array([0.5, 0.5])
        w = (np.linalg.norm(pv - gt, axis=1) < np.linalg.norm(cen - gt, axis=1)).astype(float)
        print(f"{name:9} FAR: win {w.mean()*100:.0f}% med {np.median(np.linalg.norm(pv-gt,axis=1)):.3f} (n={len(pv)})")

    # plain centroid baseline (image space) for reference
    cwr, cmed = winrate(np.stack([X[:, 0], X[:, 1]], 1), far)
    print(f"\nRef: image-space centroid FAR win {cwr*100:.0f}% med {cmed:.3f}")
    res = {"specialist_far_win": wr, "specialist_far_win_ci": [lo, hi], "specialist_far_med": med,
           "n_far": int(far.sum())}
    (OUT / "paper_bridge.json").write_text(json.dumps(res, indent=2))
    print(f"Saved {OUT/'paper_bridge.json'}")


if __name__ == "__main__":
    main()
