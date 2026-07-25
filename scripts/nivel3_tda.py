"""Level-3 topological strand — does persistent homology beat interpretable geometry?

The rule (set upfront): topological features only count if they BEAT their geometric
counterpart. The geometric GBM recovered 92% of the deep-over-centroid gap in soccer
(0.1115). So TDA has to clear that bar, or it's topology theater.

Per frame, from the player point cloud, we build topological features:
  - Persistent homology (Vietoris-Rips, ripser): H0/H1 persistence statistics
    (loop count, max/total persistence, persistence entropy) — the *shape* of the
    configuration at every scale.
  - Location-aware topology (the ball-relevant part): centre of the most persistent
    H1 loop (via its representative cocycle — "the ball sits in the hole"), and the
    two largest empty circumcircles of the Delaunay triangulation ("the ball sits in
    the biggest gap"), each with centre + radius.

Then the same HistGradientBoostingRegressor and the same Metrica g1→g2 / SportVU splits
as nivel3_geometry.py. We report: geo-only (reproduced), tda-only, geo+tda, and the
permutation importance of the TDA block inside geo+tda (does it add anything at all?).

Usage: uv run python scripts/nivel3_tda.py
"""
from __future__ import annotations

import glob
import json
import pathlib

import numpy as np
from ripser import ripser
from scipy.spatial import Delaunay
from sklearn.ensemble import HistGradientBoostingRegressor

from wheres_the_ball.data.field_tracking import load_metrica_game, load_sportvu_game
from wheres_the_ball.features.geometry import frame_features

OUT = pathlib.Path("results/nivel3")


def _circumcircle(a, b, c):
    ax, ay = a; bx, by = b; cx, cy = c
    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-12:
        return None
    ux = ((ax**2 + ay**2) * (by - cy) + (bx**2 + by**2) * (cy - ay) + (cx**2 + cy**2) * (ay - by)) / d
    uy = ((ax**2 + ay**2) * (cx - bx) + (bx**2 + by**2) * (ax - cx) + (cx**2 + cy**2) * (bx - ax)) / d
    ctr = np.array([ux, uy])
    return ctr, float(np.linalg.norm(ctr - a))


def _empty_circles(pos, k=2):
    """Top-k largest circumcircles of the Delaunay triangulation (centre_x, centre_y, r)."""
    out = []
    if len(pos) >= 4:
        try:
            tri = Delaunay(pos)
            circ = []
            for s in tri.simplices:
                cc = _circumcircle(pos[s[0]], pos[s[1]], pos[s[2]])
                if cc is not None:
                    circ.append((cc[1], cc[0]))
            circ.sort(key=lambda t: -t[0])
            for r, ctr in circ[:k]:
                out += [float(ctr[0]), float(ctr[1]), float(r)]
        except Exception:
            pass
    while len(out) < 3 * k:  # pad with centroid / zero radius
        c = pos.mean(0)
        out += [float(c[0]), float(c[1]), 0.0]
    return out


def _persistence_entropy(pers):
    pers = pers[pers > 0]
    if len(pers) == 0:
        return 0.0
    p = pers / pers.sum()
    return float(-(p * np.log(p)).sum())


def tda_features(players: np.ndarray) -> np.ndarray:
    pos = players[:, :2].astype(np.float64)
    res = ripser(pos, maxdim=1, do_cocycles=True)
    h0, h1 = res["dgms"][0], res["dgms"][1]
    d0 = h0[:, 1]; d0 = d0[np.isfinite(d0)]
    p1 = (h1[:, 1] - h1[:, 0]) if len(h1) else np.array([])

    # global persistence stats
    stats = [
        float(d0.max()) if len(d0) else 0.0,      # largest H0 death (widest gap to merge)
        float(d0.sum()) if len(d0) else 0.0,
        float(d0.mean()) if len(d0) else 0.0,
        _persistence_entropy(d0),
        float(len(p1)),                            # number of loops
        float(p1.max()) if len(p1) else 0.0,       # most persistent loop
        float(p1.sum()) if len(p1) else 0.0,
        _persistence_entropy(p1),
        float(h1[:, 0].min()) if len(h1) else 0.0,  # earliest loop birth (tightest hole)
    ]

    # location of the most persistent H1 loop (centroid of cocycle vertices)
    if len(p1):
        k = int(np.argmax(p1))
        coc = res["cocycles"][1][k]
        verts = np.unique(coc[:, :2].astype(int)) if len(coc) else np.array([], int)
        loop_ctr = pos[verts].mean(0) if len(verts) else pos.mean(0)
        loop_scale = float(h1[k, 1])
    else:
        loop_ctr, loop_scale = pos.mean(0), 0.0
    loc = [float(loop_ctr[0]), float(loop_ctr[1]), loop_scale]

    return np.array(stats + loc + _empty_circles(pos), dtype=np.float32)


TDA_NAMES = ["h0_max", "h0_sum", "h0_mean", "h0_entropy",
             "h1_count", "h1_maxpers", "h1_sumpers", "h1_entropy", "h1_earliest_birth",
             "loop_x", "loop_y", "loop_scale",
             "gap1_x", "gap1_y", "gap1_r", "gap2_x", "gap2_y", "gap2_r"]


def fit(X, Y):
    mx = HistGradientBoostingRegressor(max_iter=300, random_state=0).fit(X, Y[:, 0])
    my = HistGradientBoostingRegressor(max_iter=300, random_state=0).fit(X, Y[:, 1])
    return mx, my


def med_err(models, X, Y):
    mx, my = models
    pred = np.stack([mx.predict(X), my.predict(X)], axis=1)
    return float(np.median(np.linalg.norm(pred - Y, axis=1)))


def build(samples):
    geo = np.stack([frame_features(p) for p, _ in samples])
    tda = np.stack([tda_features(p) for p, _ in samples])
    Y = np.stack([b for _, b in samples])
    return geo, tda, Y


STRIDE = 15  # 3x coarser than the geometry baseline's stride=5; geo-vs-tda stays fair
             # (both featurized on the SAME frames), just fewer of them → cheaper.


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    base = pathlib.Path("data/sample-data/data")
    print(f"Loading + featurizing (geo + TDA), stride={STRIDE}…", flush=True)
    g_s_tr, t_s_tr, Y_s_tr = build(list(load_metrica_game(base / "Sample_Game_1", stride=STRIDE)))
    print("  soccer train done", flush=True)
    g_s_te, t_s_te, Y_s_te = build(list(load_metrica_game(base / "Sample_Game_2", stride=STRIDE)))
    print("  soccer test done", flush=True)
    eval_games = [glob.glob("data/sportvu/game1/*.json")[0],
                  glob.glob("data/sportvu/01.02.2016.BKN.at.BOS/*.json")[0]]
    pool_games = [glob.glob(f"data/sportvu/{d}/*.json")[0] for d in
                  ("01.01.2016.DAL.at.MIA", "01.01.2016.NYK.at.CHI",
                   "01.02.2016.MIL.at.MIN", "01.02.2016.PHX.at.SAC")]
    g_b_tr, t_b_tr, Y_b_tr = build([s for g in pool_games for s in load_sportvu_game(g, stride=STRIDE)])
    print("  basket train done", flush=True)
    g_b_te, t_b_te, Y_b_te = build([s for g in eval_games for s in load_sportvu_game(g, stride=STRIDE)])
    print("  basket test done", flush=True)
    print(f"soccer {len(Y_s_tr)}/{len(Y_s_te)}  basket {len(Y_b_tr)}/{len(Y_b_te)}")

    refs = {"soccer": {"centroid": 0.231, "deep": 0.101},
            "basket": {"centroid": 0.227, "deep": 0.158}}
    res = {}
    for name, (gtr, ttr, ytr, gte, tte, yte) in {
        "soccer": (g_s_tr, t_s_tr, Y_s_tr, g_s_te, t_s_te, Y_s_te),
        "basket": (g_b_tr, t_b_tr, Y_b_tr, g_b_te, t_b_te, Y_b_te),
    }.items():
        r = refs[name]
        e_geo = med_err(fit(gtr, ytr), gte, yte)
        e_tda = med_err(fit(ttr, ytr), tte, yte)
        both_tr = np.concatenate([gtr, ttr], 1); both_te = np.concatenate([gte, tte], 1)
        m_both = fit(both_tr, ytr)
        e_both = med_err(m_both, both_te, yte)
        rec = lambda e: (r["centroid"] - e) / (r["centroid"] - r["deep"])
        print(f"\n=== {name} (centroide {r['centroid']} · deep {r['deep']}) ===")
        print(f"  geo-only : {e_geo:.4f}  ({rec(e_geo):.0%} del gap)")
        print(f"  tda-only : {e_tda:.4f}  ({rec(e_tda):.0%})   {'BATE' if e_tda < e_geo else 'NO bate'} a geo")
        print(f"  geo+tda  : {e_both:.4f}  ({rec(e_both):.0%})   {'mejora' if e_both < e_geo - 1e-4 else 'no mejora'} sobre geo")

        # does the TDA block add anything inside geo+tda? permutation importance
        rng = np.random.default_rng(0)
        base_e = e_both
        tda_cols = list(range(gtr.shape[1], both_tr.shape[1]))
        deltas = []
        for _ in range(5):
            Xp = both_te.copy(); idx = rng.permutation(len(Xp))
            Xp[:, tda_cols] = both_te[idx][:, tda_cols]
            deltas.append(med_err(m_both, Xp, yte) - base_e)
        print(f"  Δ al permutar TODO el bloque TDA en geo+tda: {np.mean(deltas):+.4f}"
              f"  (≈0 ⇒ el GBM lo ignora)")
        res[name] = {"geo": e_geo, "tda": e_tda, "geo_tda": e_both,
                     "tda_block_importance": float(np.mean(deltas))}

    (OUT / "tda.json").write_text(json.dumps(res, indent=2))
    print(f"\nSaved {OUT/'tda.json'}")


if __name__ == "__main__":
    main()
