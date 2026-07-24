"""Independent audit of the Level-2 numbers — recompute, stress-test, add controls.

Targets (the claims that would embarrass us if wrong):
  A. 30-min few-shot advantage of real pretraining: seed-PAIRED deltas pooled across
     the three replicas (v1 fwd, consolidated fwd, reverse), sign test + bootstrap CI.
  B. Ceiling paired comparison: item-level specialist-vs-VLM wins on far, binomial CIs;
     drop-bias check (are the 17 dropped items different?).
  C. B2-B4 anti-correlation in image space: recompute with bootstrap CI.
  D. Asymmetry volume control: basket source has 2x the soccer source samples —
     subsample basket to 76k, retrain, re-eval zero-shot on soccer. If ~0.18 persists,
     the asymmetry is not a data-volume artifact.

Usage: uv run python scripts/audit_nivel2.py [--skip-volume-control]
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import pathlib

import numpy as np

OUT = pathlib.Path("results/nivel2")
FASE1 = pathlib.Path("results/fase1")
rng = np.random.default_rng(0)


def sign_and_boot(deltas, label):
    d = np.array(deltas)
    wins = int((d > 0).sum())
    # exact one-sided sign test
    from math import comb
    n = len(d)
    p = sum(comb(n, k) for k in range(wins, n + 1)) / 2**n
    bs = [np.mean(d[rng.integers(0, n, n)]) for _ in range(10000)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    print(f"  {label:28} n={n:2}  wins={wins:2}  mean Δ={d.mean():+.4f} "
          f"[{lo:+.4f},{hi:+.4f}]  sign-test p={p:.3f}")
    return wins, n, lo


def section_a():
    print("=== A. Ventaja del init real a 30 min — pareado por semilla, 3 réplicas ===")
    print("  (Δ = err_control − err_real_init; >0 = el init real gana)")
    pooled_sc, pooled_pm = [], []
    # forward v1 (3 seeds, keys ft/scratch)
    v1 = json.loads((OUT / "v1_temporal.json").read_text())
    pooled_sc += [s - f for f, s in zip(v1["30min_finetune"], v1["30min_scratch"])]
    # consolidated (5 seeds, 3 arms)
    con = json.loads((OUT / "consolidated.json").read_text())["30min"]
    pooled_sc += [s - f for f, s in zip(con["soccer"], con["scratch"])]
    pooled_pm += [p - f for f, p in zip(con["soccer"], con["permuted"])]
    # reverse (5 seeds, 3 arms)
    rev = json.loads((OUT / "reverse.json").read_text())["30min"]
    pooled_sc += [s - f for f, s in zip(rev["basket"], rev["scratch"])]
    pooled_pm += [p - f for f, p in zip(rev["basket"], rev["permuted"])]
    sign_and_boot(pooled_sc, "vs scratch (13 semillas)")
    sign_and_boot(pooled_pm, "vs permutado (10 semillas)")


def section_b():
    print("\n=== B. Techo pareado: victorias ítem a ítem en far + sesgo del drop ===")
    man = {i["id"]: i for i in json.loads((FASE1 / "manifest.json").read_text())["items"]}
    p = json.loads((FASE1 / "predictions.json").read_text())

    def xy(d):
        return (d["x"], d["y"]) if isinstance(d, dict) and d.get("x") is not None and "error" not in d else None

    have = [k for k in man if xy(p.get(k, {}).get("specialist"))]
    far_have = [k for k in have if man[k]["center_bin"] == "far"]
    far_all = [k for k in man if man[k]["center_bin"] == "far"]
    # drop bias: center-error difficulty of kept vs dropped far items
    def cerr(k):
        return math.hypot(0.5 - man[k]["gt"]["x"], 0.5 - man[k]["gt"]["y"])
    dropped = [k for k in far_all if k not in far_have]
    print(f"  far: usados {len(far_have)}/{len(far_all)}; dificultad (err centro) "
          f"usados={np.median([cerr(k) for k in far_have]):.3f} "
          f"vs descartados={np.median([cerr(k) for k in dropped]):.3f}")
    # item-paired wins vs each VLM on far
    for rival in ("gpt", "claude_opus"):
        wins = 0; n = 0
        for k in far_have:
            ps, pr = xy(p[k]["specialist"]), xy(p[k].get(rival))
            if not pr:
                continue
            g = (man[k]["gt"]["x"], man[k]["gt"]["y"])
            n += 1
            wins += math.hypot(ps[0]-g[0], ps[1]-g[1]) < math.hypot(pr[0]-g[0], pr[1]-g[1])
        se = 1.96 * math.sqrt(wins/n*(1-wins/n)/n)
        print(f"  especialista bate a {rival:12} en far: {wins}/{n} "
              f"({wins/n:.0%} ± {se:.0%})")
    # vs center
    wins = sum(1 for k in far_have
               if math.hypot(xy(p[k]["specialist"])[0]-man[k]["gt"]["x"],
                             xy(p[k]["specialist"])[1]-man[k]["gt"]["y"]) < cerr(k))
    se = 1.96 * math.sqrt(wins/len(far_have)*(1-wins/len(far_have))/len(far_have))
    print(f"  especialista bate a centro       en far: {wins}/{len(far_have)} "
          f"({wins/len(far_have):.0%} ± {se:.0%})")


def section_c():
    print("\n=== C. Anti-correlación geométrica en imagen (bootstrap) ===")
    man = {i["id"]: i for i in json.loads((FASE1 / "manifest.json").read_text())["items"]}
    P, G = [], []
    for k, v in man.items():
        ps = v["players"]
        if not ps:
            continue
        P.append((sum(q["x"] for q in ps)/len(ps), sum(q["y"] for q in ps)/len(ps)))
        G.append((v["gt"]["x"], v["gt"]["y"]))
    P, G = np.array(P), np.array(G)
    r = np.corrcoef(P[:, 0], G[:, 0])[0, 1]
    bs = [np.corrcoef(P[i, 0], G[i, 0])[0, 1] for i in (rng.integers(0, len(P), len(P)) for _ in range(10000))]
    print(f"  centroide corr_x = {r:+.2f} [{np.percentile(bs,2.5):+.2f},{np.percentile(bs,97.5):+.2f}]"
          f"  (robustamente negativa: {np.percentile(bs,97.5) < 0})")


def section_d():
    print("\n=== D. Control de volumen para la asimetría (subsample basket→76k) ===")
    import torch
    import sys
    sys.path.insert(0, "scripts")
    from wheres_the_ball.data.field_tracking import (load_metrica_trajectories,
                                                     load_sportvu_trajectories)
    from nivel2_specialist import DeepSets
    from nivel2_v1_temporal import collate, train, med_corr, D_IN
    torch.manual_seed(0)
    src_games = [glob.glob(f"data/sportvu/{d}/*.json")[0] for d in
                 ("game1", "01.01.2016.DAL.at.MIA", "01.01.2016.NYK.at.CHI",
                  "01.02.2016.BKN.at.BOS")]
    src = [s for g in src_games for s in load_sportvu_trajectories(g)]
    idx = np.random.default_rng(3).permutation(len(src))[:75820]
    sub = [src[i] for i in idx]
    eval_s = list(load_metrica_trajectories(pathlib.Path("data/sample-data/data/Sample_Game_2")))
    Pe, Me, Ye = collate(eval_s)
    m = train(DeepSets(d_in=D_IN), sub, 3000, 1e-3, 0)
    med, corr = med_corr(m, Pe, Me, Ye)
    print(f"  zero-shot basket(76k)→fútbol: med={med:.4f} corr=({corr[0]:+.2f},{corr[1]:+.2f})")
    print(f"  (con 155k era 0.177; fútbol(76k)→basket era 0.347)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-volume-control", action="store_true")
    args = ap.parse_args()
    section_a(); section_b(); section_c()
    if not args.skip_volume_control:
        section_d()
