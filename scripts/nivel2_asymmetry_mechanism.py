"""Why does basketball→soccer transfer but not the reverse? Mechanism tests.

Hypothesis (from Level 2): basketball's ball is tightly coupled to the moving mass,
so the model learns a strong coupling that generalizes; soccer's loose play teaches
a weak one. Three tests:

  1. Coupling stats: distribution of dist(ball, speed-weighted centroid) per sport.
  2. Error-vs-coupling: stratify each direction's zero-shot errors by the target
     frame's coupling distance (quartiles). Prediction: basket→soccer degrades on
     loose-ball frames; soccer→basket is bad everywhere (weak prior).
  3. Causal-ish: train a soccer model ONLY on tightly-coupled frames (below-median
     coupling, subsampled to a fixed count) vs a matched all-frames model, and
     compare zero-shot on basketball. If coupled-only transfers better, the
     mechanism is the coupling, not the sport.

Usage: uv run python scripts/nivel2_asymmetry_mechanism.py
"""
from __future__ import annotations

import glob
import json
import pathlib

import numpy as np
import torch

from wheres_the_ball.data.field_tracking import (load_metrica_trajectories,
                                                 load_sportvu_trajectories)
from nivel2_specialist import DeepSets
from nivel2_v1_temporal import collate, train, D_IN

OUT = pathlib.Path("results/nivel3")


def coupling(sample):
    """dist(ball, speed-weighted centroid) at the target step."""
    p, b = sample
    pos = p[:, [16, 17]]              # last-step x,y
    vel = p[:, [18, 19]]
    w = np.linalg.norm(vel, axis=1) + 1e-6
    c = (pos * w[:, None]).sum(0) / w.sum()
    return float(np.linalg.norm(b - c))


def errors(model, samples):
    P, M, Y = collate(samples)
    with torch.no_grad():
        pred = model(P, M).numpy()
    return np.linalg.norm(pred - Y.numpy(), axis=1)


def main() -> None:
    torch.manual_seed(0)
    base = pathlib.Path("data/sample-data/data")
    print("Loading…")
    soccer_tr = list(load_metrica_trajectories(base / "Sample_Game_1"))
    soccer_te = list(load_metrica_trajectories(base / "Sample_Game_2"))
    eval_games = [glob.glob("data/sportvu/game1/*.json")[0],
                  glob.glob("data/sportvu/01.02.2016.BKN.at.BOS/*.json")[0]]
    pool_games = [glob.glob(f"data/sportvu/{d}/*.json")[0] for d in
                  ("01.01.2016.DAL.at.MIA", "01.01.2016.NYK.at.CHI",
                   "01.02.2016.MIL.at.MIN", "01.02.2016.PHX.at.SAC")]
    basket_tr = [s for g in pool_games for s in load_sportvu_trajectories(g)]
    basket_te = [s for g in eval_games for s in load_sportvu_trajectories(g)]

    # --- 1. coupling stats per sport ---
    cs = np.array([coupling(s) for s in soccer_te])
    cb = np.array([coupling(s) for s in basket_te])
    print("\n1) Acoplamiento dist(balón, vel-centroide):")
    for name, c in (("fútbol", cs), ("basket", cb)):
        q = np.percentile(c, [25, 50, 75, 90])
        print(f"   {name:7} p25={q[0]:.3f} p50={q[1]:.3f} p75={q[2]:.3f} p90={q[3]:.3f}")

    # --- 2. error vs coupling for both zero-shot directions ---
    soccer_model = train(DeepSets(d_in=D_IN), soccer_tr, 3000, 1e-3, 0)
    basket_model = train(DeepSets(d_in=D_IN), basket_tr, 3000, 1e-3, 0)
    res = {"coupling": {"soccer": np.percentile(cs, [25, 50, 75]).tolist(),
                        "basket": np.percentile(cb, [25, 50, 75]).tolist()}}
    print("\n2) Error zero-shot por cuartil de acoplamiento del frame destino:")
    for tag, model, samples, c in (("basket→fútbol", basket_model, soccer_te, cs),
                                   ("fútbol→basket", soccer_model, basket_te, cb)):
        e = errors(model, samples)
        qs = np.quantile(c, [0.25, 0.5, 0.75])
        bins = np.digitize(c, qs)
        meds = [float(np.median(e[bins == k])) for k in range(4)]
        print(f"   {tag}: Q1(pegado)={meds[0]:.3f}  Q2={meds[1]:.3f}  "
              f"Q3={meds[2]:.3f}  Q4(suelto)={meds[3]:.3f}")
        res[tag] = meds

    # --- 3. coupled-only soccer training ---
    csr = np.array([coupling(s) for s in soccer_tr])
    med = np.median(csr)
    coupled = [s for s, c in zip(soccer_tr, csr) if c <= med]
    n = len(coupled)
    rng = np.random.default_rng(0)
    matched_all = [soccer_tr[i] for i in rng.permutation(len(soccer_tr))[:n]]
    m_coupled = train(DeepSets(d_in=D_IN), coupled, 3000, 1e-3, 1)
    m_all = train(DeepSets(d_in=D_IN), matched_all, 3000, 1e-3, 1)
    e_coupled = float(np.median(errors(m_coupled, basket_te)))
    e_all = float(np.median(errors(m_all, basket_te)))
    print(f"\n3) Zero-shot fútbol→basket (mismos n={n} muestras de entreno):")
    print(f"   entrenado solo en frames ACOPLADOS: {e_coupled:.4f}")
    print(f"   entrenado en frames mezclados:      {e_all:.4f}")
    res["coupled_only"] = {"coupled": e_coupled, "all_matched": e_all}

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "asymmetry_mechanism.json").write_text(json.dumps(res, indent=2))
    print(f"\nSaved {OUT/'asymmetry_mechanism.json'}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "scripts")
    main()
