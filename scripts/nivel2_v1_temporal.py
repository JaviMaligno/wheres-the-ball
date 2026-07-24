"""Level-2 specialist v1 (temporal) — the decisive transfer test.

Same DeepSets as v0 but each player is a 1-second TRAJECTORY (5 steps of x,y,vx,vy
+ team flag = 21 dims). Same battery as v0/few-shot so results are comparable:

  1. in-domain soccer (train Metrica g1 → eval g2)          [v0: 0.126]
  2. zero-shot basketball (SportVU eval game)               [v0: 0.325 · centroid 0.227]
  3. few-shot finetune-vs-scratch at {1,5,30} min           [v0: scratch >= finetune everywhere]

Question: does temporal structure make soccer pretraining transfer, where the
per-frame model didn't?

Usage: uv run python scripts/nivel2_v1_temporal.py
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import torch

from wheres_the_ball.data.field_tracking import (load_metrica_trajectories,
                                                 load_sportvu_trajectories)
from nivel2_specialist import DeepSets, collate as collate5  # base class

OUT = pathlib.Path("results/nivel2")
EVAL_GAME = "data/sportvu/game1/0021500492.json"
POOL_GAMES = ["data/sportvu/01.01.2016.DAL.at.MIA/0021500491.json",
              "data/sportvu/01.01.2016.NYK.at.CHI/0021500493.json"]
D_IN = 21
SAMPLES_PER_MIN = 300
BUDGETS_MIN = [1, 5, 30]
SEEDS = [0, 1, 2]
STEPS = 600
BATCH = 256


def collate(samples):
    n = max(len(p) for p, _ in samples)
    P = torch.zeros(len(samples), n, D_IN)
    M = torch.zeros(len(samples), n, dtype=torch.bool)
    Y = torch.zeros(len(samples), 2)
    for i, (p, b) in enumerate(samples):
        P[i, :len(p)] = torch.from_numpy(p); M[i, :len(p)] = True
        Y[i] = torch.from_numpy(b)
    return P, M, Y


def train(model, samples, steps, lr, seed):
    rng = np.random.default_rng(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        batch = [samples[j] for j in rng.integers(0, len(samples), min(BATCH, len(samples)))]
        P, M, Y = collate(batch)
        loss = torch.nn.functional.mse_loss(model(P, M), Y)
        opt.zero_grad(); loss.backward(); opt.step()
    return model


def med_corr(model, P, M, Y):
    with torch.no_grad():
        pred = model(P, M).numpy()
    gt = Y.numpy()
    med = float(np.median(np.linalg.norm(pred - gt, axis=1)))
    return med, (float(np.corrcoef(pred[:, 0], gt[:, 0])[0, 1]),
                 float(np.corrcoef(pred[:, 1], gt[:, 1])[0, 1]))


def main() -> None:
    torch.manual_seed(0)
    base = pathlib.Path("data/sample-data/data")
    print("Loading trajectories…")
    train_s = list(load_metrica_trajectories(base / "Sample_Game_1"))
    test_soccer = list(load_metrica_trajectories(base / "Sample_Game_2"))
    eval_b = list(load_sportvu_trajectories(EVAL_GAME))
    pool = [s for g in POOL_GAMES for s in load_sportvu_trajectories(g)]
    print(f"train={len(train_s)}  test_soccer={len(test_soccer)}  "
          f"eval_basket={len(eval_b)}  pool={len(pool)}")

    # 1. train v1 on soccer (2000 Adam steps ≈ the 12-epoch budget of v0)
    model = DeepSets(d_in=D_IN)
    model = train(model, train_s, 2000, 1e-3, 0)
    torch.save(model.state_dict(), OUT / "specialist_v1.pt")

    Ps, Ms, Ys = collate(test_soccer)
    med_s, corr_s = med_corr(model, Ps, Ms, Ys)
    print(f"\nIN-DOMAIN fútbol v1: med={med_s:.4f} corr=({corr_s[0]:+.2f},{corr_s[1]:+.2f})  [v0: 0.126]")

    Pb, Mb, Yb = collate(eval_b)
    med_b, corr_b = med_corr(model, Pb, Mb, Yb)
    print(f"ZERO-SHOT basket v1: med={med_b:.4f} corr=({corr_b[0]:+.2f},{corr_b[1]:+.2f})  [v0: 0.325 · centroide 0.227]")

    results = {"in_domain_soccer": med_s, "zero_shot_basket": med_b}

    # 3. few-shot finetune vs scratch
    print("\nFew-shot (mediana de 3 semillas):")
    for mins in BUDGETS_MIN:
        n = mins * SAMPLES_PER_MIN
        for cond in ("finetune", "scratch"):
            errs = []
            for seed in SEEDS:
                start = np.random.default_rng(100 + seed).integers(0, len(pool) - n)
                chunk = pool[start:start + n]
                m = DeepSets(d_in=D_IN)
                if cond == "finetune":
                    m.load_state_dict(torch.load(OUT / "specialist_v1.pt"))
                    m = train(m, chunk, STEPS, 5e-4, seed)
                else:
                    m = train(m, chunk, STEPS, 1e-3, seed)
                errs.append(med_corr(m, Pb, Mb, Yb)[0])
            results[f"{mins}min_{cond}"] = errs
            print(f"  {mins:>3} min {cond:8} → mediana {np.median(errs):.4f}  ({['%.3f'%e for e in errs]})")

    m = DeepSets(d_in=D_IN)
    m = train(m, pool, 1500, 1e-3, 0)
    results["full_scratch"] = med_corr(m, Pb, Mb, Yb)[0]
    print(f"  full pool scratch → {results['full_scratch']:.4f}  [v0: 0.170]")

    (OUT / "v1_temporal.json").write_text(json.dumps(results, indent=2))
    print(f"Saved {OUT/'v1_temporal.json'}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "scripts")
    main()
