"""Level-2 RQ2 — feature ablation: which features carry the (transferable) signal?

Variants sliced from the same 21-dim trajectories ([x,y,vx,vy]x5 + team):
  pos_traj  : positions only        (x,y)x5 + team      -> 11 dims
  vel_traj  : motion only           (vx,vy)x5 + team    -> 11 dims
  single    : last-step snapshot    (x,y,vx,vy) + team  -> 5 dims (v0)
  full_traj : everything            (x,y,vx,vy)x5 + team-> 21 dims (v1)

For each variant: in-domain soccer (train Metrica g1 -> eval g2), zero-shot basket
(2 eval games), and the transfer readout: 30-min finetune vs scratch (3 seeds).
H2 of the design predicted velocities carry the transferable part.

Usage: uv run python scripts/nivel2_ablation_rq2.py
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
from nivel2_v1_temporal import train, med_corr

OUT = pathlib.Path("results/nivel2")
EVAL_GAMES = [glob.glob("data/sportvu/game1/*.json")[0],
              glob.glob("data/sportvu/01.02.2016.BKN.at.BOS/*.json")[0]]
POOL_GAMES = [glob.glob(f"data/sportvu/{d}/*.json")[0] for d in
              ("01.01.2016.DAL.at.MIA", "01.01.2016.NYK.at.CHI",
               "01.02.2016.MIL.at.MIN", "01.02.2016.PHX.at.SAC")]

POS_IDX = [i for k in range(5) for i in (4*k, 4*k+1)] + [20]
VEL_IDX = [i for k in range(5) for i in (4*k+2, 4*k+3)] + [20]
VARIANTS = {"pos_traj": POS_IDX, "vel_traj": VEL_IDX,
            "single": [16, 17, 18, 19, 20], "full_traj": list(range(21))}
SEEDS = [0, 1, 2]
N30 = 30 * 300


def slice_set(samples, idx):
    return [(p[:, idx], b) for p, b in samples]


def collate_d(samples, d):
    n = max(len(p) for p, _ in samples)
    P = torch.zeros(len(samples), n, d)
    M = torch.zeros(len(samples), n, dtype=torch.bool)
    Y = torch.zeros(len(samples), 2)
    for i, (p, b) in enumerate(samples):
        P[i, :len(p)] = torch.from_numpy(np.ascontiguousarray(p)); M[i, :len(p)] = True
        Y[i] = torch.from_numpy(b)
    return P, M, Y


def train_d(model, samples, d, steps, lr, seed):
    rng = np.random.default_rng(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        batch = [samples[j] for j in rng.integers(0, len(samples), min(256, len(samples)))]
        P, M, Y = collate_d(batch, d)
        loss = torch.nn.functional.mse_loss(model(P, M), Y)
        opt.zero_grad(); loss.backward(); opt.step()
    return model


def main() -> None:
    torch.manual_seed(0)
    base = pathlib.Path("data/sample-data/data")
    print("Loading…")
    tr = list(load_metrica_trajectories(base / "Sample_Game_1"))
    te = list(load_metrica_trajectories(base / "Sample_Game_2"))
    ev = [s for g in EVAL_GAMES for s in load_sportvu_trajectories(g)]
    pool = [s for g in POOL_GAMES for s in load_sportvu_trajectories(g)]
    print(f"train={len(tr)} test_soccer={len(te)} eval_basket={len(ev)} pool={len(pool)}")

    results = {}
    print(f"\n{'variant':10} in_domain  zero_shot  30min_ft  30min_scratch")
    for name, idx in VARIANTS.items():
        d = len(idx)
        trv, tev = slice_set(tr, idx), slice_set(te, idx)
        evv, poolv = slice_set(ev, idx), slice_set(pool, idx)
        Ps, Ms, Ys = collate_d(tev, d)
        Pb, Mb, Yb = collate_d(evv, d)

        src = train_d(DeepSets(d_in=d), trv, d, 2000, 1e-3, 0)
        ind = float(np.median(np.linalg.norm(src(Ps, Ms).detach().numpy() - Ys.numpy(), axis=1)))
        zs = float(np.median(np.linalg.norm(src(Pb, Mb).detach().numpy() - Yb.numpy(), axis=1)))

        ft, sc = [], []
        for seed in SEEDS:
            start = np.random.default_rng(100 + seed).integers(0, len(poolv) - N30)
            chunk = poolv[start:start + N30]
            m = DeepSets(d_in=d); m.load_state_dict(src.state_dict())
            m = train_d(m, chunk, d, 600, 5e-4, seed)
            ft.append(float(np.median(np.linalg.norm(m(Pb, Mb).detach().numpy() - Yb.numpy(), axis=1))))
            m2 = train_d(DeepSets(d_in=d), chunk, d, 600, 1e-3, seed)
            sc.append(float(np.median(np.linalg.norm(m2(Pb, Mb).detach().numpy() - Yb.numpy(), axis=1))))
        results[name] = {"in_domain": ind, "zero_shot": zs, "ft30": ft, "scratch30": sc}
        print(f"{name:10} {ind:.4f}     {zs:.4f}     {np.median(ft):.4f}    {np.median(sc):.4f}")

    (OUT / "ablation_rq2.json").write_text(json.dumps(results, indent=2))
    print(f"Saved {OUT/'ablation_rq2.json'}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "scripts")
    main()
