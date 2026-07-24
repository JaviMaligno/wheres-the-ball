"""Control for the v1 transfer claim — genuine dynamics transfer, or just warm-start?

Alternative explanation for "finetune-from-soccer beats scratch in v1": the 21-dim
trajectory input is harder to optimize from scratch at low budgets, so ANY pretrained
init would help. Control: pretrain the same architecture on soccer inputs with
PERMUTED ball targets (identical input distribution, destroyed input→ball mapping),
then run the same few-shot finetune. Three arms per budget:

  soccer   : init from real soccer pretraining (the transfer claim)
  permuted : init from permuted-target pretraining (warm-start control)
  scratch  : random init

If permuted ≈ soccer  → the v1 advantage is warm-start, not transferable dynamics.
If permuted ≈ scratch → the dynamics transfer is genuine.

Usage: uv run python scripts/nivel2_control_warmstart.py
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import torch

from wheres_the_ball.data.field_tracking import (load_metrica_trajectories,
                                                 load_sportvu_trajectories)
from nivel2_specialist import DeepSets
from nivel2_v1_temporal import collate, train, med_corr, EVAL_GAME, POOL_GAMES, D_IN

OUT = pathlib.Path("results/nivel2")
SAMPLES_PER_MIN = 300
BUDGETS_MIN = [1, 5, 30]
SEEDS = [0, 1, 2]
STEPS = 600


def main() -> None:
    torch.manual_seed(0)
    base = pathlib.Path("data/sample-data/data")
    print("Loading…")
    train_s = list(load_metrica_trajectories(base / "Sample_Game_1"))
    eval_b = list(load_sportvu_trajectories(EVAL_GAME))
    pool = [s for g in POOL_GAMES for s in load_sportvu_trajectories(g)]
    Pb, Mb, Yb = collate(eval_b)

    # Pretrain the warm-start control: same inputs, permuted ball targets.
    perm = np.random.default_rng(7).permutation(len(train_s))
    permuted = [(train_s[i][0], train_s[perm[i]][1]) for i in range(len(train_s))]
    ctrl = DeepSets(d_in=D_IN)
    ctrl = train(ctrl, permuted, 2000, 1e-3, 0)
    torch.save(ctrl.state_dict(), OUT / "specialist_v1_permuted.pt")
    print(f"permuted-pretrain zero-shot basket: {med_corr(ctrl, Pb, Mb, Yb)[0]:.4f} "
          f"(soccer-pretrain was 0.333)")

    results = {}
    inits = {"soccer": OUT / "specialist_v1.pt",
             "permuted": OUT / "specialist_v1_permuted.pt",
             "scratch": None}
    for mins in BUDGETS_MIN:
        n = mins * SAMPLES_PER_MIN
        row = {}
        for cond, ckpt in inits.items():
            errs = []
            for seed in SEEDS:
                start = np.random.default_rng(100 + seed).integers(0, len(pool) - n)
                chunk = pool[start:start + n]
                m = DeepSets(d_in=D_IN)
                if ckpt is not None:
                    m.load_state_dict(torch.load(ckpt))
                    m = train(m, chunk, STEPS, 5e-4, seed)
                else:
                    m = train(m, chunk, STEPS, 1e-3, seed)
                errs.append(med_corr(m, Pb, Mb, Yb)[0])
            row[cond] = errs
        results[f"{mins}min"] = row
        print(f"{mins:>3} min  " + "  ".join(f"{c}={np.median(v):.4f}" for c, v in row.items()))

    (OUT / "control_warmstart.json").write_text(json.dumps(results, indent=2))
    print(f"Saved {OUT/'control_warmstart.json'}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "scripts")
    main()
