"""Level-2 reverse direction — basketball -> soccer (completes the transfer matrix).

Symmetric to nivel2_consolidate.py: source = 4 SportVU games; eval = Metrica game 2;
few-shot pool = Metrica game 1 + 2 SkillCorner matches. Arms: basket-init /
permuted-init / scratch at {1, 5, 30} min, 5 seeds.

Usage: uv run python scripts/nivel2_reverse.py
"""
from __future__ import annotations

import glob
import json
import pathlib

import numpy as np
import torch

from wheres_the_ball.data.field_tracking import (load_metrica_trajectories,
                                                 load_skillcorner_trajectories,
                                                 load_sportvu_trajectories)
from nivel2_specialist import DeepSets
from nivel2_v1_temporal import collate, train, med_corr, D_IN

OUT = pathlib.Path("results/nivel2")
SRC_GAMES = [glob.glob(f"data/sportvu/{d}/*.json")[0] for d in
             ("game1", "01.01.2016.DAL.at.MIA", "01.01.2016.NYK.at.CHI",
              "01.02.2016.BKN.at.BOS")]
SEEDS = [0, 1, 2, 3, 4]
BUDGETS_MIN = [1, 5, 30]
STEPS = 600
SAMPLES_PER_MIN = 300


def main() -> None:
    torch.manual_seed(0)
    base = pathlib.Path("data/sample-data/data")
    print("Loading…")
    src = [s for g in SRC_GAMES for s in load_sportvu_trajectories(g)]
    eval_s = list(load_metrica_trajectories(base / "Sample_Game_2"))
    pool = (list(load_metrica_trajectories(base / "Sample_Game_1"))
            + list(load_skillcorner_trajectories("data/opendata/data/matches/1886347"))
            + list(load_skillcorner_trajectories("data/opendata/data/matches/1899585")))
    print(f"basket source={len(src)}  eval_soccer={len(eval_s)}  pool_soccer={len(pool)}")
    Pe, Me, Ye = collate(eval_s)

    basket = train(DeepSets(d_in=D_IN), src, 3000, 1e-3, 0)
    zs, zc = med_corr(basket, Pe, Me, Ye)
    print(f"zero-shot soccer: med={zs:.4f} corr=({zc[0]:+.2f},{zc[1]:+.2f})")
    torch.save(basket.state_dict(), OUT / "specialist_v1_basket.pt")

    perm = np.random.default_rng(7).permutation(len(src))
    permuted = train(DeepSets(d_in=D_IN),
                     [(src[i][0], src[perm[i]][1]) for i in range(len(src))], 3000, 1e-3, 0)
    torch.save(permuted.state_dict(), OUT / "specialist_v1_basket_permuted.pt")

    inits = {"basket": OUT / "specialist_v1_basket.pt",
             "permuted": OUT / "specialist_v1_basket_permuted.pt", "scratch": None}
    results = {"zero_shot": zs}
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
                errs.append(med_corr(m, Pe, Me, Ye)[0])
            row[cond] = errs
        results[f"{mins}min"] = row
        print(f"{mins:>3} min  " + "  ".join(f"{c}={np.median(v):.4f}" for c, v in row.items()))

    (OUT / "reverse.json").write_text(json.dumps(results, indent=2))
    print(f"Saved {OUT/'reverse.json'}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "scripts")
    main()
