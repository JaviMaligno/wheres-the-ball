"""Level-2 consolidation — the v1 transfer claim with more data and seeds.

Scale-up of the few-shot + warm-start-control experiment:
  - Soccer source: Metrica g1+g2 + SkillCorner 1886347+1899585 (4 matches).
  - Basketball eval: 2 games (fixed); few-shot pool: 4 other games.
  - Arms: soccer-init / permuted-init (warm-start control) / scratch,
    budgets {1, 5, 30} min, 5 seeds.

Usage: uv run python scripts/nivel2_consolidate.py
"""
from __future__ import annotations

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
import glob as _glob

def _game(dirname):
    return _glob.glob(f"data/sportvu/{dirname}/*.json")[0]

EVAL_GAMES = [_game("game1"), _game("01.02.2016.BKN.at.BOS")]
POOL_GAMES = [_game("01.01.2016.DAL.at.MIA"), _game("01.01.2016.NYK.at.CHI"),
              _game("01.02.2016.MIL.at.MIN"), _game("01.02.2016.PHX.at.SAC")]
SAMPLES_PER_MIN = 300
BUDGETS_MIN = [1, 5, 30]
SEEDS = [0, 1, 2, 3, 4]
STEPS = 600


def main() -> None:
    torch.manual_seed(0)
    base = pathlib.Path("data/sample-data/data")
    print("Loading soccer source (2 Metrica + 2 SkillCorner)…")
    src = (list(load_metrica_trajectories(base / "Sample_Game_1"))
           + list(load_metrica_trajectories(base / "Sample_Game_2"))
           + list(load_skillcorner_trajectories("data/opendata/data/matches/1886347"))
           + list(load_skillcorner_trajectories("data/opendata/data/matches/1899585")))
    print(f"soccer source samples: {len(src)}")
    eval_b = [s for g in EVAL_GAMES for s in load_sportvu_trajectories(g)]
    pool = [s for g in POOL_GAMES for s in load_sportvu_trajectories(g)]
    print(f"eval_basket={len(eval_b)} (2 games)  pool={len(pool)} (4 games)")
    Pb, Mb, Yb = collate(eval_b)

    print("Pretraining source model (3000 steps)…")
    soccer = train(DeepSets(d_in=D_IN), src, 3000, 1e-3, 0)
    torch.save(soccer.state_dict(), OUT / "specialist_v1_big.pt")
    zs, zc = med_corr(soccer, Pb, Mb, Yb)
    print(f"zero-shot basket (2 games): med={zs:.4f} corr=({zc[0]:+.2f},{zc[1]:+.2f})")

    perm = np.random.default_rng(7).permutation(len(src))
    permuted_data = [(src[i][0], src[perm[i]][1]) for i in range(len(src))]
    permuted = train(DeepSets(d_in=D_IN), permuted_data, 3000, 1e-3, 0)
    torch.save(permuted.state_dict(), OUT / "specialist_v1_big_permuted.pt")

    inits = {"soccer": OUT / "specialist_v1_big.pt",
             "permuted": OUT / "specialist_v1_big_permuted.pt",
             "scratch": None}
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
                errs.append(med_corr(m, Pb, Mb, Yb)[0])
            row[cond] = errs
        results[f"{mins}min"] = row
        print(f"{mins:>3} min  " + "  ".join(f"{c}={np.median(v):.4f}" for c, v in row.items()))

    full = train(DeepSets(d_in=D_IN), pool, 2000, 1e-3, 0)
    results["full_scratch"] = med_corr(full, Pb, Mb, Yb)[0]
    print(f"full pool (4 games) scratch: {results['full_scratch']:.4f}")

    (OUT / "consolidated.json").write_text(json.dumps(results, indent=2))
    print(f"Saved {OUT/'consolidated.json'}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "scripts")
    main()
