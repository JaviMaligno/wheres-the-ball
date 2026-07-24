"""Level-2 · H3 few-shot — does soccer pretraining help learn basketball faster?

For each budget of basketball footage {1, 5, 30} minutes (contiguous chunks from
games NOT used for eval), train two models on exactly the same samples:
  - finetune : initialized from the soccer specialist (specialist_v0.pt)
  - scratch  : random initialization
and evaluate on the held-out game (same eval as the zero-shot number, 0.325).
If finetune wins at low budgets, soccer geometry transfers as a useful prior;
if they tie, the pretraining adds nothing (the transferable part was already
captured by the untrained centroid, 0.227).

3 seeds per condition (chunk start varies); fixed optimization steps per budget.
Also trains on the full pool (~2 games) as the in-domain basketball reference.

Usage: uv run python scripts/nivel2_fewshot.py
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import torch

from wheres_the_ball.data.field_tracking import load_sportvu_game
from nivel2_specialist import DeepSets, collate  # same model/collate

OUT = pathlib.Path("results/nivel2")
EVAL_GAME = "data/sportvu/game1/0021500492.json"
POOL_GAMES = ["data/sportvu/01.01.2016.DAL.at.MIA/0021500491.json",
              "data/sportvu/01.01.2016.NYK.at.CHI/0021500493.json"]
SAMPLES_PER_MIN = 300   # stride-5 sampling at 25 Hz → 5 samples/s
BUDGETS_MIN = [1, 5, 30]
SEEDS = [0, 1, 2]
STEPS = 600
BATCH = 256


def train(model, samples, steps, lr, seed):
    rng = np.random.default_rng(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        batch = [samples[j] for j in rng.integers(0, len(samples), min(BATCH, len(samples)))]
        P, M, Y = collate(batch)
        loss = torch.nn.functional.mse_loss(model(P, M), Y)
        opt.zero_grad(); loss.backward(); opt.step()
    return model


def med_err(model, P, M, Y):
    with torch.no_grad():
        pred = model(P, M).numpy()
    return float(np.median(np.linalg.norm(pred - Y.numpy(), axis=1)))


def main() -> None:
    torch.manual_seed(0)
    print("Loading eval game + few-shot pool…")
    eval_s = list(load_sportvu_game(EVAL_GAME))
    pool = [s for g in POOL_GAMES for s in load_sportvu_game(g)]
    print(f"eval={len(eval_s)}  pool={len(pool)}")
    Pe, Me, Ye = collate(eval_s)

    base = DeepSets()
    base.load_state_dict(torch.load(OUT / "specialist_v0.pt"))
    print(f"zero-shot reference: {med_err(base, Pe, Me, Ye):.4f}")

    results = {}
    for mins in BUDGETS_MIN:
        n = mins * SAMPLES_PER_MIN
        for cond in ("finetune", "scratch"):
            errs = []
            for seed in SEEDS:
                start = np.random.default_rng(100 + seed).integers(0, len(pool) - n)
                chunk = pool[start:start + n]
                m = DeepSets()
                if cond == "finetune":
                    m.load_state_dict(torch.load(OUT / "specialist_v0.pt"))
                    m = train(m, chunk, STEPS, 5e-4, seed)
                else:
                    m = train(m, chunk, STEPS, 1e-3, seed)
                errs.append(med_err(m, Pe, Me, Ye))
            results[f"{mins}min_{cond}"] = errs
            print(f"{mins:>3} min  {cond:8}  median err per seed: "
                  f"{['%.4f' % e for e in errs]}  → median {np.median(errs):.4f}")

    # in-domain basketball reference: full pool
    m = DeepSets()
    m = train(m, pool, 1500, 1e-3, 0)
    results["full_scratch"] = [med_err(m, Pe, Me, Ye)]
    print(f"full pool (~{len(pool)//SAMPLES_PER_MIN} min) scratch: {results['full_scratch'][0]:.4f}")

    (OUT / "fewshot.json").write_text(json.dumps(results, indent=2))
    print(f"Saved {OUT/'fewshot.json'}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "scripts")
    main()
