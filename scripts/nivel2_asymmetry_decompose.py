"""Close the asymmetry: is it a VELOCITY-SCALE mismatch?

Hypothesis chain: RQ2 showed velocity-USE is the transferable-but-sport-specific
part. Soccer has long balls -> heavy-tailed velocities; the soccer model may learn
a velocity-use tuned to that wide distribution that misfires on basketball's tighter
velocities, while basketball's milder velocity-use transfers safely.

Decisive test: train POSITIONS-ONLY models in both sports and cross them zero-shot.
If positions-only transfers ~symmetrically (both directions good), the asymmetry
lives entirely in the velocity channel.

Setups mirror the headline zero-shot numbers:
  soccer source = 2 Metrica + 2 SkillCorner   -> eval 2 SportVU games  (headline 0.347)
  basket source = 4 SportVU games             -> eval Metrica g2       (headline 0.177)

Usage: uv run python scripts/nivel2_asymmetry_decompose.py
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
from nivel2_v1_temporal import train, med_corr

OUT = pathlib.Path("results/nivel3")
POS_IDX = [i for k in range(5) for i in (4 * k, 4 * k + 1)] + [20]  # (x,y)x5 + team = 11


def collate_idx(samples, idx):
    d = len(idx)
    n = max(len(p) for p, _ in samples)
    P = torch.zeros(len(samples), n, d); M = torch.zeros(len(samples), n, dtype=torch.bool)
    Y = torch.zeros(len(samples), 2)
    for i, (p, b) in enumerate(samples):
        P[i, :len(p)] = torch.from_numpy(np.ascontiguousarray(p[:, idx])); M[i, :len(p)] = True
        Y[i] = torch.from_numpy(b)
    return P, M, Y


def train_idx(samples, idx, steps=3000, lr=1e-3, seed=0):
    d = len(idx); model = DeepSets(d_in=d)
    rng = np.random.default_rng(seed); opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        batch = [samples[j] for j in rng.integers(0, len(samples), min(256, len(samples)))]
        P, M, Y = collate_idx(batch, idx)
        loss = torch.nn.functional.mse_loss(model(P, M), Y)
        opt.zero_grad(); loss.backward(); opt.step()
    return model


def zshot(model, samples, idx):
    P, M, Y = collate_idx(samples, idx)
    with torch.no_grad():
        pred = model(P, M).numpy()
    return float(np.median(np.linalg.norm(pred - Y.numpy(), axis=1)))


def main() -> None:
    torch.manual_seed(0)
    base = pathlib.Path("data/sample-data/data")
    print("Loading…")
    soccer_src = (list(load_metrica_trajectories(base / "Sample_Game_1"))
                  + list(load_metrica_trajectories(base / "Sample_Game_2"))
                  + list(load_skillcorner_trajectories("data/opendata/data/matches/1886347"))
                  + list(load_skillcorner_trajectories("data/opendata/data/matches/1899585")))
    soccer_eval = list(load_metrica_trajectories(base / "Sample_Game_2"))
    bg = lambda d: glob.glob(f"data/sportvu/{d}/*.json")[0]
    basket_src = [s for d in ("01.01.2016.DAL.at.MIA", "01.01.2016.NYK.at.CHI",
                              "01.02.2016.MIL.at.MIN", "01.02.2016.PHX.at.SAC")
                  for s in load_sportvu_trajectories(bg(d))]
    basket_eval = [s for d in ("game1", "01.02.2016.BKN.at.BOS")
                   for s in load_sportvu_trajectories(bg(d))]

    # velocity magnitude distributions (last-step speed, field-fractions/s)
    def speeds(samples):
        v = np.concatenate([np.linalg.norm(p[:, [18, 19]], axis=1) for p, _ in samples])
        return np.percentile(v, [50, 90, 99]), v.mean()
    print("\nVelocidad (magnitud, fracciones campo/s):")
    for name, s in (("fútbol", soccer_src), ("basket", basket_src)):
        q, m = speeds(s)
        print(f"  {name:7} media={m:.3f} p50={q[0]:.3f} p90={q[1]:.3f} p99={q[2]:.3f}")

    full = list(range(21))
    res = {}
    for variant, idx in (("full", full), ("pos_only", POS_IDX)):
        sm = train_idx(soccer_src, idx)
        bm = train_idx(basket_src, idx)
        s2b = zshot(sm, basket_eval, idx)   # soccer→basket
        b2s = zshot(bm, soccer_eval, idx)   # basket→soccer
        res[variant] = {"soccer_to_basket": s2b, "basket_to_soccer": b2s,
                        "asymmetry": s2b - b2s}
        print(f"\n[{variant}] zero-shot:")
        print(f"  fútbol→basket = {s2b:.4f}   basket→fútbol = {b2s:.4f}   "
              f"asimetría (s2b−b2s) = {s2b - b2s:+.4f}")

    drop = res["full"]["asymmetry"] - res["pos_only"]["asymmetry"]
    print(f"\nLa asimetría pasa de {res['full']['asymmetry']:+.3f} (full) a "
          f"{res['pos_only']['asymmetry']:+.3f} (solo-posiciones) → "
          f"el canal de velocidad explica {drop:+.3f} de la asimetría")
    (OUT / "asymmetry_decompose.json").write_text(json.dumps(res, indent=2))
    print(f"Saved {OUT/'asymmetry_decompose.json'}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "scripts")
    main()
