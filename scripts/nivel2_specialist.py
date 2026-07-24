"""Level-2 Fase 0 — lightweight specialist v0 (DeepSets) on field coordinates.

Train: Metrica soccer game 1. In-domain eval: Metrica game 2 (unseen match).
Zero-shot transfer: SportVU basketball game(s). Baselines evaluated in the same
field coordinates: field-center, centroid, speed-weighted centroid.

The model is permutation-invariant and player-count-agnostic (DeepSets: per-player
MLP -> mean+max pool -> head), so it crosses sports (22 vs 10 players) unchanged.

Usage: uv run python scripts/nivel2_specialist.py [--epochs 12]
Outputs: results/nivel2/specialist_v0.json + model weights.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib

import numpy as np
import torch
import torch.nn as nn

from wheres_the_ball.data.field_tracking import load_metrica_game, load_sportvu_game

OUT = pathlib.Path("results/nivel2")
torch.manual_seed(0)


class DeepSets(nn.Module):
    def __init__(self, d_in=5, d_h=64):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(d_in, d_h), nn.ReLU(),
                                 nn.Linear(d_h, d_h), nn.ReLU())
        self.rho = nn.Sequential(nn.Linear(2 * d_h, d_h), nn.ReLU(),
                                 nn.Linear(d_h, 2), nn.Sigmoid())

    def forward(self, players, mask):
        h = self.phi(players)                        # [B, N, H]
        m = mask.unsqueeze(-1)
        mean = (h * m).sum(1) / m.sum(1).clamp(min=1)
        maxp = h.masked_fill(~mask.unsqueeze(-1), -1e9).max(1).values
        return self.rho(torch.cat([mean, maxp], -1))  # [B, 2] in [0,1]^2


def collate(samples):
    n = max(len(p) for p, _ in samples)
    P = torch.zeros(len(samples), n, 5)
    M = torch.zeros(len(samples), n, dtype=torch.bool)
    Y = torch.zeros(len(samples), 2)
    for i, (p, b) in enumerate(samples):
        P[i, :len(p)] = torch.from_numpy(p); M[i, :len(p)] = True
        Y[i] = torch.from_numpy(b)
    return P, M, Y


def evaluate(model, samples, label, extra_baselines=True):
    P, M, Y = collate(samples)
    with torch.no_grad():
        pred = model(P, M).numpy()
    gt = Y.numpy()
    def med(p):  # median euclidean error in field fractions
        return float(np.median(np.linalg.norm(p - gt, axis=1)))
    def corr(p):
        return (float(np.corrcoef(p[:, 0], gt[:, 0])[0, 1]),
                float(np.corrcoef(p[:, 1], gt[:, 1])[0, 1]))
    res = {"n": len(samples), "specialist": {"med": med(pred), "corr": corr(pred)}}
    if extra_baselines:
        center = np.full_like(gt, 0.5)
        cents, wcents = [], []
        for p, _ in samples:
            cents.append(p[:, :2].mean(0))
            w = np.linalg.norm(p[:, 2:4], axis=1) + 1e-6
            wcents.append((p[:, :2] * w[:, None]).sum(0) / w.sum())
        cents, wcents = np.array(cents), np.array(wcents)
        res["center"] = {"med": med(center)}
        res["centroid"] = {"med": med(cents), "corr": corr(cents)}
        res["vel_centroid"] = {"med": med(wcents), "corr": corr(wcents)}
    print(f"\n=== {label} (n={res['n']}) — median err (field fractions) ===")
    for k, v in res.items():
        if isinstance(v, dict):
            c = f"  corr=({v['corr'][0]:+.2f},{v['corr'][1]:+.2f})" if "corr" in v else ""
            print(f"  {k:14} med={v['med']:.4f}{c}")
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    base = pathlib.Path("data/sample-data/data")
    print("Loading Metrica game 1 (train) / game 2 (in-domain eval)…")
    train = list(load_metrica_game(base / "Sample_Game_1"))
    test_soccer = list(load_metrica_game(base / "Sample_Game_2"))
    print(f"train={len(train)}  test_soccer={len(test_soccer)}")

    svu = sorted(pathlib.Path("data/sportvu").rglob("*.json"))
    test_basket = [s for f in svu for s in load_sportvu_game(f)]
    print(f"test_basket={len(test_basket)} (from {len(svu)} SportVU game(s))")

    model = DeepSets()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    idx = np.arange(len(train))
    bs = 256
    for ep in range(args.epochs):
        np.random.default_rng(ep).shuffle(idx)
        tot = 0.0
        for i in range(0, len(idx), bs):
            batch = [train[j] for j in idx[i:i+bs]]
            P, M, Y = collate(batch)
            loss = nn.functional.mse_loss(model(P, M), Y)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(batch)
        print(f"epoch {ep+1}/{args.epochs}  mse={tot/len(idx):.5f}")

    res = {
        "in_domain_soccer": evaluate(model, test_soccer, "IN-DOMAIN fútbol (Metrica g2)"),
        "zero_shot_basket": evaluate(model, test_basket, "ZERO-SHOT baloncesto (SportVU)"),
    }
    # transfer ratio vs the center baseline gap (how much of the beat-center gap survives)
    torch.save(model.state_dict(), OUT / "specialist_v0.pt")
    (OUT / "specialist_v0.json").write_text(json.dumps(res, indent=2))
    print(f"\nSaved results to {OUT}/specialist_v0.json")


if __name__ == "__main__":
    main()
