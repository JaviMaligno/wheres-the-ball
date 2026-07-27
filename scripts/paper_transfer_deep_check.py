"""URGENT check — does the Level-2 DeepSets transfer asymmetry survive at scale?

Article 2's headline (basketball->soccer transfers, soccer->basketball fails) was
measured with the DeepSets on few matches. The interpretable geo-GBM at scale shows
near-symmetric ABSOLUTE zero-shot transfer. This isolates the model: run the SAME
DeepSets family at scale (12 soccer + 6 basket) and read the absolute zero-shot errors
both directions. If b2s << s2b holds, article 2 is fine; if it washes out, it's fragile.

Usage: uv run python scripts/paper_transfer_deep_check.py
"""
from __future__ import annotations

import glob
import json
import pathlib

import numpy as np
import torch
import torch.nn as nn

from wheres_the_ball.data.field_tracking import (load_metrica_trajectories,
                                                 load_skillcorner_trajectories,
                                                 load_sportvu_trajectories)

CUR = [16, 17, 18, 19, 20]
OUT = pathlib.Path("results/nivel3")
torch.manual_seed(0)


class DeepSets(nn.Module):
    def __init__(self, d_in=5, d_h=64):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(d_in, d_h), nn.ReLU(), nn.Linear(d_h, d_h), nn.ReLU())
        self.rho = nn.Sequential(nn.Linear(2 * d_h, d_h), nn.ReLU(), nn.Linear(d_h, 2))

    def forward(self, P, M):
        h = self.phi(P); m = M.unsqueeze(-1)
        pooled = torch.cat([(h * m).sum(1) / m.sum(1).clamp(min=1),
                            h.masked_fill(~m.bool(), -1e9).max(1).values], -1)
        return self.rho(pooled)


def collate(frames, cols):
    n = max(len(p) for p, _ in frames)
    P = torch.zeros(len(frames), n, len(cols)); M = torch.zeros(len(frames), n); Y = torch.zeros(len(frames), 2)
    for i, (p, b) in enumerate(frames):
        P[i, :len(p)] = torch.from_numpy(np.asarray(p[:, cols], np.float32)); M[i, :len(p)] = 1
        Y[i] = torch.from_numpy(np.asarray(b, np.float32))
    return P, M, Y


def train(frames, cols, steps=800):
    m = DeepSets(d_in=len(cols)); opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    rng = np.random.default_rng(0)
    for _ in range(steps):
        idx = rng.integers(0, len(frames), 256)
        P, M, Y = collate([frames[i] for i in idx], cols)
        loss = nn.functional.mse_loss(m(P, M), Y); opt.zero_grad(); loss.backward(); opt.step()
    return m


def err(m, frames, cols):
    P, M, Y = collate(frames, cols)
    with torch.no_grad():
        pred = m(P, M).numpy()
    return float(np.median(np.linalg.norm(pred - Y.numpy(), axis=1)))


def load_sport(specs):
    out = []
    for loader, path in specs:
        try:
            out += [(p[:, CUR], b) for p, b in loader(path)]
        except Exception:
            pass
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    base = pathlib.Path("data/sample-data/data")
    soccer = load_sport([(load_metrica_trajectories, g) for g in sorted(glob.glob(str(base / "Sample_Game_*")))]
                        + [(load_skillcorner_trajectories, m) for m in sorted(glob.glob("data/opendata/data/matches/*"))])
    basket = load_sport([(load_sportvu_trajectories, glob.glob(f"{d}/*.json")[0])
                         for d in sorted(glob.glob("data/sportvu/*")) if glob.glob(f"{d}/*.json")])
    print(f"soccer {len(soccer)} frames · basket {len(basket)} frames")
    rng = np.random.default_rng(0)

    # hold out 20% of each sport for in-domain; train zero-shot on the full other sport
    def split(a):
        idx = rng.permutation(len(a)); k = int(0.8 * len(a))
        return [a[i] for i in idx[:k]], [a[i] for i in idx[k:]]
    s_tr, s_te = split(soccer); b_tr, b_te = split(basket)

    res = {}
    for tag, cols in [("full (x,y,vx,vy,team)", [0, 1, 2, 3, 4]), ("positions-only (x,y,team)", [0, 1, 4])]:
        m_s = train(s_tr, cols); m_b = train(b_tr, cols)
        id_s, id_b = err(m_s, s_te, cols), err(m_b, b_te, cols)
        s2b, b2s = err(m_s, b_te, cols), err(m_b, s_te, cols)
        print(f"\n=== DeepSets · {tag} ===")
        print(f"  in-domain: soccer {id_s:.4f}  basket {id_b:.4f}")
        print(f"  zero-shot ABS: soccer→basket {s2b:.4f}   basket→soccer {b2s:.4f}")
        print(f"  ratio: s→b {s2b/id_b:.2f}×   b→s {b2s/id_s:.2f}×")
        print(f"  L2 claim (b→s << s→b, absolute)?  {'HOLDS' if b2s < s2b - 0.02 else 'does NOT hold'}")
        res[tag] = {"id_soccer": id_s, "id_basket": id_b, "s2b": s2b, "b2s": b2s}
    (OUT / "paper_transfer_deep_check.json").write_text(json.dumps(res, indent=2))
    print(f"\nSaved {OUT/'paper_transfer_deep_check.json'}")


if __name__ == "__main__":
    main()
