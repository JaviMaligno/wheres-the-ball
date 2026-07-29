"""B2 — a properly TUNED, multi-seed DeepSets ceiling (kills the strawman baseline).

paper_scale_core.py trained the per-fold DeepSets for only 500 steps, single seed, no
validation — so the interpretable geometry model (0.099) beat it (0.126) and "recovery"
came out at an inflated 132%. Reviewers correctly called this a strawman. Here we train
the DeepSets properly: more steps, a validation split with early stopping, and multiple
seeds, leave-one-match-out over the same 12 soccer matches. We report the tuned deep
error (mean over seeds, ±across-fold and ±across-seed) and the corrected recovery
= (centroid - geo) / (centroid - deep_tuned).

Loads all 12 matches once (featurization is the slow part) then runs the folds in memory.

Usage: uv run python scripts/paper_deep_tuned.py
"""
from __future__ import annotations

import glob
import json
import pathlib
import pickle

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingRegressor

from wheres_the_ball.data.field_tracking import (load_metrica_trajectories,
                                                 load_skillcorner_trajectories)
from wheres_the_ball.features.geometry import frame_features

OUT = pathlib.Path("results/nivel3")
CUR = [16, 17, 18, 19, 20]
SEEDS = [0, 1, 2]
MAX_STEPS = 4000
PATIENCE = 8          # early-stop patience on val (checked every EVAL_EVERY)
EVAL_EVERY = 200
VAL_FRAC = 0.1
BATCH = 256


class DeepSets(nn.Module):
    def __init__(self, d_h=64):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(5, d_h), nn.ReLU(), nn.Linear(d_h, d_h), nn.ReLU())
        self.rho = nn.Sequential(nn.Linear(2 * d_h, d_h), nn.ReLU(), nn.Linear(d_h, 2))

    def forward(self, P, M):
        h = self.phi(P); m = M.unsqueeze(-1)
        pooled = torch.cat([(h * m).sum(1) / m.sum(1).clamp(min=1),
                            h.masked_fill(~m.bool(), -1e9).max(1).values], -1)
        return self.rho(pooled)


def collate(frames):
    n = max(len(p) for p, _ in frames)
    P = torch.zeros(len(frames), n, 5); M = torch.zeros(len(frames), n); Y = torch.zeros(len(frames), 2)
    for i, (p, b) in enumerate(frames):
        P[i, :len(p)] = torch.from_numpy(np.asarray(p, np.float32)); M[i, :len(p)] = 1
        Y[i] = torch.from_numpy(np.asarray(b, np.float32))
    return P, M, Y


def med_err(pred, Y):
    return float(np.median(np.linalg.norm(pred - Y, axis=1)))


def train_tuned(frames, seed):
    """Train with a val split + early stopping; return the best-val model."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    idx = rng.permutation(len(frames)); k = int((1 - VAL_FRAC) * len(frames))
    tr = [frames[i] for i in idx[:k]]; va = [frames[i] for i in idx[k:]]
    Pv, Mv, Yv = collate(va)
    m = DeepSets(); opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    best_val, best_state, since = np.inf, None, 0
    for step in range(1, MAX_STEPS + 1):
        bi = rng.integers(0, len(tr), BATCH)
        P, M, Y = collate([tr[j] for j in bi])
        loss = nn.functional.mse_loss(m(P, M), Y)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % EVAL_EVERY == 0:
            with torch.no_grad():
                v = float(nn.functional.mse_loss(m(Pv, Mv), Yv))
            if v < best_val - 1e-6:
                best_val, best_state, since = v, {k: x.clone() for k, x in m.state_dict().items()}, 0
            else:
                since += 1
                if since >= PATIENCE:
                    break
    if best_state is not None:
        m.load_state_dict(best_state)
    return m


def deep_err(m, frames):
    P, M, Y = collate(frames)
    with torch.no_grad():
        pred = m(P, M).numpy()
    return med_err(pred, Y.numpy())


def load_match(loader, path):
    fr = [(p[:, CUR], b) for p, b in loader(path)]
    if not fr:
        return None
    X = np.stack([frame_features(p) for p, _ in fr]); Y = np.stack([b for _, b in fr])
    cen = []
    for p, b in fr:
        w = np.linalg.norm(p[:, 2:4], axis=1) + 1e-6
        cen.append((p[:, :2] * w[:, None]).sum(0) / w.sum())
    return {"X": X, "Y": Y, "cen": np.array(cen), "frames": fr}


def gbm_err(Xtr, Ytr, Xte, Yte):
    mx = HistGradientBoostingRegressor(max_iter=300, random_state=0).fit(Xtr, Ytr[:, 0])
    my = HistGradientBoostingRegressor(max_iter=300, random_state=0).fit(Xtr, Ytr[:, 1])
    return med_err(np.stack([mx.predict(Xte), my.predict(Xte)], 1), Yte)


def build_matches():
    base = pathlib.Path("data/sample-data/data")
    matches = {}
    for g in sorted(glob.glob(str(base / "Sample_Game_*"))):
        try:
            d = load_match(load_metrica_trajectories, g)
        except Exception:
            continue
        if d:
            matches["metrica/" + pathlib.Path(g).name.replace("Sample_Game_", "g")] = d
    for m in sorted(glob.glob("data/opendata/data/matches/*")):
        try:
            d = load_match(load_skillcorner_trajectories, m)
        except Exception:
            continue
        if d:
            matches["skillcorner/" + pathlib.Path(m).name] = d
    return matches


def load_cached_matches():
    """Featurize the 12 soccer matches once; cache to disk (reused by B2/B4/M5)."""
    cache = OUT / "_feat_soccer12.pkl"
    if cache.exists():
        print(f"Loading featurized matches from cache {cache}", flush=True)
        return pickle.loads(cache.read_bytes())
    print("Featurizing 12 matches (first run, will cache)…", flush=True)
    matches = build_matches()
    cache.write_bytes(pickle.dumps(matches))
    print(f"Cached to {cache}", flush=True)
    return matches


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    matches = load_cached_matches()
    names = list(matches)
    print(f"{len(names)} matches", flush=True)

    rows = []
    for held in names:
        tr = [n for n in names if n != held]
        Xtr = np.concatenate([matches[n]["X"] for n in tr]); Ytr = np.concatenate([matches[n]["Y"] for n in tr])
        frtr = [f for n in tr for f in matches[n]["frames"]]
        d = matches[held]
        geo = gbm_err(Xtr, Ytr, d["X"], d["Y"])
        cen = med_err(d["cen"], d["Y"])
        deep_seeds = [deep_err(train_tuned(frtr, s), d["frames"]) for s in SEEDS]
        deep = float(np.mean(deep_seeds))
        rec = (cen - geo) / (cen - deep) if cen > deep else float("nan")
        rows.append({"match": held, "centroid": cen, "geo": geo, "deep_tuned": deep,
                     "deep_seed_std": float(np.std(deep_seeds)), "recovery": rec})
        print(f"{held:26} centroid={cen:.4f} geo={geo:.4f} deep_tuned={deep:.4f}"
              f"(±{np.std(deep_seeds):.4f} seed) rec={rec:.0%}", flush=True)

    geo = np.array([r["geo"] for r in rows]); deep = np.array([r["deep_tuned"] for r in rows])
    cen = np.array([r["centroid"] for r in rows]); rec = np.array([r["recovery"] for r in rows])
    print(f"\nAcross {len(rows)} matches:")
    print(f"  centroid {cen.mean():.4f}±{cen.std():.4f}  geo {geo.mean():.4f}±{geo.std():.4f}  "
          f"deep_tuned {deep.mean():.4f}±{deep.std():.4f}")
    print(f"  mean across-seed std of deep = {np.mean([r['deep_seed_std'] for r in rows]):.4f}")
    print(f"  corrected recovery = {rec.mean():.0%} ± {rec.std():.0%}")
    print(f"  deep_tuned beats geo in {int((deep < geo).sum())}/{len(rows)} matches")
    (OUT / "paper_deep_tuned.json").write_text(json.dumps(rows, indent=2))
    print(f"Saved {OUT/'paper_deep_tuned.json'}")


if __name__ == "__main__":
    main()
