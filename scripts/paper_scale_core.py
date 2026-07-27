"""Paper scale-up — the interpretability & velocity claims across 12 matches.

Leave-one-match-out over Metrica g1/g2 + 10 SkillCorner (two leagues). Per held-out
match, four models trained on the rest:
  - centroid baseline (velocity-weighted, untrained)
  - DeepSets  (deep specialist, the "black box" reference)
  - geo-GBM full        (interpretable geometry)
  - geo-GBM positions-only (velocity channel removed)

Reports, per match and aggregated across matches:
  - recovery = (centroid - geo) / (centroid - deep)   → does interpretable geometry
    recover the deep net, at scale?
  - velocity ablation = geo_pos - geo_full             → is velocity the signal, at scale?

Usage: uv run python scripts/paper_scale_core.py
"""
from __future__ import annotations

import glob
import json
import pathlib

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingRegressor

from wheres_the_ball.data.field_tracking import (load_metrica_trajectories,
                                                 load_skillcorner_trajectories)
from wheres_the_ball.features.geometry import frame_features, FEATURE_NAMES

OUT = pathlib.Path("results/nivel3")
CUR = [16, 17, 18, 19, 20]
_VEL = {"vel_centroid", "fastest", "converge", "mean_speed", "max_speed"}
POS_COLS = [i for i, n in enumerate(FEATURE_NAMES)
            if n.replace("_x", "").replace("_y", "") not in _VEL]
torch.manual_seed(0)


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


def train_deepsets(frames, steps=500):
    m = DeepSets(); opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    rng = np.random.default_rng(0)
    for _ in range(steps):
        idx = rng.integers(0, len(frames), 256)
        P, M, Y = collate([frames[i] for i in idx])
        loss = nn.functional.mse_loss(m(P, M), Y)
        opt.zero_grad(); loss.backward(); opt.step()
    return m


def deep_err(m, frames):
    P, M, Y = collate(frames)
    with torch.no_grad():
        pred = m(P, M).numpy()
    return float(np.median(np.linalg.norm(pred - Y.numpy(), axis=1)))


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


def gbm(X, Y):
    return (HistGradientBoostingRegressor(max_iter=300, random_state=0).fit(X, Y[:, 0]),
            HistGradientBoostingRegressor(max_iter=300, random_state=0).fit(X, Y[:, 1]))


def gerr(m, X, Y):
    return float(np.median(np.linalg.norm(np.stack([m[0].predict(X), m[1].predict(X)], 1) - Y, axis=1)))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    base = pathlib.Path("data/sample-data/data")
    matches = {}
    print("Loading + featurizing 12 matches…")
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
    names = list(matches)
    print(f"{len(names)} matches, {sum(len(matches[n]['Y']) for n in names)} frames")

    rows = []
    print(f"\n{'held-out':26}{'centroid':>9}{'deep':>8}{'geo':>8}{'geo-pos':>9}{'recovery':>10}{'vel-abl':>9}")
    for held in names:
        tr = [n for n in names if n != held]
        Xtr = np.concatenate([matches[n]["X"] for n in tr])
        Ytr = np.concatenate([matches[n]["Y"] for n in tr])
        frtr = [f for n in tr for f in matches[n]["frames"]]
        d = matches[held]
        e_cen = float(np.median(np.linalg.norm(d["cen"] - d["Y"], axis=1)))
        e_deep = deep_err(train_deepsets(frtr), d["frames"])
        e_geo = gerr(gbm(Xtr, Ytr), d["X"], d["Y"])
        e_pos = gerr(gbm(Xtr[:, POS_COLS], Ytr), d["X"][:, POS_COLS], d["Y"])
        rec = (e_cen - e_geo) / (e_cen - e_deep) if e_cen > e_deep else float("nan")
        rows.append({"match": held, "centroid": e_cen, "deep": e_deep, "geo": e_geo,
                     "geo_pos": e_pos, "recovery": rec, "vel_ablation": e_pos - e_geo})
        print(f"{held:26}{e_cen:>9.4f}{e_deep:>8.4f}{e_geo:>8.4f}{e_pos:>9.4f}{rec:>9.0%}{e_pos-e_geo:>+9.4f}")

    rec = np.array([r["recovery"] for r in rows]); abl = np.array([r["vel_ablation"] for r in rows])
    deep = np.array([r["deep"] for r in rows]); geo = np.array([r["geo"] for r in rows])
    print(f"\nAcross {len(rows)} matches:")
    print(f"  deep {deep.mean():.4f}±{deep.std():.4f}   geo {geo.mean():.4f}±{geo.std():.4f}")
    print(f"  geometry recovers {rec.mean():.0%} ± {rec.std():.0%} of the deep net")
    print(f"  velocity ablation (geo_pos - geo_full) {abl.mean():+.4f} ± {abl.std():.4f}  "
          f"(velocity helps in {int((abl>0).sum())}/{len(abl)})")
    (OUT / "paper_scale_core.json").write_text(json.dumps(rows, indent=2))
    print(f"\nSaved {OUT/'paper_scale_core.json'}")


if __name__ == "__main__":
    main()
