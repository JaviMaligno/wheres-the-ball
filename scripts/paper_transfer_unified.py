"""B4 — cross-sport transfer under ONE protocol and ONE metric (kills the confound).

The paper mixed protocols: the deep transfer (0.196/0.395) used a random 80/20 split over
pooled frames (leakage-prone: adjacent frames of the same match in train and test), while
the geo transfer used leave-one-match-out. And the quoted "ratio 0.52 -> 0.09" was
untraceable (it was the deep-check ratio grafted onto the geo narrative). Here EVERYTHING
uses the same protocol and metric:

  - Protocol: leave-one-MATCH/GAME-out in-domain (no within-match frame leakage), pooled
    cross-sport zero-shot. Same for deep AND geo.
  - Models: tuned multi-seed DeepSets (same trainer as B2) and the geo-GBM.
  - Metric (single, stated once): transfer penalty = zero-shot error / in-domain error,
    per direction; asymmetry = penalty(s->b) - penalty(b->s). Reported for full features
    and positions-only, for both models.

Reuses the cached soccer featurization (_feat_soccer12.pkl) and caches basketball too.

Usage: uv run python scripts/paper_transfer_unified.py
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
                                                 load_skillcorner_trajectories,
                                                 load_sportvu_trajectories)
from wheres_the_ball.features.geometry import frame_features, FEATURE_NAMES
# reuse the B2 tuned trainer + helpers (same net, val split, early stopping, seeds)
from paper_deep_tuned import (DeepSets, collate, train_tuned, deep_err, load_match,
                              load_cached_matches, SEEDS)

OUT = pathlib.Path("results/nivel3")
CUR = [16, 17, 18, 19, 20]
_VEL = {"vel_centroid", "fastest", "converge", "mean_speed", "max_speed"}
POS_COLS = [i for i, n in enumerate(FEATURE_NAMES)
            if n.replace("_x", "").replace("_y", "") not in _VEL]


def load_cached_basket():
    cache = OUT / "_feat_basket6.pkl"
    if cache.exists():
        print(f"Loading basketball from cache {cache}", flush=True)
        return pickle.loads(cache.read_bytes())
    print("Featurizing basketball (first run, will cache)…", flush=True)
    matches = {}
    for d in sorted(glob.glob("data/sportvu/*")):
        js = glob.glob(f"{d}/*.json")
        if not js:
            continue
        try:
            m = load_match(load_sportvu_trajectories, js[0])
        except Exception:
            continue
        if m:
            matches["sportvu/" + pathlib.Path(d).name] = m
    cache.write_bytes(pickle.dumps(matches))
    print(f"Cached to {cache}", flush=True)
    return matches


def gbm_fit(X, Y):
    return (HistGradientBoostingRegressor(max_iter=300, random_state=0).fit(X, Y[:, 0]),
            HistGradientBoostingRegressor(max_iter=300, random_state=0).fit(X, Y[:, 1]))


def gbm_err(m, X, Y):
    return float(np.median(np.linalg.norm(np.stack([m[0].predict(X), m[1].predict(X)], 1) - Y, axis=1)))


def geo_indomain(matches, cols):
    """Leave-one-match-out in-domain median error (mean over folds)."""
    names = list(matches); errs = []
    for held in names:
        tr = [n for n in names if n != held]
        Xtr = np.concatenate([matches[n]["X"][:, cols] for n in tr]); Ytr = np.concatenate([matches[n]["Y"] for n in tr])
        errs.append(gbm_err(gbm_fit(Xtr, Ytr), matches[held]["X"][:, cols], matches[held]["Y"]))
    return float(np.mean(errs))


def geo_zeroshot(src, tgt, cols):
    Xtr = np.concatenate([m["X"][:, cols] for m in src.values()]); Ytr = np.concatenate([m["Y"] for m in src.values()])
    Xte = np.concatenate([m["X"][:, cols] for m in tgt.values()]); Yte = np.concatenate([m["Y"] for m in tgt.values()])
    return gbm_err(gbm_fit(Xtr, Ytr), Xte, Yte)


# ---- deep, same LOMO in-domain + pooled zero-shot, multi-seed ----
def _frames_cols(frames, use_pos):
    if not use_pos:
        return frames
    keep = [0, 1, 4]  # x, y, team (drop vx, vy) — positions-only for the 5-dim frame
    return [(p[:, keep], b) for p, b in frames]


class DeepSetsN(nn.Module):
    def __init__(self, d_in, d_h=64):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(d_in, d_h), nn.ReLU(), nn.Linear(d_h, d_h), nn.ReLU())
        self.rho = nn.Sequential(nn.Linear(2 * d_h, d_h), nn.ReLU(), nn.Linear(d_h, 2))

    def forward(self, P, M):
        h = self.phi(P); m = M.unsqueeze(-1)
        pooled = torch.cat([(h * m).sum(1) / m.sum(1).clamp(min=1),
                            h.masked_fill(~m.bool(), -1e9).max(1).values], -1)
        return self.rho(pooled)


def collate_n(frames, d_in):
    n = max(len(p) for p, _ in frames)
    P = torch.zeros(len(frames), n, d_in); M = torch.zeros(len(frames), n); Y = torch.zeros(len(frames), 2)
    for i, (p, b) in enumerate(frames):
        P[i, :len(p)] = torch.from_numpy(np.asarray(p, np.float32)); M[i, :len(p)] = 1
        Y[i] = torch.from_numpy(np.asarray(b, np.float32))
    return P, M, Y


def train_deep_n(frames, d_in, seed, max_steps=4000, patience=8, every=200, val_frac=0.1, batch=256):
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    idx = rng.permutation(len(frames)); k = int((1 - val_frac) * len(frames))
    tr = [frames[i] for i in idx[:k]]; va = [frames[i] for i in idx[k:]]
    Pv, Mv, Yv = collate_n(va, d_in)
    m = DeepSetsN(d_in); opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    best, best_state, since = np.inf, None, 0
    for step in range(1, max_steps + 1):
        bi = rng.integers(0, len(tr), batch); P, M, Y = collate_n([tr[j] for j in bi], d_in)
        loss = nn.functional.mse_loss(m(P, M), Y); opt.zero_grad(); loss.backward(); opt.step()
        if step % every == 0:
            with torch.no_grad():
                v = float(nn.functional.mse_loss(m(Pv, Mv), Yv))
            if v < best - 1e-6:
                best, best_state, since = v, {k: x.clone() for k, x in m.state_dict().items()}, 0
            else:
                since += 1
                if since >= patience:
                    break
    if best_state:
        m.load_state_dict(best_state)
    return m


def deep_err_n(m, frames, d_in):
    P, M, Y = collate_n(frames, d_in)
    with torch.no_grad():
        pred = m(P, M).numpy()
    return float(np.median(np.linalg.norm(pred - Y.numpy(), axis=1)))


def deep_indomain(matches, use_pos):
    """LOMO in-domain, mean over folds, mean over seeds."""
    d_in = 3 if use_pos else 5
    names = list(matches); errs = []
    for held in names:
        tr = [f for n in names if n != held for f in _frames_cols(matches[n]["frames"], use_pos)]
        te = _frames_cols(matches[held]["frames"], use_pos)
        errs.append(np.mean([deep_err_n(train_deep_n(tr, d_in, s), te, d_in) for s in SEEDS]))
    return float(np.mean(errs))


def deep_zeroshot(src, tgt, use_pos):
    d_in = 3 if use_pos else 5
    tr = [f for m in src.values() for f in _frames_cols(m["frames"], use_pos)]
    te = [f for m in tgt.values() for f in _frames_cols(m["frames"], use_pos)]
    return float(np.mean([deep_err_n(train_deep_n(tr, d_in, s), te, d_in) for s in SEEDS]))


def report(tag, model_fns, soccer, basket):
    ind, zs = model_fns
    id_s = ind(soccer); id_b = ind(basket)
    s2b = zs(soccer, basket); b2s = zs(basket, soccer)
    pen_s2b = s2b / id_b; pen_b2s = b2s / id_s
    asym = pen_s2b - pen_b2s
    print(f"  {tag}: in-domain soccer {id_s:.3f} basket {id_b:.3f} | "
          f"s->b {s2b:.3f} ({pen_s2b:.2f}x) b->s {b2s:.3f} ({pen_b2s:.2f}x) | asym {asym:+.2f}", flush=True)
    return {"tag": tag, "id_soccer": id_s, "id_basket": id_b, "s2b": s2b, "b2s": b2s,
            "pen_s2b": pen_s2b, "pen_b2s": pen_b2s, "asym": asym}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    soccer = load_cached_matches()
    basket = load_cached_basket()
    print(f"soccer {len(soccer)} / basket {len(basket)}  (unified LOMO protocol, {len(SEEDS)} seeds for deep)", flush=True)
    res = {}
    allcols = list(range(len(FEATURE_NAMES)))
    print("GEO-GBM:")
    res["geo_full"] = report("geo full", (lambda m: geo_indomain(m, allcols), lambda s, t: geo_zeroshot(s, t, allcols)), soccer, basket)
    res["geo_pos"] = report("geo pos-only", (lambda m: geo_indomain(m, POS_COLS), lambda s, t: geo_zeroshot(s, t, POS_COLS)), soccer, basket)
    print("DEEP (tuned, multi-seed):")
    res["deep_full"] = report("deep full", (lambda m: deep_indomain(m, False), lambda s, t: deep_zeroshot(s, t, False)), soccer, basket)
    res["deep_pos"] = report("deep pos-only", (lambda m: deep_indomain(m, True), lambda s, t: deep_zeroshot(s, t, True)), soccer, basket)
    print("\nAsymmetry (penalty s->b minus b->s), same LOMO protocol + same ratio metric:")
    for k in ("deep_full", "deep_pos", "geo_full", "geo_pos"):
        print(f"  {res[k]['tag']:14} asym = {res[k]['asym']:+.2f}")
    print("→ velocity channel carries the asymmetry if full >> pos-only for each model.")
    (OUT / "paper_transfer_unified.json").write_text(json.dumps(res, indent=2))
    print(f"Saved {OUT/'paper_transfer_unified.json'}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "scripts")
    main()
