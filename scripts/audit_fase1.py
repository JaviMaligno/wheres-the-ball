"""Independent audit of the Phase 1 numbers — recompute from raw JSON, stress-test.

Does NOT reuse the report code. Checks:
  1. Recompute far-bin medians + paired win-rate (single & temporal), cross-check.
  2. Bootstrap 95% CIs (n per bin is small → are the claims robust?).
  3. Degeneracy: does each model actually vary predictions, or just guess center?
  4. Isotropic pixel metric (our normalized dist mixes x/W and y/H — anisotropic).
  5. Exclude leak-flagged items and recheck the far comparison.

Usage: uv run python scripts/audit_fase1.py
"""
from __future__ import annotations

import json
import math
import pathlib

import numpy as np

OUT = pathlib.Path("results/fase1")
rng = np.random.default_rng(0)


def load():
    man = {i["id"]: i for i in json.loads((OUT / "manifest.json").read_text())["items"]}
    single = json.loads((OUT / "predictions.json").read_text())
    temporal = json.loads((OUT / "predictions_temporal.json").read_text())
    return man, single, temporal


def xy(p):
    if isinstance(p, dict) and p.get("x") is not None and "error" not in p:
        return float(p["x"]), float(p["y"])
    return None


def err_norm(p, gt):
    return math.hypot(p[0] - gt[0], p[1] - gt[1])


def err_px_over_w(p, gt, W, H):
    # isotropic: real pixel distance, expressed as fraction of image width
    return math.hypot((p[0] - gt[0]) * W, (p[1] - gt[1]) * H) / W


def boot_median(vals, n=10000):
    if len(vals) < 2:
        return (float("nan"), float("nan"))
    a = np.array(vals)
    meds = np.median(a[rng.integers(0, len(a), size=(n, len(a)))], axis=1)
    return float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def boot_rate(bools, n=10000):
    if not bools:
        return (float("nan"), float("nan"))
    a = np.array(bools, float)
    r = a[rng.integers(0, len(a), size=(n, len(a)))].mean(axis=1)
    return float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))


def main():
    man, single, temporal = load()
    SYS = ["center", "centroid", "gpt", "claude", "claude_opus"]

    def pred(preds, iid, s):
        it = man[iid]
        if s == "center":
            return (0.5, 0.5)
        if s == "centroid":
            ps = it["players"]
            if not ps:
                return None
            return (sum(p["x"] for p in ps) / len(ps), sum(p["y"] for p in ps) / len(ps))
        return xy(preds.get(iid, {}).get(s))

    def is_leak(iid):
        lk = single.get(iid, {}).get("leak", {}) or {}
        return bool(lk.get("ball_visible") or lk.get("artifact_visible"))

    # ---- 1+2: far-bin medians with bootstrap CI (single), normalized + isotropic ----
    print("=== FAR bin (off-center) — single frame ===")
    print(f"{'system':12} n  med_norm  95%CI_norm        med_pxW   win_vs_center  CI")
    far_ids = [i for i in man if man[i]["center_bin"] == "far"]
    for s in SYS:
        en, ep, wins = [], [], []
        for iid in far_ids:
            p = pred(single, iid, s); gt = (man[iid]["gt"]["x"], man[iid]["gt"]["y"])
            if p is None:
                continue
            en.append(err_norm(p, gt))
            ep.append(err_px_over_w(p, gt, man[iid]["width"], man[iid]["height"]))
            if s != "center":
                cen = err_norm((0.5, 0.5), gt)
                wins.append(err_norm(p, gt) < cen)
        if not en:
            continue
        lo, hi = boot_median(en)
        wr = f"{sum(wins)}/{len(wins)}" if wins else "—"
        wlo, whi = boot_rate(wins)
        print(f"{s:12} {len(en):2} {np.median(en):.3f}   [{lo:.3f},{hi:.3f}]   "
              f"{np.median(ep):.3f}   {wr:>7}  [{wlo:.2f},{whi:.2f}]")

    # ---- 3: degeneracy — spread of predictions & distance to center ----
    print("\n=== Degeneración de predicciones (todos los ítems, single) ===")
    print(f"{'system':12} n  std_x  std_y  %pred cerca centro(<.05)  corr(x,gtx) corr(y,gty)")
    for s in ["gpt", "claude", "claude_opus", "center"]:
        P, G = [], []
        for iid in man:
            p = pred(single, iid, s)
            if p is None:
                continue
            P.append(p); G.append((man[iid]["gt"]["x"], man[iid]["gt"]["y"]))
        P, G = np.array(P), np.array(G)
        if len(P) < 2:
            continue
        near = np.mean([math.hypot(px - .5, py - .5) < .05 for px, py in P])
        cx = np.corrcoef(P[:, 0], G[:, 0])[0, 1] if np.std(P[:, 0]) > 1e-9 else float("nan")
        cy = np.corrcoef(P[:, 1], G[:, 1])[0, 1] if np.std(P[:, 1]) > 1e-9 else float("nan")
        print(f"{s:12} {len(P):2} {P[:,0].std():.3f}  {P[:,1].std():.3f}   {near*100:4.0f}%"
              f"                 {cx:+.2f}       {cy:+.2f}")

    # ---- 5: far comparison excluding leak-flagged items ----
    print("\n=== FAR bin excluyendo ítems con fuga ===")
    far_clean = [i for i in far_ids if not is_leak(i)]
    print(f"far total={len(far_ids)}, sin fuga={len(far_clean)}")
    for s in SYS:
        en = [err_norm(pred(single, i, s), (man[i]["gt"]["x"], man[i]["gt"]["y"]))
              for i in far_clean if pred(single, i, s) is not None]
        if en:
            print(f"  {s:12} n={len(en)} med={np.median(en):.3f}")

    # ---- temporal: GPT far improvement, paired, with CI on the paired delta ----
    print("\n=== Temporal en FAR: mejora pareada (single - temporal) por modelo ===")
    for s in ["gpt", "claude", "claude_opus"]:
        deltas = []
        for iid in far_ids:
            ps = xy(single.get(iid, {}).get(s)); pt = xy((temporal.get(iid, {}) or {}).get(s))
            gt = (man[iid]["gt"]["x"], man[iid]["gt"]["y"])
            if ps and pt:
                deltas.append(err_norm(ps, gt) - err_norm(pt, gt))  # >0 = temporal better
        if deltas:
            lo, hi = boot_median(deltas)
            print(f"  {s:12} n={len(deltas)} median Δ={np.median(deltas):+.3f} "
                  f"[{lo:+.3f},{hi:+.3f}]  (>0: temporal mejora)")


if __name__ == "__main__":
    main()
