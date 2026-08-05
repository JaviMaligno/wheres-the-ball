"""Paper — scaled VLM benchmark analysis (de-biased SoccerNet set, n=260).

Reads results/fase1/{manifest,predictions}.json and reports, per model and per
center-distance bin (near/mid/far), with bootstrap 95% CIs:
  - median localization error (normalized)
  - win-rate vs the camera-center baseline (fraction of items closer than center=(.5,.5))
  - correlation of predicted vs ground-truth position (x and y)

The headline test is the FAR bin: off-center balls where the camera-center baseline is
weak. A model with genuine spatial inference beats center there; one that just guesses
center does not.

Usage: uv run python scripts/paper_vlm_benchmark.py
"""
from __future__ import annotations

import json
import pathlib

import numpy as np

OUT = pathlib.Path("results/fase1")
MODELS = [("gpt", "GPT-5.4"), ("llama4", "Llama-4-Maverick"),
          ("claude_opus", "Claude Opus 4.8"), ("claude", "Claude Sonnet 4.6")]
rng = np.random.default_rng(0)


def xy(p):
    return None if not isinstance(p, dict) or "x" not in p else np.array([p["x"], p["y"]], float)


def boot_ci(vals, fn, n=5000):
    vals = np.asarray(vals, float)
    if len(vals) < 3:
        return (float("nan"), float("nan"))
    stats = [fn(vals[rng.integers(0, len(vals), len(vals))]) for _ in range(n)]
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def main() -> None:
    manifest = {it["id"]: it for it in json.loads((OUT / "manifest.json").read_text())["items"]}
    preds = json.loads((OUT / "predictions.json").read_text())
    bins = ["near", "mid", "far"]

    print(f"{'model':20}{'bin':6}{'n':>5}{'med err':>16}{'win vs center':>20}{'corr_x':>9}{'corr_y':>9}")
    summary = {}
    for key, name in MODELS:
        summary[key] = {}
        for b in bins + ["all"]:
            errs, wins, px, py, gx, gy = [], [], [], [], [], []
            for iid, rec in preds.items():
                it = manifest.get(iid)
                if it is None or (b != "all" and it["center_bin"] != b):
                    continue
                p = xy(rec.get(key));
                if p is None:
                    continue
                gt = np.array([it["gt"]["x"], it["gt"]["y"]])
                cen = np.array([0.5, 0.5])
                errs.append(float(np.linalg.norm(p - gt)))
                wins.append(np.linalg.norm(p - gt) < np.linalg.norm(cen - gt))
                px.append(p[0]); py.append(p[1]); gx.append(gt[0]); gy.append(gt[1])
            if len(errs) < 3:
                continue
            med = float(np.median(errs)); med_ci = boot_ci(errs, np.median)
            wr = float(np.mean(wins)); wr_ci = boot_ci(np.array(wins, float), np.mean)
            cx = float(np.corrcoef(px, gx)[0, 1]) if len(px) > 2 else float("nan")
            cy = float(np.corrcoef(py, gy)[0, 1]) if len(py) > 2 else float("nan")
            summary[key][b] = {"n": len(errs), "med": med, "med_ci": med_ci,
                               "win": wr, "win_ci": wr_ci, "corr_x": cx, "corr_y": cy}
            print(f"{name:20}{b:6}{len(errs):>5}{med:>8.3f} [{med_ci[0]:.2f},{med_ci[1]:.2f}]"
                  f"{wr:>10.0%} [{wr_ci[0]:.0%},{wr_ci[1]:.0%}]{cx:>9.2f}{cy:>9.2f}")
        print()

    # center baseline reference in the far bin (win-rate is by definition 0 vs itself; show error)
    far_items = [manifest[i] for i in preds if manifest.get(i) and manifest[i]["center_bin"] == "far"]
    cen_err = np.median([np.linalg.norm([0.5 - it["gt"]["x"], 0.5 - it["gt"]["y"]]) for it in far_items])
    print(f"Reference: camera-center baseline error in FAR bin = {cen_err:.3f} (n={len(far_items)})")

    # --- leak control: flag rate + far-bin robustness EXCLUDING leak-flagged items ---
    def flagged(rec):
        lk = rec.get("leak")
        return isinstance(lk, dict) and (lk.get("ball_visible") or lk.get("artifact_visible"))
    n_flag = sum(1 for r in preds.values() if flagged(r))
    print(f"\nLeak control: {n_flag}/{len(preds)} = {n_flag/len(preds):.1%} items flagged (ball/artifact visible)")
    print("Far-bin win-rate excluding leak-flagged items:")
    leak_excl = {}
    for key, name in MODELS:
        wins = []
        for iid, rec in preds.items():
            it = manifest.get(iid)
            if it is None or it["center_bin"] != "far" or flagged(rec):
                continue
            p = xy(rec.get(key))
            if p is None:
                continue
            gt = np.array([it["gt"]["x"], it["gt"]["y"]])
            wins.append(np.linalg.norm(p - gt) < np.linalg.norm([0.5, 0.5] - gt))
        if len(wins) >= 3:
            wr = float(np.mean(wins))
            leak_excl[key] = {"n": len(wins), "win": wr}
            print(f"  {name:20} n={len(wins):>4}  win {wr:.0%}  (all-items {summary[key]['far']['win']:.0%})")
    summary["_leak"] = {"flag_rate": n_flag / len(preds), "n_flagged": n_flag, "far_excl_leak": leak_excl}
    (OUT / "paper_vlm_benchmark.json").write_text(json.dumps(summary, indent=2))
    print(f"Saved {OUT/'paper_vlm_benchmark.json'}")


if __name__ == "__main__":
    main()
