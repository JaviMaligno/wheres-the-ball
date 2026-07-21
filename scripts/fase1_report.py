"""Phase 1 report — errors overall and per ball-state stratum, PCK, possessor
accuracy, leak rate, plus example overlays.

Reads results/fase1/{manifest.json, predictions.json}; writes results/fase1/{report.md,
errors.json, viz/*}.

Usage:
  uv run python scripts/fase1_report.py
"""
from __future__ import annotations

import json
import pathlib

import cv2

from wheres_the_ball.baselines.geometric import nearest_player
from wheres_the_ball.eval import viz
from wheres_the_ball.eval.metrics import euclidean_norm, summarize

OUT = pathlib.Path("results/fase1")
SYSTEMS = ["center", "centroid", "gpt", "claude"]
STRATA = ["possession", "short_pass", "long_pass", "contested"]
PCK = [0.05, 0.10, 0.15]


def _xy(p):
    if isinstance(p, dict) and "x" in p and "y" in p and p.get("x") is not None:
        return float(p["x"]), float(p["y"])
    return None


def main() -> None:
    manifest = json.loads((OUT / "manifest.json").read_text())
    preds = json.loads((OUT / "predictions.json").read_text())
    viz_dir = OUT / "viz"; viz_dir.mkdir(exist_ok=True)

    errors = {s: [] for s in SYSTEMS}
    by_stratum = {s: {st: [] for st in STRATA} for s in SYSTEMS}
    possessor_hit = {s: [] for s in SYSTEMS}
    leaked = 0
    rows = []

    for it in manifest["items"]:
        gt = (it["gt"]["x"], it["gt"]["y"])
        pr = preds.get(it["id"], {})
        leak = pr.get("leak", {})
        is_leak = bool(leak.get("ball_visible") or leak.get("artifact_visible"))
        leaked += is_leak
        gt_poss = nearest_player(gt, it["players"])
        row = {"id": it["id"], "state": it["state"], "leak": is_leak}
        plot_preds = {}
        for s in SYSTEMS:
            xy = _xy(pr.get(s))
            if xy is None:
                row[s] = None
                continue
            err = euclidean_norm(xy, gt)
            errors[s].append(err)
            by_stratum[s][it["state"]].append(err)
            row[s] = round(err, 4)
            plot_preds[s] = xy
            if gt_poss is not None:
                possessor_hit[s].append(nearest_player(xy, it["players"]) == gt_poss)
        rows.append(row)
        img = cv2.imread(it["masked_path"])
        if img is not None:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            viz.plot_item(rgb, gt, plot_preds, str(viz_dir / f"{it['id']}.png"),
                          title=f"{it['id']} [{it['state']}]{' LEAK' if is_leak else ''}")

    summary = {s: summarize(errors[s]) for s in SYSTEMS}

    def pck(errs, r):
        return sum(1 for e in errs if e < r) / len(errs) if errs else float("nan")

    # ---- markdown ----
    L = ["# Fase 1 — resultados (SoccerNet-Tracking, LaMa inpainting)\n",
         f"Ítems: {manifest['n']} · estratos {manifest['counts']} · "
         f"enmascarado LaMa · control de fuga marcó **{leaked}/{manifest['n']}**.\n",
         "Error = distancia euclídea normalizada al balón (menor = mejor).\n",
         "## Global (mediana [IQR])\n",
         "| Sistema | n | mediana | IQR | media | PCK@.05 | PCK@.10 | PCK@.15 | poseedor |",
         "|---|---|---|---|---|---|---|---|---|"]
    for s in SYSTEMS:
        d = summary[s]
        if d["n"] == 0:
            L.append(f"| {s} | 0 |—|—|—|—|—|—|—|"); continue
        ph = sum(possessor_hit[s]) / len(possessor_hit[s]) if possessor_hit[s] else float("nan")
        L.append(f"| {s} | {d['n']} | {d['median']:.3f} | {d['q1']:.3f}–{d['q3']:.3f} | "
                 f"{d['mean']:.3f} | {pck(errors[s],.05):.2f} | {pck(errors[s],.10):.2f} | "
                 f"{pck(errors[s],.15):.2f} | {ph:.2f} |")
    L += ["\n## Mediana de error por estrato\n",
          "| Sistema | " + " | ".join(STRATA) + " |",
          "|---|" + "---|" * len(STRATA)]
    for s in SYSTEMS:
        cells = []
        for st in STRATA:
            e = by_stratum[s][st]
            cells.append(f"{summarize(e)['median']:.3f} (n={len(e)})" if e else "—")
        L.append(f"| {s} | " + " | ".join(cells) + " |")
    L.append(f"\nVisualizaciones por ítem en `{viz_dir}/`.")

    (OUT / "report.md").write_text("\n".join(L) + "\n")
    (OUT / "errors.json").write_text(json.dumps(
        {"summary": summary, "leaked": leaked, "rows": rows,
         "possessor_acc": {s: (sum(v)/len(v) if v else None) for s, v in possessor_hit.items()}},
        indent=2))
    print("\n".join(L))
    print(f"\nWrote {OUT/'report.md'} + errors.json")


if __name__ == "__main__":
    main()
