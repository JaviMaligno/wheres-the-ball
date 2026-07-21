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
CENTER_BINS = ["near", "mid", "far"]
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
    errors_clean = {s: [] for s in SYSTEMS}  # excluding leak-flagged items
    by_stratum = {s: {st: [] for st in STRATA} for s in SYSTEMS}
    by_center = {s: {cb: [] for cb in CENTER_BINS} for s in SYSTEMS}
    beats_center = {s: [] for s in SYSTEMS}          # per-item: model err < center err
    beats_center_far = {s: [] for s in SYSTEMS}      # only off-center (far) items
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
        cbin = it.get("center_bin", "?")
        center_err = it.get("center_dist", euclidean_norm((0.5, 0.5), gt))
        row = {"id": it["id"], "state": it["state"], "center_bin": cbin, "leak": is_leak}
        plot_preds = {}
        for s in SYSTEMS:
            xy = _xy(pr.get(s))
            if xy is None:
                row[s] = None
                continue
            err = euclidean_norm(xy, gt)
            errors[s].append(err)
            if not is_leak:
                errors_clean[s].append(err)
            by_stratum[s][it["state"]].append(err)
            if cbin in by_center[s]:
                by_center[s][cbin].append(err)
            if s != "center":
                beats_center[s].append(err < center_err)
                if cbin == "far":
                    beats_center_far[s].append(err < center_err)
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
    L += [f"\n## Global sin ítems con fuga (n={manifest['n']-leaked})\n",
          "| Sistema | n | mediana | IQR | media |",
          "|---|---|---|---|---|"]
    for s in SYSTEMS:
        d = summarize(errors_clean[s])
        if d["n"] == 0:
            L.append(f"| {s} | 0 |—|—|—|"); continue
        L.append(f"| {s} | {d['n']} | {d['median']:.3f} | {d['q1']:.3f}–{d['q3']:.3f} | {d['mean']:.3f} |")
    L += ["\n## Mediana de error por estrato\n",
          "| Sistema | " + " | ".join(STRATA) + " |",
          "|---|" + "---|" * len(STRATA)]
    for s in SYSTEMS:
        cells = []
        for st in STRATA:
            e = by_stratum[s][st]
            cells.append(f"{summarize(e)['median']:.3f} (n={len(e)})" if e else "—")
        L.append(f"| {s} | " + " | ".join(cells) + " |")
    L += ["\n## Mediana de error por distancia al centro (de-sesgo cámara)\n",
          "| Sistema | " + " | ".join(CENTER_BINS) + " |",
          "|---|" + "---|" * len(CENTER_BINS)]
    for s in SYSTEMS:
        cells = []
        for cb in CENTER_BINS:
            e = by_center[s][cb]
            cells.append(f"{summarize(e)['median']:.3f} (n={len(e)})" if e else "—")
        L.append(f"| {s} | " + " | ".join(cells) + " |")

    L += ["\n## ¿Bate al baseline 'centro' ítem a ítem? (win-rate pareado)\n",
          "| Sistema | global | solo 'far' (descentrados) |",
          "|---|---|---|"]
    for s in SYSTEMS:
        if s == "center":
            continue
        g = beats_center[s]; f = beats_center_far[s]
        gr = f"{sum(g)}/{len(g)} ({100*sum(g)/len(g):.0f}%)" if g else "—"
        fr = f"{sum(f)}/{len(f)} ({100*sum(f)/len(f):.0f}%)" if f else "—"
        L.append(f"| {s} | {gr} | {fr} |")

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
