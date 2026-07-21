"""Phase 0 report — merge Claude-agent predictions, compute errors, viz, verdict.

Reads results/fase0/manifest.json and (optional) results/fase0/claude_preds.json,
computes normalized localization error per system, renders one overlay per item, and
writes results/fase0/report.md + errors.json.

Usage:
  uv run python scripts/fase0_report.py
"""
from __future__ import annotations

import json
import pathlib

import cv2

from wheres_the_ball.eval.metrics import euclidean_norm, summarize
from wheres_the_ball.eval import viz

OUT = pathlib.Path("results/fase0")


def _pred_xy(pred: dict) -> tuple[float, float] | None:
    if not isinstance(pred, dict) or "x" not in pred or "y" not in pred:
        return None
    return float(pred["x"]), float(pred["y"])


def main() -> None:
    manifest = json.loads((OUT / "manifest.json").read_text())
    claude_path = OUT / "claude_preds.json"
    claude = json.loads(claude_path.read_text()) if claude_path.exists() else {}

    viz_dir = OUT / "viz"
    viz_dir.mkdir(exist_ok=True)

    systems = ["center", "gpt", "claude"]
    errors: dict[str, list[float]] = {s: [] for s in systems}
    per_item = []

    for it in manifest["items"]:
        gt = (it["gt"]["x"], it["gt"]["y"])
        preds = dict(it.get("predictions", {}))
        if it["id"] in claude:
            preds["claude"] = claude[it["id"]]

        plot_preds: dict[str, tuple[float, float]] = {}
        row = {"id": it["id"], "gt": gt, "ball_px": it.get("ball_px")}
        for s in systems:
            xy = _pred_xy(preds.get(s, {}))
            if xy is None:
                row[s] = None
                continue
            err = euclidean_norm(xy, gt)
            errors[s].append(err)
            row[s] = round(err, 4)
            plot_preds[s] = xy

        img = cv2.imread(it["masked_path"])
        if img is not None:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            leak = it.get("leak", {})
            flag = " [LEAK]" if leak.get("ball_visible") or leak.get("artifact_visible") else ""
            viz.plot_item(rgb, gt, plot_preds, str(viz_dir / f"{it['id']}.png"),
                          title=f"{it['id']}{flag}")
        per_item.append(row)

    summary = {s: summarize(errors[s]) for s in systems}
    (OUT / "errors.json").write_text(json.dumps({"per_item": per_item, "summary": summary}, indent=2))

    # Markdown report
    lines = ["# Fase 0 — resultados\n",
             f"Dataset: `{manifest['dataset']}` (split {manifest['split']}), "
             f"modelo GPT: `{manifest['deployment']}`, prompt: `{manifest['prompt_version']}`.\n",
             "Error = distancia euclídea normalizada al centro del balón (menor = mejor).\n",
             "## Resumen (mediana [IQR])\n",
             "| Sistema | n | mediana | IQR (q1–q3) | media |",
             "|---|---|---|---|---|"]
    for s in systems:
        d = summary[s]
        if d["n"] == 0:
            lines.append(f"| {s} | 0 | — | — | — |")
        else:
            lines.append(f"| {s} | {d['n']} | {d['median']:.3f} | "
                         f"{d['q1']:.3f}–{d['q3']:.3f} | {d['mean']:.3f} |")
    lines += ["\n## Por ítem\n",
              "| id | ball px | center | gpt | claude |",
              "|---|---|---|---|---|"]
    for r in per_item:
        def f(v):
            return "—" if v is None else f"{v:.3f}"
        lines.append(f"| {r['id']} | {r['ball_px']} | {f(r['center'])} | {f(r['gpt'])} | {f(r['claude'])} |")

    n_leak = sum(1 for it in manifest["items"]
                 if (it.get("leak") or {}).get("ball_visible") or (it.get("leak") or {}).get("artifact_visible"))
    lines += [f"\nÍtems marcados por control de fuga: **{n_leak}/{len(manifest['items'])}**.",
              "\nVisualizaciones por ítem en `results/fase0/viz/`."]
    (OUT / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nWrote {OUT/'report.md'} and {OUT/'errors.json'}")


if __name__ == "__main__":
    main()
