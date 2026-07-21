"""Phase 1 RQ2 report — single-frame vs temporal (sequence), paired on the same items.

Reads manifest.json, predictions.json (single), predictions_temporal.json (temporal).
Reports per-model median error single vs temporal, per center-bin, paired improvement,
and whether temporal beats the center baseline on off-center (far) balls.

Usage:
  uv run python scripts/fase1_report_temporal.py
"""
from __future__ import annotations

import json
import pathlib

from wheres_the_ball.eval.metrics import euclidean_norm, summarize

OUT = pathlib.Path("results/fase1")
MODELS = ["gpt", "claude"]
CENTER_BINS = ["near", "mid", "far"]


def _xy(p):
    if isinstance(p, dict) and p.get("x") is not None:
        return float(p["x"]), float(p["y"])
    return None


def main() -> None:
    manifest = {i["id"]: i for i in json.loads((OUT / "manifest.json").read_text())["items"]}
    single = json.loads((OUT / "predictions.json").read_text())
    temporal = json.loads((OUT / "predictions_temporal.json").read_text())

    def err(pred, gt):
        xy = _xy(pred)
        return euclidean_norm(xy, gt) if xy else None

    L = ["# Fase 1 · RQ2 — frame único vs. temporal (secuencia ~t-3s…t)\n",
         "Comparación pareada sobre los mismos ítems (dataset de-sesgado por distancia "
         "al centro). Error = distancia euclídea normalizada; menor = mejor.\n"]

    agg = {m: {"single": [], "temporal": [], "paired_better": [],
               "far_single": [], "far_temporal": []} for m in MODELS}
    by_bin = {m: {c: {"single": [], "temporal": []} for c in CENTER_BINS} for m in MODELS}

    for iid, it in manifest.items():
        gt = (it["gt"]["x"], it["gt"]["y"])
        cbin = it["center_bin"]
        for m in MODELS:
            es = err(single.get(iid, {}).get(m), gt)
            et = err((temporal.get(iid, {}) or {}).get(m), gt)
            if es is not None:
                agg[m]["single"].append(es); by_bin[m][cbin]["single"].append(es)
            if et is not None:
                agg[m]["temporal"].append(et); by_bin[m][cbin]["temporal"].append(et)
            if es is not None and et is not None:
                agg[m]["paired_better"].append(et < es)
            if cbin == "far":
                if es is not None:
                    agg[m]["far_single"].append(es)
                if et is not None:
                    agg[m]["far_temporal"].append(et)

    L += ["## Global (mediana de error)\n",
          "| Modelo | single | temporal | Δ (temporal−single) | temporal mejora en |",
          "|---|---|---|---|---|"]
    for m in MODELS:
        ms = summarize(agg[m]["single"])["median"]
        mt = summarize(agg[m]["temporal"])["median"]
        pb = agg[m]["paired_better"]
        pbr = f"{sum(pb)}/{len(pb)} ({100*sum(pb)/len(pb):.0f}%)" if pb else "—"
        L.append(f"| {m} | {ms:.3f} | {mt:.3f} | {mt-ms:+.3f} | {pbr} |")

    L += ["\n## Balones descentrados (`far`) — mediana de error\n",
          "| Modelo | single | temporal |", "|---|---|---|"]
    for m in MODELS:
        s = summarize(agg[m]["far_single"])["median"]
        t = summarize(agg[m]["far_temporal"])["median"]
        L.append(f"| {m} | {s:.3f} | {t:.3f} |")
    # center baseline on far, for reference
    far_center = [it["center_dist"] for it in manifest.values() if it["center_bin"] == "far"]
    L.append(f"\n(baseline centro en `far`: {summarize(far_center)['median']:.3f})")

    L += ["\n## Por distancia al centro (mediana single → temporal)\n",
          "| Modelo | near | mid | far |", "|---|---|---|---|"]
    for m in MODELS:
        cells = []
        for c in CENTER_BINS:
            s = by_bin[m][c]["single"]; t = by_bin[m][c]["temporal"]
            sm = f"{summarize(s)['median']:.3f}" if s else "—"
            tm = f"{summarize(t)['median']:.3f}" if t else "—"
            cells.append(f"{sm}→{tm}")
        L.append(f"| {m} | " + " | ".join(cells) + " |")

    (OUT / "report_temporal.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nWrote {OUT/'report_temporal.md'}")


if __name__ == "__main__":
    main()
