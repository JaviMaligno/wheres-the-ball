"""Phase 1 RQ2 harness — run GPT + Claude on masked frame SEQUENCES (temporal).

Reads results/fase1/temporal_manifest.json (ordered masked-frame paths per item,
produced by fase1_build_temporal.py + inpaint_lama.py --root results/fase1/clips).
Writes results/fase1/predictions_temporal.json incrementally.

Usage:
  source ../CooperBench/azure_env.sh
  uv run python scripts/fase1_run_temporal.py [--limit N] [--overwrite]
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib

OUT = pathlib.Path("results/fase1")
PRED = OUT / "predictions_temporal.json"
FALLBACK_ENV = pathlib.Path("../llm-language-limits/.env")


def _ensure_anthropic_key() -> None:
    if os.environ.get("ANTHROPIC_API_KEY"):
        return
    if FALLBACK_ENV.exists():
        for line in FALLBACK_ENV.read_text().splitlines():
            if line.startswith("ANTHROPIC_API_KEY="):
                os.environ["ANTHROPIC_API_KEY"] = line.split("=", 1)[1].strip()
                return


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--gpt-deployment", default="gpt-5.4")
    ap.add_argument("--claude-model", default="claude-sonnet-4-6")
    args = ap.parse_args()

    _ensure_anthropic_key()
    from wheres_the_ball.models import anthropic_claude, azure_gpt
    from wheres_the_ball.models.base import parse_prediction
    from wheres_the_ball.prompts.localize import PROMPTS

    items = json.loads((OUT / "temporal_manifest.json").read_text())["items"]
    if args.limit:
        items = items[: args.limit]
    preds = json.loads(PRED.read_text()) if PRED.exists() and not args.overwrite else {}

    def vlm(fn, frames, **kw):
        try:
            return parse_prediction(fn(frames, PROMPTS["temporal"], **kw)).model_dump()
        except Exception as e:  # noqa: BLE001
            return {"error": f"{type(e).__name__}: {e}"}

    for i, it in enumerate(items):
        rec = preds.get(it["id"], {})
        frames = [f for f in it["frames"] if pathlib.Path(f).exists()]
        rec["n_frames"] = len(frames)
        if len(frames) < 2:
            rec["skip"] = "fewer than 2 frames"
        else:
            if "gpt" not in rec:
                rec["gpt"] = vlm(azure_gpt.localize_sequence, frames, deployment=args.gpt_deployment)
            if "claude" not in rec:
                rec["claude"] = vlm(anthropic_claude.localize_sequence, frames, model=args.claude_model)
        preds[it["id"]] = rec
        PRED.write_text(json.dumps(preds, indent=2))
        g = (rec.get("gpt") or {}).get("x"); c = (rec.get("claude") or {}).get("x")
        print(f"[{i+1}/{len(items)}] {it['id']} ({it['center_bin']}) nf={len(frames)} "
              f"gpt={'%.2f'%g if g is not None else '—'} claude={'%.2f'%c if c is not None else '—'}")

    print(f"\nWrote {PRED} ({len(preds)} items)")


if __name__ == "__main__":
    main()
