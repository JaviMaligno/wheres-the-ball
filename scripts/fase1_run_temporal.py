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
    ap.add_argument("--claude-key", default="claude")
    ap.add_argument("--skip-gpt", action="store_true")
    ap.add_argument("--variant", default="normal", choices=["normal", "shuffled", "lastonly"],
                    help="ablation: shuffled = same frames, random order (kills motion, "
                         "keeps views); lastonly = only the target frame in sequence "
                         "format (isolates the multi-image format effect)")
    args = ap.parse_args()

    _ensure_anthropic_key()
    from wheres_the_ball.models import anthropic_claude, azure_gpt
    from wheres_the_ball.models.base import parse_prediction
    from wheres_the_ball.prompts.localize import PROMPTS

    pred_path = PRED if args.variant == "normal" else OUT / f"predictions_temporal_{args.variant}.json"
    items = json.loads((OUT / "temporal_manifest.json").read_text())["items"]
    if args.limit:
        items = items[: args.limit]
    preds = json.loads(pred_path.read_text()) if pred_path.exists() and not args.overwrite else {}

    def vlm(fn, frames, **kw):
        try:
            return parse_prediction(fn(frames, PROMPTS["temporal"], **kw)).model_dump()
        except Exception as e:  # noqa: BLE001
            return {"error": f"{type(e).__name__}: {e}"}

    def needs(rec, key):
        return key not in rec or (isinstance(rec.get(key), dict) and "error" in rec[key])

    import random

    for i, it in enumerate(items):
        rec = preds.get(it["id"], {})
        frames = [f for f in it["frames"] if pathlib.Path(f).exists()]
        # Ablation transforms (the LAST frame is always the prediction target):
        if args.variant == "shuffled" and len(frames) >= 2:
            # Shuffle the HISTORY frames only, deterministically per item, keeping the
            # target last — kills motion coherence while preserving the views.
            hist = frames[:-1]
            random.Random(it["id"]).shuffle(hist)
            frames = hist + [frames[-1]]
        elif args.variant == "lastonly" and frames:
            # Only the target frame, but still through the sequence prompt/format.
            frames = [frames[-1]]
        rec["n_frames"] = len(frames)
        min_frames = 1 if args.variant == "lastonly" else 2
        if len(frames) < min_frames:
            rec["skip"] = "not enough frames"
        else:
            if not args.skip_gpt and needs(rec, "gpt"):
                rec["gpt"] = vlm(azure_gpt.localize_sequence, frames, deployment=args.gpt_deployment)
            if needs(rec, args.claude_key):
                rec[args.claude_key] = vlm(anthropic_claude.localize_sequence, frames, model=args.claude_model)
        preds[it["id"]] = rec
        pred_path.write_text(json.dumps(preds, indent=2))
        g = (rec.get("gpt") or {}).get("x"); c = (rec.get(args.claude_key) or {}).get("x")
        print(f"[{i+1}/{len(items)}] {it['id']} ({it['center_bin']}) nf={len(frames)} "
              f"gpt={'%.2f'%g if g is not None else '—'} claude={'%.2f'%c if c is not None else '—'}")

    print(f"\nWrote {pred_path} ({len(preds)} items)")


if __name__ == "__main__":
    main()
