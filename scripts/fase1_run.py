"""Phase 1 harness — run leak-control + GPT + Claude + geometric baselines.

Operates on the LaMa-inpainted SoccerNet items (results/fase1/manifest.json).
Writes predictions incrementally to results/fase1/predictions.json so a crash keeps
progress. Re-running skips items already predicted (unless --overwrite).

Models:
  - center   : (0.5, 0.5) broadcast-bias baseline (B0)
  - centroid : centroid of GT player positions (B1)
  - gpt      : Azure gpt-5.4, neutral prompt
  - claude   : claude-sonnet-4-6, neutral prompt
  - leak     : GPT leak-control on the masked image

Usage:
  source ../CooperBench/azure_env.sh
  uv run python scripts/fase1_run.py [--limit N] [--overwrite] [--no-leak]
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib

OUT = pathlib.Path("results/fase1")
PRED = OUT / "predictions.json"
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
    ap.add_argument("--limit", type=int, default=0, help="only first N items (0=all)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--no-leak", action="store_true")
    ap.add_argument("--gpt-deployment", default="gpt-5.4")
    ap.add_argument("--claude-model", default="claude-sonnet-4-6")
    ap.add_argument("--claude-key", default="claude", help="prediction key for this Claude model")
    ap.add_argument("--gpt-key", default="gpt", help="prediction key for GPT")
    ap.add_argument("--prompt", default="neutral", choices=["neutral", "informed"],
                    help="prompt variant (RQ3: informed names the sport + tactics)")
    ap.add_argument("--skip-gpt", action="store_true", help="don't (re)run GPT")
    ap.add_argument("--skip-claude", action="store_true", help="don't (re)run Claude")
    args = ap.parse_args()

    _ensure_anthropic_key()
    from wheres_the_ball.baselines.geometric import center_of_frame, centroid
    from wheres_the_ball.models import anthropic_claude, azure_gpt
    from wheres_the_ball.models.base import parse_prediction
    from wheres_the_ball.prompts.localize import LEAK_CHECK, PROMPTS

    manifest = json.loads((OUT / "manifest.json").read_text())
    items = manifest["items"]
    if args.limit:
        items = items[: args.limit]

    preds = json.loads(PRED.read_text()) if PRED.exists() and not args.overwrite else {}

    def vlm(fn, img, prompt, **kw):
        try:
            return parse_prediction(fn(img, prompt, **kw)).model_dump()
        except Exception as e:  # noqa: BLE001
            return {"error": f"{type(e).__name__}: {e}"}

    def needs(rec, key):  # missing, or a previous error → retry
        return key not in rec or (isinstance(rec.get(key), dict) and "error" in rec[key])

    for i, it in enumerate(items):
        rec = preds.get(it["id"], {})
        img = it["masked_path"]

        rec.setdefault("center", dict(zip("xy", center_of_frame())))
        cen = centroid(it["players"])
        if cen:
            rec.setdefault("centroid", {"x": round(cen[0], 4), "y": round(cen[1], 4)})

        if not args.no_leak and "leak" not in rec:
            try:
                raw = azure_gpt.localize(img, LEAK_CHECK, args.gpt_deployment)
                rec["leak"] = json.loads(
                    raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
                )
            except Exception as e:  # noqa: BLE001
                rec["leak"] = {"error": str(e)}

        prompt = PROMPTS[args.prompt]
        if not args.skip_gpt and needs(rec, args.gpt_key):
            rec[args.gpt_key] = vlm(azure_gpt.localize, img, prompt, deployment=args.gpt_deployment)
        if not args.skip_claude and needs(rec, args.claude_key):
            rec[args.claude_key] = vlm(anthropic_claude.localize, img, prompt, model=args.claude_model)

        preds[it["id"]] = rec
        PRED.write_text(json.dumps(preds, indent=2))  # incremental save
        g = (rec.get(args.gpt_key) or {}).get("x"); c = (rec.get(args.claude_key) or {}).get("x")
        leak = rec.get("leak", {})
        flag = "LEAK" if leak.get("ball_visible") or leak.get("artifact_visible") else "ok"
        print(f"[{i+1}/{len(items)}] {it['id']} ({it['state']}) "
              f"gpt={'%.2f'%g if g is not None else 'ERR'} "
              f"claude={'%.2f'%c if c is not None else 'ERR'} leak={flag}")

    print(f"\nWrote {PRED} ({len(preds)} items)")


if __name__ == "__main__":
    main()
