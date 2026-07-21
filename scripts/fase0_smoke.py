"""Phase 0 smoke test — build masked items, run GPT + center baseline, write manifest.

Pipeline:
  1. Download N samples from martinjolif/football-ball-detection (ball bbox = GT).
  2. Remove the ball by inpainting (bbox as mask); save masked frame.
  3. Leak control: ask the VLM whether a ball / edit artifact is still visible.
  4. Predict with Azure GPT (neutral prompt) + center-of-frame baseline.
  5. Write results/fase0/manifest.json and masked images.

Claude's predictions are produced separately by the Claude Code agent reading the
masked images (no API key in Phase 0); scripts/fase0_report.py merges them.

Usage:
  source ../CooperBench/azure_env.sh
  uv run python scripts/fase0_smoke.py --n 12 [--no-gpt] [--no-leak-check]
"""
from __future__ import annotations

import argparse
import json
import pathlib

import cv2

from wheres_the_ball.baselines.geometric import center_of_frame
from wheres_the_ball.data.football_ball_detection import load_samples
from wheres_the_ball.masking.inpaint import BBox, inpaint_ball
from wheres_the_ball.models import azure_gpt
from wheres_the_ball.models.base import parse_prediction
from wheres_the_ball.prompts.localize import LEAK_CHECK, PROMPTS

OUT = pathlib.Path("results/fase0")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--split", default="test")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--deployment", default="gpt-5.4")
    ap.add_argument("--dilate-frac", type=float, default=0.6)
    ap.add_argument("--no-gpt", action="store_true", help="skip GPT (build + baseline only)")
    ap.add_argument("--no-leak-check", action="store_true")
    args = ap.parse_args()

    masked_dir = OUT / "masked"
    orig_dir = OUT / "original"
    masked_dir.mkdir(parents=True, exist_ok=True)
    orig_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {args.n} samples from split={args.split} …")
    samples = load_samples(args.n, split=args.split, seed=args.seed)
    print(f"Got {len(samples)} samples with a ball box.")

    items = []
    for i, s in enumerate(samples):
        img = cv2.imread(s.image_path)
        if img is None:
            print(f"  ! could not read {s.image_path}, skipping")
            continue
        h, w = img.shape[:2]
        bbox = BBox(s.cx, s.cy, s.w, s.h)
        masked, _mask = inpaint_ball(img, bbox, dilate_frac=args.dilate_frac)

        stem = f"{i:03d}_{s.name[:24]}"
        masked_path = masked_dir / f"{stem}.jpg"
        orig_path = orig_dir / f"{stem}.jpg"
        cv2.imwrite(str(masked_path), masked)
        cv2.imwrite(str(orig_path), img)

        rec = {
            "id": stem,
            "source_name": s.name,
            "masked_path": str(masked_path),
            "original_path": str(orig_path),
            "width": w,
            "height": h,
            "gt": {"x": s.cx, "y": s.cy},
            "ball_px": round(max(s.w * w, s.h * h), 1),
            "predictions": {"center": dict(zip("xy", center_of_frame()))},
        }

        if not args.no_leak_check and not args.no_gpt:
            try:
                raw = azure_gpt.localize(str(masked_path), LEAK_CHECK, args.deployment)
                leak = json.loads(raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip())
                rec["leak"] = leak
                if leak.get("ball_visible") or leak.get("artifact_visible"):
                    print(f"  [leak] {stem}: {leak}")
            except Exception as e:  # noqa: BLE001
                rec["leak"] = {"error": str(e)}

        if not args.no_gpt:
            try:
                raw = azure_gpt.localize(str(masked_path), PROMPTS["neutral"], args.deployment)
                pred = parse_prediction(raw)
                rec["predictions"]["gpt"] = pred.model_dump()
            except Exception as e:  # noqa: BLE001
                rec["predictions"]["gpt"] = {"error": str(e)}
                print(f"  ! GPT failed on {stem}: {e}")

        items.append(rec)
        # NOTE: do not print GT coordinates — the Claude-agent that produces blind
        # predictions reads this log, and leaking the GT would contaminate it.
        print(f"  [{i+1}/{len(samples)}] {stem}  ball≈{rec['ball_px']}px")

    manifest = {
        "dataset": "martinjolif/football-ball-detection",
        "split": args.split,
        "seed": args.seed,
        "deployment": args.deployment,
        "dilate_frac": args.dilate_frac,
        "prompt_version": "v0-neutral",
        "items": items,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nWrote {OUT/'manifest.json'} with {len(items)} items.")
    print(f"Masked images in {masked_dir}/ — ready for Claude-agent predictions.")


if __name__ == "__main__":
    main()
