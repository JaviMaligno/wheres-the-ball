"""Run Claude (via API) on the Phase 0 masked images and write real claude_preds.json.

This replaces the contaminated Phase 0 "Claude-via-agent" predictions with real
Claude-API predictions on the exact same items, giving a fair Claude-vs-GPT comparison.

Key handling: uses ANTHROPIC_API_KEY if set, else loads it from
../llm-language-limits/.env (Javier's existing normal sk-ant-api key).

Usage:
  uv run python scripts/fase0_claude_api.py [--model claude-sonnet-4-6]
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib

MASKED = pathlib.Path("results/fase0/masked")
OUT = pathlib.Path("results/fase0/claude_preds.json")
FALLBACK_ENV = pathlib.Path("../llm-language-limits/.env")

CANDIDATE_MODELS = [
    "claude-sonnet-4-6", "claude-sonnet-4-5", "claude-opus-4-8", "claude-sonnet-5",
]


def _ensure_key() -> None:
    if os.environ.get("ANTHROPIC_API_KEY"):
        return
    if FALLBACK_ENV.exists():
        for line in FALLBACK_ENV.read_text().splitlines():
            if line.startswith("ANTHROPIC_API_KEY="):
                os.environ["ANTHROPIC_API_KEY"] = line.split("=", 1)[1].strip()
                print(f"Loaded ANTHROPIC_API_KEY from {FALLBACK_ENV}")
                return
    raise RuntimeError("ANTHROPIC_API_KEY not set and not found in fallback .env")


def _pick_model(preferred: str | None) -> str:
    """Ping candidate models with a trivial request; return the first that works."""
    import anthropic

    from wheres_the_ball.models import anthropic_claude

    client = anthropic_claude._client()
    order = ([preferred] if preferred else []) + [m for m in CANDIDATE_MODELS if m != preferred]
    for m in order:
        try:
            client.messages.create(
                model=m, max_tokens=8,
                messages=[{"role": "user", "content": "ping"}],
            )
            print(f"[liveness] model OK: {m}")
            return m
        except anthropic.NotFoundError:
            print(f"[liveness] model not available: {m}")
        except Exception as e:  # noqa: BLE001
            # Auth/other error — surface immediately, no point trying more models.
            raise RuntimeError(f"Anthropic API error on liveness ({m}): {type(e).__name__}: {e}")
    raise RuntimeError(f"No candidate model available: {order}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="force a model id (else auto-detect)")
    args = ap.parse_args()

    _ensure_key()
    from wheres_the_ball.models import anthropic_claude
    from wheres_the_ball.models.base import parse_prediction
    from wheres_the_ball.prompts.localize import PROMPTS

    model = _pick_model(args.model)

    stems = sorted(p.stem for p in MASKED.glob("*.jpg"))
    print(f"Running Claude ({model}) on {len(stems)} masked images…")
    preds: dict[str, dict] = {}
    for i, stem in enumerate(stems):
        img = str(MASKED / f"{stem}.jpg")
        try:
            raw = anthropic_claude.localize(img, PROMPTS["neutral"], model=model)
            pred = parse_prediction(raw)
            preds[stem] = pred.model_dump()
            print(f"  [{i+1}/{len(stems)}] {stem}: ({pred.x:.3f},{pred.y:.3f}) "
                  f"conf={pred.confidence:.0f}")
        except Exception as e:  # noqa: BLE001
            preds[stem] = {"error": str(e)}
            print(f"  ! {stem}: {type(e).__name__}: {e}")

    OUT.write_text(json.dumps(preds, indent=2))
    print(f"\nWrote {OUT} (model={model})")


if __name__ == "__main__":
    main()
