"""The missing cell: VLMs given the SAME player tracks the tiny net saw (as text).

Completes the 2x2 {model: VLM, tiny-net} x {input: pixels, tracks}. If VLMs solve
off-center balls from tracks-as-text, their pixel failure is a reading problem
(the article's claim); if they still fail, the bottleneck is the geometric
inference itself. (The 4th cell, tiny-net-on-pixels, is a vision model — out of
scope, stated as a limit.)

Runs on the exact eval items the ceiling used (those with 1 s of history), same
5-step trajectories, image-normalized coords. Keys: gpt_tracks / claude_opus_tracks.

Usage:
  source ../CooperBench/azure_env.sh
  uv run python scripts/nivel1_vlm_tracks.py [--limit N]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import zipfile

OUT = pathlib.Path("results/fase1")
FALLBACK_ENV = pathlib.Path("../llm-language-limits/.env")

PROMPT = """You are given one second of player tracking data from a football (soccer) \
broadcast clip. Coordinates are normalized to the video frame: x in [0,1] left→right, \
y in [0,1] top→bottom (NOT pitch coordinates — this is the camera view, and the camera \
moves). Each player line gives: team (A/B/G for goalkeeper side noted), then 5 (x,y) \
positions sampled 0.2 s apart, oldest first — the 5th is the CURRENT moment.

The ball is not given. Infer where the ball most likely is at the CURRENT moment, \
using the players' positions and how they move.

{tracks}

Answer with a STRICT JSON object and nothing else:
{{"x": <float 0..1>, "y": <float 0..1>, "uncertainty_radius": <float 0..1>,
 "confidence": <int 0..100>, "rationale": "<one short sentence>"}}"""


def _ensure_anthropic_key():
    import os
    if os.environ.get("ANTHROPIC_API_KEY"):
        return
    for line in FALLBACK_ENV.read_text().splitlines():
        if line.startswith("ANTHROPIC_API_KEY="):
            os.environ["ANTHROPIC_API_KEY"] = line.split("=", 1)[1].strip()
            return


def tracks_text(players):
    """players[N,21] = [x,y,vx,vy]x5 + team → readable per-player track lines."""
    lines = []
    for p in players:
        team = "A" if p[20] > 0 else "B"
        pts = " ".join(f"({p[k*4]:.3f},{p[k*4+1]:.3f})" for k in range(5))
        lines.append(f"{team}: {pts}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    _ensure_anthropic_key()

    import sys
    sys.path.insert(0, "scripts")
    from nivel1_ceiling import clip_samples
    from wheres_the_ball.data.soccernet_tracking import load_clip
    from wheres_the_ball.models import anthropic_claude, azure_gpt
    from wheres_the_ball.models.base import parse_prediction

    man = json.loads((OUT / "manifest.json").read_text())
    preds = json.loads((OUT / "predictions.json").read_text())
    zf = zipfile.ZipFile("data/SoccerNet/tracking/test.zip")
    cache = {}

    def needs(rec, key):
        return key not in rec or (isinstance(rec.get(key), dict) and "error" in rec[key])

    items = [it for it in man["items"] if "specialist" in preds.get(it["id"], {})]
    if args.limit:
        items = items[: args.limit]
    print(f"{len(items)} items (paired with the ceiling)")

    for i, it in enumerate(items):
        rec = preds[it["id"]]
        c = cache.setdefault(it["clip"], load_clip(zf, f"test/{it['clip']}"))
        s = clip_samples(c, it["width"], it["height"], anchors=[it["frame"]])
        if not s:
            continue
        prompt = PROMPT.format(tracks=tracks_text(s[0][0]))
        for key, fn in (("gpt_tracks", lambda p: azure_gpt.localize_text(p, deployment="gpt-5.4")),
                        ("claude_opus_tracks", lambda p: anthropic_claude.localize_text(p, model="claude-opus-4-8"))):
            if needs(rec, key):
                try:
                    rec[key] = parse_prediction(fn(prompt)).model_dump()
                except Exception as e:  # noqa: BLE001
                    rec[key] = {"error": f"{type(e).__name__}: {e}"}
        preds[it["id"]] = rec
        (OUT / "predictions.json").write_text(json.dumps(preds, indent=2))
        g = rec.get("gpt_tracks", {}).get("x"); o = rec.get("claude_opus_tracks", {}).get("x")
        print(f"[{i+1}/{len(items)}] {it['id']} gpt={g} opus={o}")


if __name__ == "__main__":
    main()
