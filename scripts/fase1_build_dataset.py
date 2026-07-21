"""Phase 1 — build a stratified evaluation set from SoccerNet-Tracking.

Two stratification axes (choose with --balance):
  - state  : ball state (possession/short_pass/long_pass/contested)
  - center : ball distance from image center (near/mid/far). Use this to DE-BIAS the
             broadcast camera: the center-of-frame baseline is strong because the
             camera follows the ball, so a balanced spread of ball positions is needed
             for a fair test. Each item records both the state and the center distance.

The ball is removed with a mask whose padding grows with ball speed (motion blur on
fast balls exceeds the tight GT bbox). Masks feed LaMa (scripts/inpaint_lama.py).

Usage:
  uv run python scripts/fase1_build_dataset.py --balance center --near 14 --mid 14 --far 14
  uv run python scripts/fase1_build_dataset.py --balance state
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import random
import zipfile

import cv2
import numpy as np

from wheres_the_ball.data.ball_state import ball_speed, classify
from wheres_the_ball.data.soccernet_tracking import list_clips, load_clip
from wheres_the_ball.masking.inpaint import BBox, inpaint_ball, write_mask

OUT = pathlib.Path("results/fase1")
STATE_STRATA = ["possession", "short_pass", "long_pass", "contested"]
CENTER_STRATA = ["near", "mid", "far"]
SAMPLE_EVERY = 25
MIN_GAP = 100
CLIP_CAP = 2
BLUR_PAD_K = 1.5     # extra mask px per (px/frame) of ball speed
BLUR_PAD_MAX = 60


def center_bin(cx: float, cy: float) -> str:
    d = math.hypot(cx - 0.5, cy - 0.5)
    return "near" if d < 0.15 else ("mid" if d < 0.30 else "far")


def gather(zf, seed):
    cands = []
    clips = {}
    for cp in list_clips(zf):
        c = load_clip(zf, cp)
        clips[c.info.name] = c
        W, H = c.info.width, c.info.height
        for f in range(1, c.info.length + 1, SAMPLE_EVERY):
            s = classify(c, f)
            if not s:
                continue
            b = c.ball_at(f)
            cb = center_bin(b[0] / W, b[1] / H)
            cands.append({"clip": c.info.name, "frame": f, "state": s, "center": cb})
    random.Random(seed).shuffle(cands)
    return cands, clips


def select(cands, axis, targets):
    strata = STATE_STRATA if axis == "state" else CENTER_STRATA
    chosen, per_clip = [], {}
    counts = {k: 0 for k in strata}
    # Fill the SCARCEST stratum first (fewest available candidates), so a rare bin
    # (e.g. off-center 'far' balls, which the ball-following camera makes uncommon)
    # isn't starved of the per-clip budget by abundant bins.
    avail = {k: sum(1 for cd in cands if cd[axis] == k) for k in strata}
    for key in sorted(strata, key=lambda k: avail[k]):
        for cd in cands:
            if cd[axis] != key or counts[key] >= targets[key]:
                continue
            picks = per_clip.get(cd["clip"], [])
            if len(picks) >= CLIP_CAP or any(abs(cd["frame"] - p) < MIN_GAP for p in picks):
                continue
            chosen.append(cd)
            per_clip.setdefault(cd["clip"], []).append(cd["frame"])
            counts[key] += 1
    return chosen, counts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default="data/SoccerNet/tracking/test.zip")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--balance", choices=["state", "center"], default="center")
    # state targets
    ap.add_argument("--possession", type=int, default=12)
    ap.add_argument("--short_pass", type=int, default=12)
    ap.add_argument("--long_pass", type=int, default=12)
    ap.add_argument("--contested", type=int, default=8)
    # center targets
    ap.add_argument("--near", type=int, default=14)
    ap.add_argument("--mid", type=int, default=14)
    ap.add_argument("--far", type=int, default=14)
    args = ap.parse_args()
    if args.balance == "state":
        targets = {k: getattr(args, k) for k in STATE_STRATA}
    else:
        targets = {k: getattr(args, k) for k in CENTER_STRATA}

    zf = zipfile.ZipFile(args.zip)
    split = pathlib.Path(args.zip).stem
    dirs = {n: OUT / n for n in ("masked", "original", "masks", "masked_telea")}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    print(f"Gathering candidates (balance={args.balance})…")
    cands, clips = gather(zf, args.seed)
    chosen, counts = select(cands, args.balance, targets)
    print(f"Selected {len(chosen)}. Per {args.balance}: {counts} (targets {targets})")

    items = []
    for cd in chosen:
        c = clips[cd["clip"]]
        W, H, frame = c.info.width, c.info.height, cd["frame"]
        bcx, bcy, bw, bh = c.ball_at(frame)
        speed = ball_speed(c, frame) or 0.0
        pad = min(speed * BLUR_PAD_K, BLUR_PAD_MAX)  # cover motion blur

        raw = zf.read(f"{split}/{c.info.name}/img1/{frame:06d}.jpg")
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        bbox = BBox(bcx / W, bcy / H, bw / W, bh / H)
        stem = f"{c.info.name}_{frame:06d}"
        cv2.imwrite(str(dirs["original"] / f"{stem}.jpg"), img)
        write_mask(str(dirs["masks"] / f"{stem}.png"), H, W, bbox, extra_pad_px=pad)
        telea, _ = inpaint_ball(img, bbox, extra_pad_px=pad)
        cv2.imwrite(str(dirs["masked_telea"] / f"{stem}.jpg"), telea)

        players = [
            {"x": round(cx / W, 4), "y": round(cy / H, 4), "role": e.role, "team": e.team}
            for cx, cy, e in c.players_at(frame)
        ]
        items.append({
            "id": stem, "clip": c.info.name, "frame": frame,
            "state": cd["state"], "center_bin": cd["center"],
            "action_class": c.info.action_class, "width": W, "height": H,
            "gt": {"x": round(bcx / W, 4), "y": round(bcy / H, 4)},
            "center_dist": round(math.hypot(bcx / W - 0.5, bcy / H - 0.5), 4),
            "ball_px": round(max(bw, bh), 1), "ball_speed": round(speed, 1),
            "mask_pad_px": round(pad, 1), "players": players,
            "masked_path": str(dirs["masked"] / f"{stem}.png"),
            "original_path": str(dirs["original"] / f"{stem}.jpg"),
        })

    manifest = {
        "dataset": "SoccerNet-Tracking", "split": split, "seed": args.seed,
        "balance": args.balance, "targets": targets, "counts": counts,
        "n": len(items), "items": items,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nWrote {OUT/'manifest.json'} ({len(items)} items) + original/masks/telea.")


if __name__ == "__main__":
    main()
