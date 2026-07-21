"""Phase 1 — build a stratified evaluation set from SoccerNet-Tracking.

For each clip we sample candidate frames, classify the ball state, then greedily
select a stratified set (rarest stratum first, capped per clip and spaced in time to
avoid near-duplicate frames). Selected frames are extracted from the zip, the ball is
removed by inpainting (bbox as mask), and a manifest with per-frame ground truth +
player positions (for geometric baselines) is written.

Usage:
  uv run python scripts/fase1_build_dataset.py \
      --zip data/SoccerNet/tracking/test.zip \
      --possession 12 --short_pass 12 --long_pass 12 --contested 8
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import zipfile

import cv2
import numpy as np

from wheres_the_ball.data.ball_state import classify
from wheres_the_ball.data.soccernet_tracking import list_clips, load_clip
from wheres_the_ball.masking.inpaint import BBox, inpaint_ball, write_mask

OUT = pathlib.Path("results/fase1")
STRATA = ["possession", "short_pass", "long_pass", "contested"]
SAMPLE_EVERY = 25   # candidate frame cadence (~1s at 25fps)
MIN_GAP = 100       # min frames between two picks from the same clip
CLIP_CAP = 2        # max target frames per clip


def gather_candidates(zf, seed):
    """Return list of (clip_name, clip_obj, frame, state)."""
    cands = []
    clips = {}
    for cp in list_clips(zf):
        c = load_clip(zf, cp)
        clips[c.info.name] = c
        for f in range(1, c.info.length + 1, SAMPLE_EVERY):
            s = classify(c, f)
            if s:
                cands.append((c.info.name, f, s))
    random.Random(seed).shuffle(cands)
    return cands, clips


def select(cands, targets):
    """Greedy stratified selection: rarest stratum first, per-clip cap + time gap."""
    chosen = []
    per_clip: dict[str, list[int]] = {}
    counts = {s: 0 for s in STRATA}
    # Fill rarest strata first so scarce ones (contested) aren't starved.
    for stratum in sorted(STRATA, key=lambda s: targets[s]):
        for name, frame, s in cands:
            if s != stratum or counts[s] >= targets[s]:
                continue
            picks = per_clip.get(name, [])
            if len(picks) >= CLIP_CAP or any(abs(frame - p) < MIN_GAP for p in picks):
                continue
            chosen.append((name, frame, s))
            per_clip.setdefault(name, []).append(frame)
            counts[s] += 1
    return chosen, counts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default="data/SoccerNet/tracking/test.zip")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--possession", type=int, default=12)
    ap.add_argument("--short_pass", type=int, default=12)
    ap.add_argument("--long_pass", type=int, default=12)
    ap.add_argument("--contested", type=int, default=8)
    args = ap.parse_args()
    targets = {s: getattr(args, s) for s in STRATA}

    zf = zipfile.ZipFile(args.zip)
    split = pathlib.Path(args.zip).stem  # 'test'
    masked_dir = OUT / "masked"      # produced by LaMa (scripts/inpaint_lama.py)
    orig_dir = OUT / "original"      # unedited frames (LaMa input)
    mask_dir = OUT / "masks"         # white-on-black ball masks (LaMa input)
    telea_dir = OUT / "masked_telea"  # quick Telea preview for comparison
    for d in (masked_dir, orig_dir, mask_dir, telea_dir):
        d.mkdir(parents=True, exist_ok=True)

    print("Gathering candidates…")
    cands, clips = gather_candidates(zf, args.seed)
    chosen, counts = select(cands, targets)
    print(f"Selected {len(chosen)} frames. Per stratum: {counts}")
    print(f"Targets were: {targets} (contested is naturally scarce)")

    items = []
    for name, frame, state in chosen:
        c = clips[name]
        W, H = c.info.width, c.info.height
        ball = c.ball_at(frame)
        bcx, bcy, bw, bh = ball
        img_name = f"{c.info.name}/img1/{frame:06d}.jpg"
        # Read frame from zip, matching the split prefix (e.g. test/SNMOT-...).
        raw = zf.read(f"{split}/{img_name}")
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        bbox = BBox(bcx / W, bcy / H, bw / W, bh / H)
        stem = f"{c.info.name}_{frame:06d}"
        # LaMa inputs: original frame + white-on-black mask (same basename).
        cv2.imwrite(str(orig_dir / f"{stem}.jpg"), img)
        write_mask(str(mask_dir / f"{stem}.png"), H, W, bbox)
        # Quick Telea preview (for visual comparison against LaMa).
        telea, _ = inpaint_ball(img, bbox)
        cv2.imwrite(str(telea_dir / f"{stem}.jpg"), telea)

        players = [
            {"x": round(cx / W, 4), "y": round(cy / H, 4), "role": e.role, "team": e.team}
            for cx, cy, e in c.players_at(frame)
        ]
        items.append({
            "id": stem, "clip": c.info.name, "frame": frame, "state": state,
            "action_class": c.info.action_class, "width": W, "height": H,
            "gt": {"x": round(bcx / W, 4), "y": round(bcy / H, 4)},
            "ball_px": round(max(bw, bh), 1),
            "players": players,
            "masked_path": str(masked_dir / f"{stem}.png"),  # LaMa output (see inpaint_lama.py)
            "original_path": str(orig_dir / f"{stem}.jpg"),
        })

    manifest = {
        "dataset": "SoccerNet-Tracking", "split": split, "seed": args.seed,
        "targets": targets, "counts": counts, "n": len(items), "items": items,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nWrote {OUT/'manifest.json'} ({len(items)} items) + masked/original frames.")


if __name__ == "__main__":
    main()
