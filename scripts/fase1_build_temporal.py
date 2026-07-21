"""Phase 1 RQ2 — build masked frame SEQUENCES for the temporal condition.

For each item in the (single-frame) manifest, extract the frames t-75, t-50, t-25, t
(~1s apart at 25fps) and mask the ball in EVERY frame (bbox + speed-aware padding), so
the model can never just see the ball in an earlier frame. Outputs go to
results/fase1/clips/{original,masks} for LaMa (scripts/inpaint_lama.py --root ...),
and a temporal manifest records the ordered masked-frame paths per item.

Usage:
  uv run python scripts/fase1_build_temporal.py
  uv run python scripts/inpaint_lama.py --root results/fase1/clips
"""
from __future__ import annotations

import json
import pathlib
import zipfile

import cv2
import numpy as np

from wheres_the_ball.data.ball_state import ball_speed
from wheres_the_ball.data.soccernet_tracking import load_clip
from wheres_the_ball.masking.inpaint import BBox, write_mask

OUT = pathlib.Path("results/fase1")
CLIPS = OUT / "clips"
OFFSETS = [75, 50, 25, 0]   # frames before target (25fps → ~3s..0, 1s apart)
BLUR_PAD_K, BLUR_PAD_MAX = 1.5, 60


def main() -> None:
    manifest = json.loads((OUT / "manifest.json").read_text())
    split = manifest["split"]
    zf = zipfile.ZipFile(f"data/SoccerNet/tracking/{split}.zip")
    orig_dir, mask_dir = CLIPS / "original", CLIPS / "masks"
    for d in (orig_dir, mask_dir):
        d.mkdir(parents=True, exist_ok=True)

    clip_cache: dict[str, object] = {}
    tmanifest = []
    for it in manifest["items"]:
        name = it["clip"]
        if name not in clip_cache:
            clip_cache[name] = load_clip(zf, f"{split}/{name}")
        c = clip_cache[name]
        W, H = it["width"], it["height"]
        seq = []
        for k, off in enumerate(OFFSETS):
            pf = it["frame"] - off
            if pf < 1:
                continue
            raw = zf.read(f"{split}/{name}/img1/{pf:06d}.jpg")
            img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            stem = f"{it['id']}__k{k}_{pf:06d}"
            cv2.imwrite(str(orig_dir / f"{stem}.jpg"), img)
            ball = c.ball_at(pf)
            if ball:
                bcx, bcy, bw, bh = ball
                pad = min((ball_speed(c, pf) or 0.0) * BLUR_PAD_K, BLUR_PAD_MAX)
                write_mask(str(mask_dir / f"{stem}.png"), H, W,
                           BBox(bcx / W, bcy / H, bw / W, bh / H), extra_pad_px=pad)
            else:
                # No ball annotation at this frame → nothing to remove (empty mask).
                cv2.imwrite(str(mask_dir / f"{stem}.png"), np.zeros((H, W), np.uint8))
            seq.append(str(CLIPS / "masked" / f"{stem}.png"))  # LaMa output path
        tmanifest.append({
            "id": it["id"], "gt": it["gt"], "center_bin": it["center_bin"],
            "center_dist": it["center_dist"], "state": it["state"],
            "n_frames": len(seq), "frames": seq,
        })

    (OUT / "temporal_manifest.json").write_text(
        json.dumps({"offsets": OFFSETS, "n": len(tmanifest), "items": tmanifest}, indent=2))
    total = sum(t["n_frames"] for t in tmanifest)
    print(f"Wrote temporal_manifest.json: {len(tmanifest)} items, {total} frames to inpaint.")
    print(f"Next: uv run python scripts/inpaint_lama.py --root {CLIPS}")


if __name__ == "__main__":
    main()
