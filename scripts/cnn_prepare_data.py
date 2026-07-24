"""Prepare training data for the 4th cell (small vision net on masked pixels).

Extracts frames with ball GT from the SoccerNet TRAIN split (stride 5 → ~8.5k),
downscales to 960x540 (bandwidth + LaMa speed), writes frame + speed-dilated ball
mask pairs and a labels.json (normalized ball xy). LaMa masking + CNN training run
on Modal (scripts/modal_app/cnn_pixels.py); eval uses the existing 92 masked items.

Usage: uv run python scripts/cnn_prepare_data.py [--stride 5]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import zipfile

import cv2
import numpy as np

from wheres_the_ball.data.ball_state import ball_speed
from wheres_the_ball.data.soccernet_tracking import list_clips, load_clip
from wheres_the_ball.masking.inpaint import BBox, write_mask

OUT = pathlib.Path("data/cnn_train")
SCALE = 0.5  # 1920x1080 -> 960x540
BLUR_PAD_K, BLUR_PAD_MAX = 1.5, 60


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=5)
    args = ap.parse_args()
    (OUT / "original").mkdir(parents=True, exist_ok=True)
    (OUT / "masks").mkdir(parents=True, exist_ok=True)

    zf = zipfile.ZipFile("data/SoccerNet/tracking/train.zip")
    labels = {}
    n = 0
    for cp in list_clips(zf):
        c = load_clip(zf, cp)
        W, H = c.info.width, c.info.height
        w2, h2 = int(W * SCALE), int(H * SCALE)
        for f in range(1, c.info.length + 1, args.stride):
            ball = c.ball_at(f)
            if ball is None:
                continue
            bcx, bcy, bw, bh = ball
            stem = f"{c.info.name}_{f:06d}"
            raw = zf.read(f"train/{c.info.name}/img1/{f:06d}.jpg")
            img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            img = cv2.resize(img, (w2, h2), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(OUT / "original" / f"{stem}.jpg"), img,
                        [cv2.IMWRITE_JPEG_QUALITY, 88])
            pad = min((ball_speed(c, f) or 0.0) * BLUR_PAD_K, BLUR_PAD_MAX) * SCALE
            write_mask(str(OUT / "masks" / f"{stem}.png"), h2, w2,
                       BBox(bcx / W, bcy / H, bw / W, bh / H), extra_pad_px=pad)
            labels[stem] = [round(bcx / W, 4), round(bcy / H, 4)]
            n += 1
        print(f"{c.info.name}: total {n}")
    (OUT / "labels.json").write_text(json.dumps(labels))
    print(f"\n{n} frames -> {OUT}/ (original+masks+labels.json)")


if __name__ == "__main__":
    main()
