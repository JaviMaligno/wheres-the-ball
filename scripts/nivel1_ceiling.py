"""Level-1 specialist ceiling — image-space DeepSets vs the VLMs, on the same eval.

Trains the trajectory DeepSets on the SoccerNet-Tracking TRAIN split (57 clips,
gt.txt only — no images touched), in IMAGE space (pixel coords normalized by frame
size; camera motion included, as the VLMs see it). Evaluates on the exact n=92
Level-1 items so the numbers sit in the same table as GPT/Opus/Sonnet/center.

Usage: uv run python scripts/nivel1_ceiling.py
"""
from __future__ import annotations

import json
import math
import pathlib
import zipfile

import numpy as np
import torch

from wheres_the_ball.data.soccernet_tracking import list_clips, load_clip
from nivel2_specialist import DeepSets
from nivel2_v1_temporal import collate, train, D_IN

OUT = pathlib.Path("results/nivel2")
FASE1 = pathlib.Path("results/fase1")
T_STEPS, LAG = 5, 5  # 1 s windows at 25 fps, steps 0.2 s apart


def clip_samples(clip, W, H, stride=2, anchors=None):
    """(players[N,21], ball[2]) trajectory samples from one clip's GT tracks."""
    span = LAG * (T_STEPS - 1)
    dt = LAG / 25.0
    frames = sorted(clip.frames)
    out = []
    targets = anchors if anchors is not None else range(frames[0] + span, frames[-1] + 1, stride)
    for f in targets:
        ball = clip.ball_at(f)
        if ball is None:
            continue
        idxs = [f - span + k * LAG for k in range(T_STEPS)]
        if not all(i in clip.frames for i in idxs):
            continue
        players = []
        for tid, ent in clip.info.entities.items():
            if ent.role not in ("player", "goalkeeper"):
                continue
            if not all(tid in clip.frames[i] for i in idxs):
                continue
            feats = []
            prev = None
            for i in idxs:
                x, y, _w, _h = clip.frames[i][tid]
                x, y = x / W, y / H
                if prev is None:
                    feats += [x, y, 0.0, 0.0]
                else:
                    feats += [x, y, (x - prev[0]) / dt, (y - prev[1]) / dt]
                prev = (x, y)
            flag = 1.0 if ent.team == "left" else (-1.0 if ent.team == "right" else 0.0)
            players.append(feats + [flag])
        if len(players) >= 8:
            out.append((np.array(players, dtype=np.float32),
                        np.array([ball[0] / W, ball[1] / H], np.float32)))
    return out


def main() -> None:
    torch.manual_seed(0)
    print("Building train samples from SoccerNet train split (gt only)…")
    ztr = zipfile.ZipFile("data/SoccerNet/tracking/train.zip")
    train_s = []
    for cp in list_clips(ztr):
        c = load_clip(ztr, cp)
        train_s.extend(clip_samples(c, c.info.width, c.info.height, stride=2))
    print(f"train samples: {len(train_s)} (57 clips)")

    model = train(DeepSets(d_in=D_IN), train_s, 4000, 1e-3, 0)
    torch.save(model.state_dict(), OUT / "ceiling_image_space.pt")

    # eval on the exact n=92 Level-1 items
    man = json.loads((FASE1 / "manifest.json").read_text())
    zte = zipfile.ZipFile("data/SoccerNet/tracking/test.zip")
    cache = {}
    preds, errs, P, G = {}, [], [], []
    far_errs, far_wins = [], []
    for it in man["items"]:
        c = cache.setdefault(it["clip"], load_clip(zte, f"test/{it['clip']}"))
        s = clip_samples(c, it["width"], it["height"], anchors=[it["frame"]])
        if not s:
            continue
        Pb, Mb, Yb = collate(s)
        with torch.no_grad():
            p = model(Pb, Mb).numpy()[0]
        gt = (it["gt"]["x"], it["gt"]["y"])
        e = math.hypot(p[0] - gt[0], p[1] - gt[1])
        preds[it["id"]] = {"x": round(float(p[0]), 4), "y": round(float(p[1]), 4)}
        errs.append(e); P.append(p); G.append(gt)
        if it["center_bin"] == "far":
            far_errs.append(e)
            far_wins.append(e < math.hypot(0.5 - gt[0], 0.5 - gt[1]))
    P, G = np.array(P), np.array(G)
    cx = float(np.corrcoef(P[:, 0], G[:, 0])[0, 1]); cy = float(np.corrcoef(P[:, 1], G[:, 1])[0, 1])
    print(f"\nCEILING sobre el eval n={len(errs)} del Nivel 1:")
    print(f"  median err = {np.median(errs):.3f}   corr = ({cx:+.2f},{cy:+.2f})")
    print(f"  far: med={np.median(far_errs):.3f}  win vs center={sum(far_wins)}/{len(far_wins)}")
    print("  (referencias: center 0.201 · gpt 0.117 corr(+.26,+.17) · opus 0.209 corr(+.37,+.34))")

    # merge into predictions.json under 'specialist'
    allp = json.loads((FASE1 / "predictions.json").read_text())
    for k, v in preds.items():
        allp.setdefault(k, {})["specialist"] = v
    (FASE1 / "predictions.json").write_text(json.dumps(allp, indent=2))
    (OUT / "ceiling.json").write_text(json.dumps(
        {"n": len(errs), "median": float(np.median(errs)), "corr": [cx, cy],
         "far_median": float(np.median(far_errs)), "far_win": f"{sum(far_wins)}/{len(far_wins)}"},
        indent=2))
    print("Saved ceiling.json + merged 'specialist' into fase1/predictions.json")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "scripts")
    main()
