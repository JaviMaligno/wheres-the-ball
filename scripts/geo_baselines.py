"""Rich geometric baselines B2-B4 on the n=92 eval set (Level-2 Fase 0, step 1).

The Level-2 design's geometric hypothesis (H1: the dominant signal is collective
geometry) was never actually tested in Level 1 — only B0 (center) and B1 (centroid)
ran. This adds, using GT player tracks only (no learning, no APIs):

  b2_vel   : centroid weighted by player speed (fast movers point at the action)
  b3_fast  : position of the fastest player
  b4_voro  : centroid of the Voronoi cell of the highest (density+speed) player

Velocities are computed in image space and CAMERA-COMPENSATED by subtracting the
mean displacement of all players present in both frames (broadcast pan is
common-mode). Predictions merge into results/fase1/predictions.json.

Usage: uv run python scripts/geo_baselines.py
"""
from __future__ import annotations

import json
import math
import pathlib
import zipfile

import numpy as np
from scipy.spatial import Voronoi

from wheres_the_ball.data.soccernet_tracking import load_clip

OUT = pathlib.Path("results/fase1")
ZIP = "data/SoccerNet/tracking/test.zip"
DENSITY_R = 150.0  # px neighborhood for local density


def player_tracks(clip, frame):
    """Return dict tid -> (pos_now, pos_prev) for players present in both frames."""
    now = {t: (x, y) for t, (x, y, _w, _h) in clip.frames.get(frame, {}).items()
           if clip.info.entities.get(t) and clip.info.entities[t].role in ("player", "goalkeeper")}
    prev_f = frame - 1 if (frame - 1) in clip.frames else frame + 1
    prev = {t: (x, y) for t, (x, y, _w, _h) in clip.frames.get(prev_f, {}).items()}
    return {t: (now[t], prev[t]) for t in now if t in prev}


def predictions_for(clip, frame, W, H):
    tracks = player_tracks(clip, frame)
    if len(tracks) < 4:
        return None
    pos = np.array([p for p, _ in tracks.values()])
    disp = np.array([[p[0] - q[0], p[1] - q[1]] for p, q in tracks.values()])
    disp_rel = disp - disp.mean(axis=0)          # camera compensation
    speed = np.linalg.norm(disp_rel, axis=1)

    # B2 — speed-weighted centroid
    w = speed + 1e-6
    b2 = (pos * w[:, None]).sum(axis=0) / w.sum()

    # B3 — fastest player
    b3 = pos[int(np.argmax(speed))]

    # B4 — Voronoi cell centroid of the max (density + speed) player
    dens = np.array([np.sum(np.linalg.norm(pos - p, axis=1) < DENSITY_R) - 1 for p in pos])
    score = dens / max(dens.max(), 1) + speed / max(speed.max(), 1e-6)
    win = int(np.argmax(score))
    b4 = pos[win]
    try:
        vor = Voronoi(pos)
        region = vor.regions[vor.point_region[win]]
        if region and -1 not in region:  # bounded cell only
            verts = vor.vertices[region]
            # clip vertices to the frame before averaging
            verts = np.clip(verts, [0, 0], [W, H])
            b4 = verts.mean(axis=0)
    except Exception:
        pass  # degenerate geometry → keep player position

    def norm(p):
        return {"x": round(float(p[0]) / W, 4), "y": round(float(p[1]) / H, 4)}

    return {"b2_vel": norm(b2), "b3_fast": norm(b3), "b4_voro": norm(b4)}


def main() -> None:
    man = json.loads((OUT / "manifest.json").read_text())
    preds = json.loads((OUT / "predictions.json").read_text())
    zf = zipfile.ZipFile(ZIP)
    split = pathlib.Path(ZIP).stem
    cache: dict[str, object] = {}
    done = 0
    for it in man["items"]:
        clip = cache.setdefault(it["clip"], load_clip(zf, f"{split}/{it['clip']}"))
        out = predictions_for(clip, it["frame"], it["width"], it["height"])
        if out:
            preds.setdefault(it["id"], {}).update(out)
            done += 1
    (OUT / "predictions.json").write_text(json.dumps(preds, indent=2))
    print(f"B2-B4 computed for {done}/{man['n']} items → merged into predictions.json")

    # quick audit: correlation + far metrics, same style as audit_fase1
    rng = np.random.default_rng(0)
    ids = {i["id"]: i for i in man["items"]}
    far = [k for k, v in ids.items() if v["center_bin"] == "far"]
    print(f"\n{'system':10} corr_x  corr_y   far_med  far_win")
    for s in ["b2_vel", "b3_fast", "b4_voro", "centroid", "gpt", "claude_opus"]:
        P, G, fe, win = [], [], [], []
        for k, v in ids.items():
            p = preds.get(k, {}).get(s)
            if s == "centroid":
                ps = v["players"]
                p = {"x": sum(q["x"] for q in ps) / len(ps), "y": sum(q["y"] for q in ps) / len(ps)} if ps else None
            if not (isinstance(p, dict) and p.get("x") is not None and "error" not in p):
                continue
            g = (v["gt"]["x"], v["gt"]["y"])
            P.append((p["x"], p["y"])); G.append(g)
            if k in far:
                e = math.hypot(p["x"] - g[0], p["y"] - g[1])
                fe.append(e); win.append(e < math.hypot(0.5 - g[0], 0.5 - g[1]))
        P, G = np.array(P), np.array(G)
        cx = np.corrcoef(P[:, 0], G[:, 0])[0, 1]; cy = np.corrcoef(P[:, 1], G[:, 1])[0, 1]
        print(f"{s:10} {cx:+.2f}   {cy:+.2f}    {np.median(fe):.3f}    {sum(win)}/{len(win)}")


if __name__ == "__main__":
    main()
