"""M6 — pitch control as a cited, RUN interpretable baseline for ball localization.

Pitch control (Spearman, MIT Sloan 2018) is the sports-analytics standard for "who controls
space toward the ball". We build a standard time-to-arrive control surface from each player's
position + velocity and derive a ball-location prediction from it, so the paper compares its
learned geometry against the field's principled physical model---not only against a naive
centroid.

Model (per frame, in metres on a 105x68 pitch):
  - a player continues at current velocity for a reaction time t_react, then sprints at v_max
    toward a target cell x: TTI_i(x) = t_react + ||x - (p_i + v_i t_react)|| / v_max
  - team control C_team(x) = sum_i sigma((T - TTI_i(x)) / tau) over that team's players
  - predicted ball = the cell maximising CONTENTION C_home(x) * C_away(x)
    (a duel/contested point, where the ball tends to be), with a fallback to the
    max total-control cell.

No training (it is a physics baseline). Evaluated LOMO-free (it has no parameters to fit) as
median error over a frame subsample per match, vs the velocity-centroid and the geo-GBM.

Usage: uv run python scripts/paper_pitch_control.py
"""
from __future__ import annotations

import json
import pathlib

import numpy as np

from paper_deep_tuned import load_cached_matches

OUT = pathlib.Path("results/nivel3")
LX, LY = 105.0, 68.0          # pitch metres
V_MAX = 7.0                   # sprint speed (m/s)
T_REACT = 0.7                 # reaction time (s)
T_CTRL = 3.0                  # control time horizon (s)
TAU = 0.45                    # logistic steepness
GX, GY = 30, 20               # control grid
SUBSAMPLE = 1500              # frames per match (median error is stable)
rng = np.random.default_rng(0)

# grid of cell centres in metres
gx = (np.arange(GX) + 0.5) / GX * LX
gy = (np.arange(GY) + 0.5) / GY * LY
GXX, GYY = np.meshgrid(gx, gy)                     # [GY,GX]
GRID = np.stack([GXX.ravel(), GYY.ravel()], 1)     # [C,2] metres


def predict_ball(players):
    """players[N,5] = x,y,vx,vy,team (normalized). Return predicted ball (norm x,y)."""
    pos = players[:, :2] * [LX, LY]                # metres
    vel = players[:, 2:4] * [LX, LY]               # m/s
    team = players[:, 4]
    start = pos + vel * T_REACT                    # [N,2]
    d = np.linalg.norm(GRID[None, :, :] - start[:, None, :], axis=2)  # [N,C]
    tti = T_REACT + d / V_MAX
    ctrl = 1.0 / (1.0 + np.exp(-(T_CTRL - tti) / TAU))                # [N,C]
    home = ctrl[team > 0].sum(0) if (team > 0).any() else np.zeros(GRID.shape[0])
    away = ctrl[team < 0].sum(0) if (team < 0).any() else np.zeros(GRID.shape[0])
    contention = home * away
    c = int(np.argmax(contention)) if contention.max() > 0 else int(np.argmax(home + away))
    return GRID[c] / [LX, LY]                        # back to normalized


def main() -> None:
    matches = load_cached_matches()
    names = list(matches)
    pc_err, cen_err = [], []
    for n in names:
        frames = matches[n]["frames"]
        idx = rng.permutation(len(frames))[:SUBSAMPLE]
        errs, cerrs = [], []
        for i in idx:
            p, b = frames[i]
            pred = predict_ball(np.asarray(p))
            errs.append(float(np.linalg.norm(pred - b)))
            w = np.linalg.norm(p[:, 2:4], axis=1) + 1e-6      # velocity-centroid ref
            c = (p[:, :2] * w[:, None]).sum(0) / w.sum()
            cerrs.append(float(np.linalg.norm(c - b)))
        pc_err.append(np.median(errs)); cen_err.append(np.median(cerrs))
        print(f"{n:26} pitch-control={np.median(errs):.4f}  vel-centroid={np.median(cerrs):.4f}", flush=True)
    pc, cen = np.array(pc_err), np.array(cen_err)
    print(f"\nAcross {len(names)} matches (median error):")
    print(f"  pitch control  {pc.mean():.4f}±{pc.std():.4f}")
    print(f"  vel-centroid   {cen.mean():.4f}±{cen.std():.4f}")
    print(f"  (geo-GBM 0.099, tuned deep 0.087 for reference)")
    print(f"  pitch control beats vel-centroid in {int((pc < cen).sum())}/{len(names)} matches")
    (OUT / "paper_pitch_control.json").write_text(json.dumps(
        {"pitch_control": float(pc.mean()), "pitch_control_std": float(pc.std()),
         "vel_centroid": float(cen.mean()), "n": len(names),
         "params": {"v_max": V_MAX, "t_react": T_REACT, "t_ctrl": T_CTRL, "tau": TAU,
                    "grid": [GX, GY], "subsample": SUBSAMPLE}}, indent=2))
    print(f"Saved {OUT/'paper_pitch_control.json'}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "scripts")
    main()
