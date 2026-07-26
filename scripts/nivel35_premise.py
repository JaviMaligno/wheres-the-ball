"""Level 3.5 · step 0 — where does the wide error actually live?

Before building any direction-vs-position machinery, test the premise: is the
hard, high-error mass concentrated in set pieces (corners / free kicks / throw-ins),
in open-play long balls & clearances, or spread across ordinary play?

Uses Metrica event data (RawEventsData.csv: typed events with start/end frame and
start/end ball coords) to label each evaluation frame, then measures the geometric
specialist's per-frame error by category. Soccer, field coords. Local, ~$0.

Usage: uv run python scripts/nivel35_premise.py
"""
from __future__ import annotations

import csv
import pathlib

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from wheres_the_ball.data.field_tracking import load_metrica_game
from wheres_the_ball.features.geometry import frame_features

PITCH_X, PITCH_Y = 105.0, 68.0   # metres, for physical distances
FPS = 25
VEL_LAG = 5
LONG_M = 30.0                    # a pass/clearance longer than this = "long ball"


def read_tracking(game_dir: pathlib.Path):
    """Return dict frame_no -> (players[N,5], ball[2]) plus ball speed, keyed by the
    Metrica Frame column so events line up."""
    name = game_dir.name
    sides = {}
    for side in ("Home", "Away"):
        rows = list(csv.reader((game_dir / f"{name}_RawTrackingData_{side}_Team.csv").open()))
        # row[1] = Frame; row[3:] = x,y pairs (…, ball_x, ball_y)
        arr, frames = [], []
        for r in rows[3:]:
            frames.append(int(r[1]))
            arr.append([float(c) if c not in ("", "NaN") else np.nan for c in r[3:]])
        sides[side] = (np.array(frames), np.array(arr))
    fr = sides["Home"][0]
    home, away = sides["Home"][1], sides["Away"][1]
    ball = home[:, -2:]
    home_p, away_p = home[:, :-2], away[:, :-2]
    out = {}
    for idx in range(VEL_LAG, len(fr)):
        if np.isnan(ball[idx]).any() or np.isnan(ball[idx - VEL_LAG]).any():
            continue
        players = []
        for side_arr, flag in ((home_p, 1.0), (away_p, -1.0)):
            now, prev = side_arr[idx], side_arr[idx - VEL_LAG]
            for j in range(0, side_arr.shape[1], 2):
                if np.isnan(now[j]) or np.isnan(prev[j]):
                    continue
                vx = (now[j] - prev[j]) * FPS / VEL_LAG
                vy = (now[j+1] - prev[j+1]) * FPS / VEL_LAG
                players.append([now[j], now[j+1], vx, vy, flag])
        if len(players) < 8:
            continue
        bvel = (ball[idx] - ball[idx - VEL_LAG]) * FPS / VEL_LAG * [PITCH_X, PITCH_Y]
        bspeed = float(np.linalg.norm(bvel))
        out[int(fr[idx])] = (np.array(players, np.float32), ball[idx].astype(np.float32), bspeed)
    return out


def frame_categories(game_dir: pathlib.Path, max_frame: int):
    """Return array cat[frame] -> label, built from the event log."""
    name = game_dir.name
    rows = list(csv.reader((game_dir / f"{name}_RawEventsData.csv").open()))
    h = rows[0]
    iT, iS = h.index("Type"), h.index("Subtype")
    iSF, iEF = h.index("Start Frame"), h.index("End Frame")
    iSX, iSY, iEX, iEY = (h.index("Start X"), h.index("Start Y"), h.index("End X"), h.index("End Y"))
    cat = np.array(["open play"] * (max_frame + 2), dtype=object)

    def num(v):
        try:
            return float(v)
        except ValueError:
            return np.nan
    # later events overwrite earlier ones on overlap; iterate so set pieces win by re-stamping
    intervals = []
    for r in rows[1:]:
        t, sub = r[iT], r[iS]
        sf, ef = int(r[iSF]), int(r[iEF])
        sx, sy, ex, ey = num(r[iSX]), num(r[iSY]), num(r[iEX]), num(r[iEY])
        dist = np.hypot((ex - sx) * PITCH_X, (ey - sy) * PITCH_Y) if not np.isnan(ex + sx) else 0.0
        if t == "SET PIECE":
            label = "set piece"
        elif t in ("PASS", "SHOT") and dist >= LONG_M:
            label = "long ball / clearance"
        elif "AERIAL" in sub:
            label = "long ball / clearance"
        elif t in ("PASS", "SHOT"):
            label = "short pass / shot"
        else:
            continue
        intervals.append((sf, ef, label, 0 if label == "set piece" else 1))
    # stamp non-set-piece first, then set piece, so set piece wins overlaps
    for prio in (1, 0):
        for sf, ef, label, p in intervals:
            if p != prio:
                continue
            a, b = max(0, sf), min(max_frame, ef)
            cat[a:b + 1] = label
    return cat


def main() -> None:
    base = pathlib.Path("data/sample-data/data")
    print("Training geometric specialist on game 1…")
    tr = list(load_metrica_game(base / "Sample_Game_1"))
    Xtr = np.stack([frame_features(p) for p, _ in tr]); Ytr = np.stack([b for _, b in tr])
    mx = HistGradientBoostingRegressor(max_iter=300, random_state=0).fit(Xtr, Ytr[:, 0])
    my = HistGradientBoostingRegressor(max_iter=300, random_state=0).fit(Xtr, Ytr[:, 1])

    print("Loading game 2 tracking + events…")
    g2 = base / "Sample_Game_2"
    track = read_tracking(g2)
    max_frame = max(track)
    cat = frame_categories(g2, max_frame)

    frames = sorted(track)
    X = np.stack([frame_features(track[f][0]) for f in frames])
    Y = np.stack([track[f][1] for f in frames])
    spd = np.array([track[f][2] for f in frames])
    labels = np.array([cat[f] for f in frames], dtype=object)
    pred = np.stack([mx.predict(X), my.predict(X)], 1)
    err = np.linalg.norm(pred - Y, axis=1)

    order = ["open play", "short pass / shot", "long ball / clearance", "set piece"]
    print(f"\nEval frames: {len(frames)}   overall median error {np.median(err):.4f}\n")
    print(f"{'category':24}{'n':>7}{'%frames':>9}{'med err':>10}{'med ball spd (m/s)':>20}")
    for c in order:
        m = labels == c
        if m.sum() == 0:
            continue
        print(f"{c:24}{m.sum():>7}{m.mean():>8.0%}{np.median(err[m]):>10.4f}{np.median(spd[m]):>20.1f}")

    # where does the WORST error live?
    worst = err >= np.quantile(err, 0.75)
    print(f"\nComposition of the worst-25% error frames (n={worst.sum()}):")
    for c in order:
        frac = (labels[worst] == c).mean()
        base_frac = (labels == c).mean()
        lift = frac / base_frac if base_frac > 0 else 0
        print(f"  {c:24}{frac:>6.0%}   (base {base_frac:.0%}, over-representation {lift:.1f}x)")

    # the sharper cut: ball SPEED (flight state), not event label
    print("\n--- Error by ball speed (is the ball in flight?) ---")
    edges = [0, 2, 6, 12, np.inf]
    names = ["settled (<2 m/s)", "slow (2-6)", "moving (6-12)", "flight (>12 m/s)"]
    print(f"{'ball state':22}{'n':>8}{'%':>6}{'med err':>10}{'p90 err':>10}")
    for lo, hi, nm in zip(edges[:-1], edges[1:], names):
        m = (spd >= lo) & (spd < hi)
        if m.sum():
            print(f"{nm:22}{m.sum():>8}{m.mean():>5.0%}{np.median(err[m]):>10.4f}{np.quantile(err[m],0.9):>10.4f}")
    corr = float(np.corrcoef(spd, err)[0, 1])
    print(f"corr(ball speed, error) = {corr:+.3f}")
    fast = spd >= 12
    print(f"flight frames are {(labels[fast]=='long ball / clearance').mean():.0%} long-ball / "
          f"{(labels[fast]=='set piece').mean():.0%} set-piece / "
          f"{(labels[fast]=='open play').mean():.0%} open-play labelled")


if __name__ == "__main__":
    main()
