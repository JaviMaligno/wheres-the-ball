"""Level 3.5 — the model needs motion: stillness is its blind spot; in flight,
direction beats position.

Article-4 experiment, two parts, soccer field coords (train Metrica g1, eval g2).

Part A — the still ball. Confirm the settled ball (<2 m/s) is the hard case, then
prove the mechanism: a positions-only model should match the full (velocity) model
when the ball is still (no motion to read) and lose badly when it moves. Dissect the
settled-ball error tail by ball-to-mass coupling and distance to the touchline.

Part B — direction in flight. When the ball IS moving fast, is its *direction of
travel* more recoverable from the players than its instantaneous position? Predict the
ball's velocity unit vector and compare angular error to chance and to a constant-mean
baseline.

Usage: uv run python scripts/nivel35_stillness.py
"""
from __future__ import annotations

import csv
import json
import pathlib

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from wheres_the_ball.features.geometry import frame_features, FEATURE_NAMES

PITCH_X, PITCH_Y, FPS, VEL_LAG = 105.0, 68.0, 25, 5
OUT = pathlib.Path("results/nivel3")

# positions-only feature columns: drop everything velocity-derived
_VEL_FEATS = {"vel_centroid", "fastest", "converge", "mean_speed", "max_speed"}
POS_COLS = [i for i, n in enumerate(FEATURE_NAMES)
            if n.replace("_x", "").replace("_y", "") not in _VEL_FEATS]


def load(game_dir: pathlib.Path):
    """Yield dicts with players[N,5], ball[2], ball speed (m/s) and ball dir (unit)."""
    name = game_dir.name
    sides = {}
    for side in ("Home", "Away"):
        rows = list(csv.reader((game_dir / f"{name}_RawTrackingData_{side}_Team.csv").open()))
        arr = np.array([[float(c) if c not in ("", "NaN") else np.nan for c in r[3:]]
                        for r in rows[3:]])
        sides[side] = arr
    n = min(len(sides["Home"]), len(sides["Away"]))
    home, away = sides["Home"][:n], sides["Away"][:n]
    ball = home[:, -2:]
    home_p, away_p = home[:, :-2], away[:, :-2]
    samples = []
    for idx in range(VEL_LAG, n):
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
        spd = float(np.linalg.norm(bvel))
        bdir = bvel / spd if spd > 1e-6 else np.array([0.0, 0.0])
        samples.append((np.array(players, np.float32), ball[idx].astype(np.float32), spd, bdir))
    return samples


def feats(samples, cols=None):
    X = np.stack([frame_features(p) for p, _, _, _ in samples])
    return X if cols is None else X[:, cols]


def fit_xy(X, Y):
    mx = HistGradientBoostingRegressor(max_iter=300, random_state=0).fit(X, Y[:, 0])
    my = HistGradientBoostingRegressor(max_iter=300, random_state=0).fit(X, Y[:, 1])
    return mx, my


def err_xy(m, X, Y):
    p = np.stack([m[0].predict(X), m[1].predict(X)], 1)
    return np.linalg.norm(p - Y, axis=1)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    base = pathlib.Path("data/sample-data/data")
    print("Loading…")
    tr = load(base / "Sample_Game_1")
    te = load(base / "Sample_Game_2")
    Ytr = np.stack([b for _, b, _, _ in tr]); Yte = np.stack([b for _, b, _, _ in te])
    spd = np.array([s for _, _, s, _ in te])
    print(f"train {len(tr)}  test {len(te)}")

    Xtr_full, Xte_full = feats(tr), feats(te)
    Xtr_pos, Xte_pos = feats(tr, POS_COLS), feats(te, POS_COLS)
    e_full = err_xy(fit_xy(Xtr_full, Ytr), Xte_full, Yte)
    e_pos = err_xy(fit_xy(Xtr_pos, Ytr), Xte_pos, Yte)

    # ---- Part A: mechanism — velocity helps only when the ball moves ----
    still = spd < 2.0
    print("\n=== PART A: the still ball ===")
    print(f"{'ball state':16}{'n':>7}{'full err':>10}{'pos-only err':>14}{'velocity helps':>16}")
    for nm, m in [("STILL (<2)", still), ("MOVING (>=2)", ~still)]:
        ef, ep = np.median(e_full[m]), np.median(e_pos[m])
        print(f"{nm:16}{m.sum():>7}{ef:>10.4f}{ep:>14.4f}{(ep-ef):>+16.4f}")
    print("(velocity helps = pos-only minus full; ~0 means velocity adds nothing there)")

    # dissect the settled tail: coupling (ball-to-mass) and touchline distance
    coup, touch = [], []
    for p, b, _, _ in te:
        pos, vel = p[:, :2], p[:, 2:4]
        w = np.linalg.norm(vel, axis=1) + 1e-6
        c = (pos * w[:, None]).sum(0) / w.sum()
        coup.append(np.linalg.norm(b - c))
        touch.append(min(b[1], 1 - b[1]))          # y is pitch width; small = near touchline
    coup, touch = np.array(coup), np.array(touch)
    st_err = e_full[still]
    worst = st_err >= np.quantile(st_err, 0.75)
    print("\nSettled-ball tail (worst 25% of still frames) vs all still frames:")
    print(f"  ball-to-mass coupling : {np.median(coup[still][worst]):.3f} vs {np.median(coup[still]):.3f}")
    print(f"  dist to touchline     : {np.median(touch[still][worst]):.3f} vs {np.median(touch[still]):.3f}")
    print(f"  corr(coupling, err | still) = {np.corrcoef(coup[still], st_err)[0,1]:+.2f}")

    # ---- Part B: direction in flight ----
    print("\n=== PART B: direction in flight (ball >12 m/s) ===")
    fl_tr = np.array([s for _, _, s, _ in tr]) >= 12
    fl_te = spd >= 12
    Dtr = np.stack([d for _, _, _, d in tr])[fl_tr]
    Dte = np.stack([d for _, _, _, d in te])[fl_te]
    Xd_tr = feats([tr[i] for i in np.where(fl_tr)[0]])
    Xd_te = feats([te[i] for i in np.where(fl_te)[0]])
    md = fit_xy(Xd_tr, Dtr)
    pred = np.stack([md[0].predict(Xd_te), md[1].predict(Xd_te)], 1)
    pred /= np.linalg.norm(pred, axis=1, keepdims=True) + 1e-9

    def ang(a, b):  # degrees between unit vectors
        return np.degrees(np.arccos(np.clip((a * b).sum(1), -1, 1)))
    learned = ang(pred, Dte)
    mean_dir = Dtr.mean(0); mean_dir /= np.linalg.norm(mean_dir) + 1e-9
    base_mean = ang(np.tile(mean_dir, (len(Dte), 1)), Dte)
    print(f"flight frames (eval): {fl_te.sum()}")
    print(f"  learned direction   median angular error: {np.median(learned):5.1f}°")
    print(f"  constant-mean baseline                   : {np.median(base_mean):5.1f}°")
    print(f"  chance (random direction)                : 90.0°")
    print(f"  position error in flight (normalized median): {np.median(e_full[fl_te]):.4f}  "
          f"(vs still {np.median(e_full[still]):.4f})")
    frac_good = (learned < 45).mean()
    print(f"  {frac_good:.0%} of flight balls have direction pinned within 45°")

    res = {"still_full": float(np.median(e_full[still])), "still_pos": float(np.median(e_pos[still])),
           "moving_full": float(np.median(e_full[~still])), "moving_pos": float(np.median(e_pos[~still])),
           "dir_learned_deg": float(np.median(learned)), "dir_meanbase_deg": float(np.median(base_mean)),
           "dir_within_45_frac": float(frac_good), "n_flight": int(fl_te.sum())}
    (OUT / "stillness.json").write_text(json.dumps(res, indent=2))
    print(f"\nSaved {OUT/'stillness.json'}")


if __name__ == "__main__":
    main()
