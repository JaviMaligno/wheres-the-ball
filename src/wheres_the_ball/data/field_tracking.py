"""Field-coordinate tracking loaders for the Level-2 specialist.

Unified sample format across sports:
  players: float array [N, 5] = (x, y, vx, vy, team±1) — coords normalized to [0,1]
           by field dimensions; velocities in field-fractions per second.
  ball:    float array [2] = (x, y) normalized.

Sources:
  - Metrica Sports sample-data (soccer, 25 Hz, already 0-1 normalized, ball col last).
  - NBA SportVU 2015-16 dumps (basketball, 25 Hz, feet on a 94x50 court, ball first
    entity with teamid=-1). License note: the SportVU dumps circulate without an
    explicit license; standard academic use, we do not redistribute the data.
"""
from __future__ import annotations

import csv
import json
import pathlib

import numpy as np

FPS = 25
VEL_LAG = 5  # frames used for finite-difference velocity (0.2 s)


def _velocities(pos_now, pos_prev, dt):
    return (pos_now - pos_prev) / dt


def load_metrica_game(game_dir: str | pathlib.Path, stride: int = 5):
    """Yield (players[N,5], ball[2]) samples from one Metrica game (Home+Away)."""
    game_dir = pathlib.Path(game_dir)
    name = game_dir.name
    sides = {}
    for side in ("Home", "Away"):
        f = game_dir / f"{name}_RawTrackingData_{side}_Team.csv"
        rows = list(csv.reader(f.open()))
        data = rows[3:]  # 3 header rows
        # columns: Period, Frame, Time, then x,y pairs; last pair is the ball
        arr = np.array([[float(c) if c not in ("", "NaN") else np.nan for c in r[3:]]
                        for r in data])
        sides[side] = arr
    n = min(len(sides["Home"]), len(sides["Away"]))
    home, away = sides["Home"][:n], sides["Away"][:n]
    ball = home[:, -2:]  # ball duplicated in both files
    home_p, away_p = home[:, :-2], away[:, :-2]

    for i in range(VEL_LAG, n, stride):
        if np.isnan(ball[i]).any():
            continue
        players = []
        for side_arr, flag in ((home_p, 1.0), (away_p, -1.0)):
            now, prev = side_arr[i], side_arr[i - VEL_LAG]
            for j in range(0, side_arr.shape[1], 2):
                if np.isnan(now[j]) or np.isnan(prev[j]):
                    continue
                vx, vy = _velocities(now[j:j+2], prev[j:j+2], VEL_LAG / FPS)
                players.append([now[j], now[j+1], vx, vy, flag])
        if len(players) >= 8:
            yield np.array(players, dtype=np.float32), ball[i].astype(np.float32)


COURT_X, COURT_Y = 94.0, 50.0  # NBA court in feet


def load_sportvu_game(json_path: str | pathlib.Path, stride: int = 5):
    """Yield (players[N,5], ball[2]) samples from one SportVU game JSON."""
    d = json.loads(pathlib.Path(json_path).read_text())
    team_flags: dict[int, float] = {}
    for ev in d["events"]:
        moments = ev["moments"]
        for i in range(VEL_LAG, len(moments), stride):
            ents_now = {e[1]: (e[2], e[3]) for e in moments[i][5]}
            ents_prev = {e[1]: (e[2], e[3]) for e in moments[i - VEL_LAG][5]}
            ball_now = next((e for e in moments[i][5] if e[0] == -1), None)
            if ball_now is None:
                continue
            players = []
            for e in moments[i][5]:
                team, pid = e[0], e[1]
                if team == -1 or pid not in ents_prev:
                    continue
                if team not in team_flags:
                    team_flags[team] = 1.0 if len(team_flags) == 0 else -1.0
                x, y = e[2] / COURT_X, e[3] / COURT_Y
                px, py = ents_prev[pid][0] / COURT_X, ents_prev[pid][1] / COURT_Y
                vx, vy = _velocities(np.array([x, y]), np.array([px, py]), VEL_LAG / FPS)
                players.append([x, y, vx, vy, team_flags[team]])
            if len(players) >= 8:
                yield (np.array(players, dtype=np.float32),
                       np.array([ball_now[2] / COURT_X, ball_now[3] / COURT_Y], np.float32))


# ---------- trajectory (temporal window) variants for the v1 specialist ----------
T_STEPS = 5  # steps per window, spaced VEL_LAG frames apart → 1 s at 25 Hz


def _traj_feature(xy_steps, dt):
    """[T,2] positions → flat [x,y,vx,vy]×T (velocity via backward difference)."""
    out = []
    for k in range(len(xy_steps)):
        x, y = xy_steps[k]
        px, py = xy_steps[k - 1] if k > 0 else xy_steps[k]
        out += [x, y, (x - px) / dt, (y - py) / dt]
    return out


def load_metrica_trajectories(game_dir, stride: int = 5):
    """Yield (players[N, 4*T+1], ball[2]) — per-player 1 s trajectories."""
    import csv as _csv
    game_dir = pathlib.Path(game_dir)
    name = game_dir.name
    sides = {}
    for side in ("Home", "Away"):
        rows = list(_csv.reader((game_dir / f"{name}_RawTrackingData_{side}_Team.csv").open()))
        sides[side] = np.array([[float(c) if c not in ("", "NaN") else np.nan for c in r[3:]]
                                for r in rows[3:]])
    n = min(len(sides["Home"]), len(sides["Away"]))
    home, away = sides["Home"][:n], sides["Away"][:n]
    ball = home[:, -2:]
    dt = VEL_LAG / FPS
    span = VEL_LAG * (T_STEPS - 1)
    for i in range(span, n, stride):
        if np.isnan(ball[i]).any():
            continue
        players = []
        for arr, flag in ((home[:, :-2], 1.0), (away[:, :-2], -1.0)):
            for j in range(0, arr.shape[1], 2):
                steps = [arr[i - span + k * VEL_LAG, j:j+2] for k in range(T_STEPS)]
                if any(np.isnan(s).any() for s in steps):
                    continue
                players.append(_traj_feature(steps, dt) + [flag])
        if len(players) >= 8:
            yield np.array(players, dtype=np.float32), ball[i].astype(np.float32)


def load_sportvu_trajectories(json_path, stride: int = 5):
    """Yield (players[N, 4*T+1], ball[2]) from SportVU, per-event windows."""
    d = json.loads(pathlib.Path(json_path).read_text())
    dt = VEL_LAG / FPS
    span = VEL_LAG * (T_STEPS - 1)
    team_flags: dict[int, float] = {}
    for ev in d["events"]:
        mom = ev["moments"]
        maps = [{e[1]: (e[2] / COURT_X, e[3] / COURT_Y) for e in m[5] if e[0] != -1}
                for m in mom]
        balls = [next((e for e in m[5] if e[0] == -1), None) for m in mom]
        teams = [{e[1]: e[0] for e in m[5] if e[0] != -1} for m in mom]
        for i in range(span, len(mom), stride):
            if balls[i] is None:
                continue
            idxs = [i - span + k * VEL_LAG for k in range(T_STEPS)]
            players = []
            for pid, team in teams[i].items():
                if not all(pid in maps[j] for j in idxs):
                    continue
                if team not in team_flags:
                    team_flags[team] = 1.0 if len(team_flags) == 0 else -1.0
                steps = [np.array(maps[j][pid]) for j in idxs]
                players.append(_traj_feature(steps, dt) + [team_flags[team]])
            if len(players) >= 8:
                yield (np.array(players, dtype=np.float32),
                       np.array([balls[i][2] / COURT_X, balls[i][3] / COURT_Y], np.float32))
