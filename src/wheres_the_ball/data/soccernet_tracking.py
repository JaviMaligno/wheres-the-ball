"""Parser for SoccerNet-Tracking clips (MOT format), read directly from the zip.

Layout per clip `SNMOT-XXX/`:
  img1/NNNNNN.jpg           frames (1-indexed, zero-padded to 6)
  gt/gt.txt                 lines: frame,id,x,y,w,h,conf,cls,vis  (x,y = top-left)
  gameinfo.ini              trackletID_N = <role> [team <side>];<jersey>; + actionClass
  seqinfo.ini               name, frameRate, seqLength, imWidth, imHeight, imExt

We read from the zip on demand (the test split is ~8GB) and only extract the few
frames we actually select for evaluation.
"""
from __future__ import annotations

import configparser
import zipfile
from dataclasses import dataclass, field


@dataclass
class Entity:
    tracklet_id: int
    role: str  # "player" | "goalkeeper" | "referee" | "ball"
    team: str | None  # "left" | "right" | None


@dataclass
class ClipInfo:
    name: str
    width: int
    height: int
    length: int
    fps: float
    action_class: str
    ball_id: int | None
    entities: dict[int, Entity] = field(default_factory=dict)


def _role_of(desc: str) -> tuple[str, str | None]:
    """Parse 'player team left' / 'goalkeepers team right' / 'referee' / 'ball'."""
    d = desc.strip().lower()
    team = None
    if "team left" in d:
        team = "left"
    elif "team right" in d:
        team = "right"
    if d.startswith("ball"):
        return "ball", None
    if d.startswith("referee"):
        return "referee", None
    if d.startswith("goalkeeper"):
        return "goalkeeper", team
    return "player", team


def parse_gameinfo(text: str, name: str) -> ClipInfo:
    cp = configparser.ConfigParser()
    cp.read_string(text)
    seq = cp["Sequence"]
    info = ClipInfo(
        name=name, width=0, height=0, length=0, fps=0.0,
        action_class=seq.get("actionClass", "unknown"), ball_id=None,
    )
    for key, val in seq.items():
        if not key.startswith("trackletid_"):
            continue
        tid = int(key.rsplit("_", 1)[1])
        role, team = _role_of(val.split(";", 1)[0])
        info.entities[tid] = Entity(tid, role, team)
        if role == "ball":
            info.ball_id = tid
    return info


def parse_seqinfo(text: str, info: ClipInfo) -> None:
    cp = configparser.ConfigParser()
    cp.read_string(text)
    seq = cp["Sequence"]
    info.width = int(seq["imWidth"])
    info.height = int(seq["imHeight"])
    info.length = int(seq["seqLength"])
    info.fps = float(seq["frameRate"])


def parse_gt(text: str) -> dict[int, dict[int, tuple[float, float, float, float]]]:
    """frame -> {tracklet_id -> (cx, cy, w, h)} in pixels (center + size)."""
    frames: dict[int, dict[int, tuple[float, float, float, float]]] = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        f, tid, x, y, w, h = line.split(",")[:6]
        f, tid = int(f), int(tid)
        x, y, w, h = float(x), float(y), float(w), float(h)
        frames.setdefault(f, {})[tid] = (x + w / 2, y + h / 2, w, h)
    return frames


@dataclass
class Clip:
    info: ClipInfo
    frames: dict[int, dict[int, tuple[float, float, float, float]]]

    def ball_at(self, frame: int) -> tuple[float, float, float, float] | None:
        if self.info.ball_id is None:
            return None
        return self.frames.get(frame, {}).get(self.info.ball_id)

    def players_at(self, frame: int) -> list[tuple[float, float, Entity]]:
        out = []
        for tid, (cx, cy, _w, _h) in self.frames.get(frame, {}).items():
            ent = self.info.entities.get(tid)
            if ent and ent.role in ("player", "goalkeeper"):
                out.append((cx, cy, ent))
        return out


def load_clip(zf: zipfile.ZipFile, clip_path: str) -> Clip:
    """clip_path like 'test/SNMOT-116'."""
    name = clip_path.rstrip("/").split("/")[-1]
    info = parse_gameinfo(zf.read(f"{clip_path}/gameinfo.ini").decode(), name)
    parse_seqinfo(zf.read(f"{clip_path}/seqinfo.ini").decode(), info)
    frames = parse_gt(zf.read(f"{clip_path}/gt/gt.txt").decode())
    return Clip(info=info, frames=frames)


def list_clips(zf: zipfile.ZipFile) -> list[str]:
    """Return clip paths like 'test/SNMOT-116' (dirs containing gameinfo.ini)."""
    clips = sorted({
        n.rsplit("/", 1)[0] for n in zf.namelist() if n.endswith("/gameinfo.ini")
    })
    return clips
