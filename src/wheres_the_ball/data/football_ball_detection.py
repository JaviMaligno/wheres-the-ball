"""Loader for the `martinjolif/football-ball-detection` HuggingFace dataset.

Layout (YOLO):  data/{train,valid,test}/images/<name>.jpg
                data/{train,valid,test}/labels/<name>.txt   # "cls cx cy w h" normalized

Only the ball is annotated (single class). We download individual image+label pairs on
demand (cached by huggingface_hub) rather than the whole repo.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from huggingface_hub import HfApi, hf_hub_download

REPO_ID = "martinjolif/football-ball-detection"
REPO_TYPE = "dataset"


@dataclass
class Sample:
    name: str
    image_path: str
    cx: float  # normalized ball center (YOLO)
    cy: float
    w: float
    h: float


def _list_label_names(split: str) -> list[str]:
    api = HfApi()
    files = api.list_repo_files(REPO_ID, repo_type=REPO_TYPE)
    prefix = f"data/{split}/labels/"
    return sorted(
        f[len(prefix):-4]  # strip prefix and ".txt"
        for f in files
        if f.startswith(prefix) and f.endswith(".txt")
    )


def _parse_label(path: str) -> tuple[float, float, float, float] | None:
    """Return the ball bbox (cx, cy, w, h) from a YOLO label, or None if empty."""
    with open(path) as fh:
        lines = [ln.strip() for ln in fh if ln.strip()]
    if not lines:
        return None
    # Single-class dataset: take the first (and usually only) box.
    parts = lines[0].split()
    _cls, cx, cy, w, h = parts[:5]
    return float(cx), float(cy), float(w), float(h)


def load_samples(n: int, split: str = "test", seed: int = 0) -> list[Sample]:
    """Download and parse up to `n` samples that actually contain a ball box."""
    names = _list_label_names(split)
    random.Random(seed).shuffle(names)
    out: list[Sample] = []
    for name in names:
        if len(out) >= n:
            break
        label_path = hf_hub_download(
            REPO_ID, f"data/{split}/labels/{name}.txt", repo_type=REPO_TYPE
        )
        box = _parse_label(label_path)
        if box is None:
            continue
        image_path = hf_hub_download(
            REPO_ID, f"data/{split}/images/{name}.jpg", repo_type=REPO_TYPE
        )
        cx, cy, w, h = box
        out.append(Sample(name=name, image_path=image_path, cx=cx, cy=cy, w=w, h=h))
    return out
