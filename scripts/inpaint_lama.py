"""Remove the ball with LaMa (deep inpainting) via iopaint, run in isolation.

Telea (classical) leaves visible smears on lines and on players (a leak: it reveals
where the ball was). LaMa reconstructs structured backgrounds coherently. iopaint pins
pillow<10, so we run it in its own uvx environment instead of adding it as a dependency.

Inputs come from the build step (scripts/fase1_build_dataset.py):
  results/fase1/original/<stem>.jpg   frames
  results/fase1/masks/<stem>.png      white-on-black ball masks
Output:
  results/fase1/masked/<stem>.jpg     LaMa-inpainted frames (used by the harness)

Usage:
  uv run python scripts/inpaint_lama.py            # defaults to results/fase1
  uv run python scripts/inpaint_lama.py --root results/fase1 --device cpu
"""
from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/fase1")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    args = ap.parse_args()
    root = pathlib.Path(args.root)
    image, mask, out = root / "original", root / "masks", root / "masked"
    out.mkdir(parents=True, exist_ok=True)

    cmd = [
        "uvx", "--from", "iopaint", "iopaint", "run",
        "--model", "lama", "--device", args.device,
        "--image", str(image), "--mask", str(mask), "--output", str(out),
    ]
    print("Running:", " ".join(cmd))
    rc = subprocess.call(cmd)
    if rc != 0:
        sys.exit(rc)
    n = len(list(out.glob("*")))
    print(f"LaMa wrote {n} inpainted frames to {out}/")


if __name__ == "__main__":
    main()
