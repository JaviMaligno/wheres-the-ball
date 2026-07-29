"""Redraw the inferability map from the saved grid (no MDN retraining).

Reads results/nivel3/inferability_grid.npy and rewrites wtb3-inferability-map.png with
the pitch overlay. NaN cells (too few samples) are filled by nearest-neighbour so the
bilinear smoothing leaves no white holes. Fast iteration on the figure only.

Usage: uv run python scripts/redraw_inferability.py
"""
from __future__ import annotations

import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.ndimage import distance_transform_edt  # noqa: E402

OUT = pathlib.Path("results/nivel3")
FIG = pathlib.Path("/Users/javieraguilarmartin1/Documents/repos/personal-website/public/blog")
plt.rcParams.update({"font.size": 12, "figure.dpi": 150})


def fill_nan_nearest(a):
    m = np.isnan(a)
    if not m.any():
        return a
    idx = distance_transform_edt(m, return_distances=False, return_indices=True)
    return a[tuple(idx)]


def draw_pitch(ax, c="#0b3d1e", lw=1.6):
    ax.add_patch(mp.Rectangle((0, 0), 1, 1, fill=False, ec=c, lw=lw))
    ax.plot([0.5, 0.5], [0, 1], color=c, lw=lw)
    ax.add_patch(mp.Ellipse((0.5, 0.5), 2 * 9.15 / 105, 2 * 9.15 / 68, fill=False, ec=c, lw=lw))
    ax.plot(0.5, 0.5, "o", color=c, ms=2.5)
    pen_d, pen_w = 16.5 / 105, 40.32 / 68
    six_d, six_w = 5.5 / 105, 18.32 / 68
    goal_w = 7.32 / 68
    for x0, sgn in ((0.0, 1), (1.0, -1)):
        ax.add_patch(mp.Rectangle((x0, 0.5 - pen_w / 2), sgn * pen_d, pen_w, fill=False, ec=c, lw=lw))
        ax.add_patch(mp.Rectangle((x0, 0.5 - six_w / 2), sgn * six_d, six_w, fill=False, ec=c, lw=lw))
        ax.add_patch(mp.Rectangle((x0, 0.5 - goal_w / 2), sgn * -0.012, goal_w, fill=False, ec=c, lw=lw))
        ax.plot(x0 + sgn * 11 / 105, 0.5, "o", color=c, ms=2.5)


def main() -> None:
    grid = fill_nan_nearest(np.load(OUT / "inferability_grid.npy"))
    fig, ax = plt.subplots(figsize=(9.0, 6.2))
    im = ax.imshow(grid, origin="upper", cmap="RdYlGn_r", vmin=0.05, vmax=0.20,
                   extent=[0, 1, 1, 0], aspect=68 / 105, interpolation="bilinear", alpha=0.9)
    draw_pitch(ax)
    ax.set_xlim(-0.01, 1.01); ax.set_ylim(1.01, -0.01)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Where is the ball inferable from the players?\n"
                 "median localization error per pitch zone — greener = better determined", fontsize=12)
    ax.text(0.015, 0.5, "goal", rotation=90, va="center", ha="left", fontsize=9, color="#0b3d1e")
    ax.text(0.985, 0.5, "goal", rotation=90, va="center", ha="right", fontsize=9, color="#0b3d1e")
    fig.colorbar(im, ax=ax, label="median localization error", shrink=0.62, pad=0.02)
    fig.tight_layout(); fig.savefig(FIG / "wtb3-inferability-map.png", bbox_inches="tight", dpi=150); plt.close(fig)
    print("wrote wtb3-inferability-map.png (pitch overlay, NaN-filled)")


if __name__ == "__main__":
    main()
