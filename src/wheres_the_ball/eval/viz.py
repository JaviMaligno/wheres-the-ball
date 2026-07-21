"""Visualization: overlay ground truth vs model predictions on the masked frame."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_COLORS = {
    "gt": "#00ff88",
    "center": "#888888",
}
_MODEL_COLORS = ["#ff3b30", "#0a84ff", "#ffcc00", "#bf5af2"]


def plot_item(
    image_rgb,
    gt: tuple[float, float],
    predictions: dict[str, tuple[float, float]],
    out_path: str,
    title: str = "",
) -> None:
    """Save a figure with GT (green), center baseline (grey), and each model."""
    h, w = image_rgb.shape[:2]
    fig, ax = plt.subplots(figsize=(w / 120, h / 120), dpi=120)
    ax.imshow(image_rgb)
    ax.scatter([gt[0] * w], [gt[1] * h], s=180, marker="*",
               edgecolors="black", c=_COLORS["gt"], label="ground truth", zorder=5)
    ci = 0
    for name, (px, py) in predictions.items():
        if name == "center":
            c = _COLORS["center"]
        else:
            c = _MODEL_COLORS[ci % len(_MODEL_COLORS)]
            ci += 1
        ax.scatter([px * w], [py * h], s=120, marker="X",
                   edgecolors="black", c=c, label=name, zorder=4)
    ax.set_title(title, fontsize=9)
    ax.axis("off")
    ax.legend(loc="upper right", fontsize=7, framealpha=0.8)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
