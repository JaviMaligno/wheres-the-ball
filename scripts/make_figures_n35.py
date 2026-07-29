"""Figures for blog article 4 (Level 3.5 — the model needs motion).

Numbers are the committed, cross-validated results from scripts/nivel35_premise.py and
scripts/nivel35_stillness.py (see docs/nivel-3.5-quietud.md). g1->g2 shown as primary.
Palette matches articles 1-3.

Usage: uv run python scripts/make_figures_n35.py
"""
from __future__ import annotations

import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OUT = pathlib.Path("/Users/javieraguilarmartin1/Documents/repos/personal-website/public/blog")
TEAL, AMBER, GRAPHITE, RED = "#0f9b8e", "#e8973a", "#3a4149", "#c0392b"
plt.rcParams.update({"font.size": 12, "figure.dpi": 150, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.axisbelow": True})

# ---- F1: error by ball state — the still ball is the hard case ----
states = [("settled\n(<2 m/s)", 0.1417, RED), ("slow\n(2–6)", 0.0999, TEAL),
          ("moving\n(6–12)", 0.1045, TEAL), ("flight\n(>12 m/s)", 0.1205, AMBER)]
fig, ax = plt.subplots(figsize=(8.2, 4.8))
x = np.arange(len(states))
ax.bar(x, [s[1] for s in states], 0.62, color=[s[2] for s in states], edgecolor="white")
for xi, s in zip(x, states):
    ax.text(xi, s[1] + 0.003, f"{s[1]:.3f}", ha="center", fontweight="bold", color="#333")
ax.set_xticks(x); ax.set_xticklabels([s[0] for s in states])
ax.set_ylabel("median localization error\n(lower is better)")
ax.set_title("The model needs motion: the still ball is the blind spot",
             fontsize=13, fontweight="bold")
ax.set_ylim(0, 0.16)
ax.text(0, 0.150, "no motion\nto read", ha="center", fontsize=9, color=RED)
fig.tight_layout(); fig.savefig(OUT / "wtb4-ball-state.png", bbox_inches="tight"); plt.close(fig)
print("wrote wtb4-ball-state.png")

# ---- F2: velocity's help collapses when the ball is still (mechanism) ----
# full vs positions-only, still vs moving (g1->g2)
groups = [("ball still", 0.1419, 0.1602), ("ball moving", 0.1053, 0.1445)]
fig, ax = plt.subplots(figsize=(8.0, 4.8))
x = np.arange(len(groups)); w = 0.34
ax.bar(x - w/2, [g[1] for g in groups], w, color=TEAL, edgecolor="white", label="full model (uses velocity)")
ax.bar(x + w/2, [g[2] for g in groups], w, color=GRAPHITE, edgecolor="white", label="positions only")
for i, g in enumerate(groups):
    ax.text(x[i]-w/2, g[1]+0.003, f"{g[1]:.3f}", ha="center", fontsize=9, color="#333")
    ax.text(x[i]+w/2, g[2]+0.003, f"{g[2]:.3f}", ha="center", fontsize=9, color="#333")
    help_ = g[2] - g[1]
    ax.annotate("", xy=(x[i]+w/2, g[1]), xytext=(x[i]+w/2, g[2]),
                arrowprops=dict(arrowstyle="<->", color=RED, lw=1.6))
    ax.text(x[i]+w/2+0.06, (g[1]+g[2])/2, f"velocity\nhelps {help_:+.3f}", color=RED, fontsize=9, va="center")
ax.set_xticks(x); ax.set_xticklabels([g[0] for g in groups])
ax.set_ylabel("median localization error")
ax.set_title("Velocity is the signal — and it's nearly useless when nothing moves",
             fontsize=12.5, fontweight="bold")
ax.legend(loc="upper left", fontsize=10); ax.set_ylim(0, 0.20)
fig.tight_layout(); fig.savefig(OUT / "wtb4-velocity-mechanism.png", bbox_inches="tight"); plt.close(fig)
print("wrote wtb4-velocity-mechanism.png")

# ---- F3: direction in flight is readable ----
fig, ax = plt.subplots(figsize=(8.0, 4.4))
bars = [("learned from\nplayers", 44.8, TEAL), ("constant-mean\nbaseline", 91.7, GRAPHITE),
        ("chance\n(random)", 90.0, GRAPHITE)]
y = np.arange(len(bars))
ax.barh(y, [b[1] for b in bars], color=[b[2] for b in bars], edgecolor="white")
for yi, b in zip(y, bars):
    ax.text(b[1] + 1.2, yi, f"{b[1]:.0f}°", va="center", fontweight="bold", color="#333")
ax.set_yticks(y); ax.set_yticklabels([b[0] for b in bars]); ax.invert_yaxis()
ax.set_xlabel("median angular error of predicted ball direction (lower = better)")
ax.set_title("In flight, which way the ball is going IS readable\n(50% of fast balls pinned within 45°)",
             fontsize=12.5, fontweight="bold")
ax.set_xlim(0, 105)
fig.tight_layout(); fig.savefig(OUT / "wtb4-direction-flight.png", bbox_inches="tight"); plt.close(fig)
print("wrote wtb4-direction-flight.png")
