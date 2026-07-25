"""Figures for the Level-2 blog article (into personal-website/public/blog).

F1 hook   : far-bin win-rate vs center — tiny specialist 82% vs VLMs ~53% (chance).
F2 camera : centroid correlation, image space (-0.58) vs field coords (+0.83).
F3 asym   : zero-shot transfer both directions vs untrained-centroid reference.
F4 arms   : 30-min few-shot, real init vs permuted control vs scratch (both dirs).
"""
from __future__ import annotations

import json
import math
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OUT = pathlib.Path("/Users/javieraguilarmartin1/Documents/repos/personal-website/public/blog")
N2 = pathlib.Path("results/nivel2")
F1 = pathlib.Path("results/fase1")
TEAL, AMBER, GRAPHITE, RED = "#0f9b8e", "#e8973a", "#3a4149", "#c0392b"
plt.rcParams.update({"font.size": 12, "figure.dpi": 150, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.axisbelow": True})


def binom_err(k, n):
    p = k / n
    return 1.96 * math.sqrt(p * (1 - p) / n) * 100


# ---- F1: far win-rate vs center, grouped by INPUT (the honest 2x2-ish view) ----
fig, ax = plt.subplots(figsize=(9.2, 4.8))
groups = [
    ("given the player TRACKS", [("Tiny net\n(trained)", 28, 34, TEAL),
                                 ("GPT-5.4", 12, 34, GRAPHITE),
                                 ("Opus 4.8", 13, 34, GRAPHITE)]),
    ("given the PIXELS", [("Small CNN\n(trained)", 25, 34, TEAL),
                          ("GPT-5.4", 18, 34, GRAPHITE),
                          ("Opus 4.8", 18, 34, GRAPHITE),
                          ("Sonnet 4.6", 13, 34, GRAPHITE)]),
]
import matplotlib.patches as mpatches
xpos, xticks, xlabels = 0.0, [], []
for gname, rows in groups:
    start = xpos
    for lbl, k, n, c in rows:
        ax.bar(xpos, k / n * 100, 0.7, yerr=binom_err(k, n), capsize=4, color=c,
               edgecolor="white", error_kw={"ecolor": "#333", "lw": 1.2})
        ax.text(xpos, k / n * 100 + binom_err(k, n) + 2.5, f"{k/n:.0%}", ha="center",
                fontsize=11, fontweight="bold")
        xticks.append(xpos); xlabels.append(lbl)
        xpos += 1.0
    ax.text((start + xpos - 1) / 2, -32, gname, ha="center", fontsize=11,
            fontweight="bold", color="#333")
    xpos += 0.6
ax.axhline(50, color=RED, ls="--", lw=1.6)
ax.text(xpos - 0.7, 52, "chance", color=RED, fontsize=10, ha="right")
ax.set_xticks(xticks); ax.set_xticklabels(xlabels, fontsize=9)
ax.set_ylabel("Beats the camera bias on\noff-center balls (%)")
ax.set_ylim(0, 108); ax.set_yticks([0, 20, 40, 60, 80, 100])
ax.legend(handles=[mpatches.Patch(color=TEAL, label="trained for the task"),
                   mpatches.Patch(color=GRAPHITE, label="zero-shot generalist")],
          loc="upper right", framealpha=0.9, fontsize=10)
ax.set_title("Same hidden-ball items (n=34): training matters, the input barely does")
fig.tight_layout(); fig.savefig(OUT / "wtb2-david-goliath.png", bbox_inches="tight"); plt.close(fig)
print("wrote wtb2-david-goliath.png")

# ---- F2: the camera lies — centroid correlation in two spaces ----
fig, ax = plt.subplots(figsize=(7.2, 4.4))
vals = [("Broadcast image space\n(camera follows the ball)", -0.58, RED),
        ("Field coordinates\n(no camera)", 0.83, TEAL)]
for i, (lbl, r, c) in enumerate(vals):
    ax.bar(i, r, 0.5, color=c, edgecolor="white")
    ax.text(i, r + (0.05 if r > 0 else -0.09), f"{r:+.2f}", ha="center",
            fontsize=13, fontweight="bold")
ax.axhline(0, color="#333", lw=1)
ax.set_xticks([0, 1]); ax.set_xticklabels([v[0] for v in vals], fontsize=10)
ax.set_ylabel("Correlation: player centroid vs ball position")
ax.set_title("The same statistic, two spaces — the camera flips its sign")
ax.set_ylim(-0.85, 1.0)
fig.tight_layout(); fig.savefig(OUT / "wtb2-camera-lies.png", bbox_inches="tight"); plt.close(fig)
print("wrote wtb2-camera-lies.png")

# ---- F3: zero-shot asymmetry ----
fig, ax = plt.subplots(figsize=(8.0, 4.6))
groups = [("soccer → basketball", 0.347, 0.227), ("basketball → soccer", 0.174, 0.231)]
x = np.arange(2); w = 0.36
zs = ax.bar(x - w/2, [g[1] for g in groups], w, color=AMBER, edgecolor="white",
            label="zero-shot (trained on the other sport)")
ct = ax.bar(x + w/2, [g[2] for g in groups], w, color=GRAPHITE, edgecolor="white",
            label="untrained centroid (reference)")
for i, g in enumerate(groups):
    ax.text(i - w/2, g[1] + .008, f"{g[1]:.2f}", ha="center", fontweight="bold")
    ax.text(i + w/2, g[2] + .008, f"{g[2]:.2f}", ha="center", color="#555")
ax.set_xticks(x); ax.set_xticklabels([g[0] for g in groups])
ax.set_ylabel("Median error (field fractions, lower = better)")
ax.set_ylim(0, 0.40)
ax.set_title("Zero-shot transfer is asymmetric\n(and it's not the amount of data — controlled)")
ax.legend(framealpha=0.9)
fig.tight_layout(); fig.savefig(OUT / "wtb2-asymmetry.png", bbox_inches="tight"); plt.close(fig)
print("wrote wtb2-asymmetry.png")

# ---- F4: 30-min arms, both directions ----
con = json.loads((N2 / "consolidated.json").read_text())["30min"]
rev = json.loads((N2 / "reverse.json").read_text())["30min"]
fig, ax = plt.subplots(figsize=(8.4, 4.6))
x = np.arange(2); w = 0.26
arms = [("real pretraining", TEAL, [np.median(con["soccer"]), np.median(rev["basket"])]),
        ("shuffled-target control", AMBER, [np.median(con["permuted"]), np.median(rev["permuted"])]),
        ("from scratch", GRAPHITE, [np.median(con["scratch"]), np.median(rev["scratch"])])]
for j, (lbl, c, vals) in enumerate(arms):
    ax.bar(x + (j - 1) * w, vals, w, color=c, edgecolor="white", label=lbl)
    for i, v in enumerate(vals):
        ax.text(i + (j - 1) * w, v + .002, f"{v:.3f}", ha="center", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(["soccer → basketball\n(30 min of target data)",
                                      "basketball → soccer\n(30 min of target data)"])
ax.set_ylabel("Median error after fine-tuning")
ax.set_title("Real pretraining wins in 12/13 seeds vs scratch (p=0.002)\nand 9/10 vs the shuffled control (p=0.011)")
ax.legend(framealpha=0.9, fontsize=10)
ax.set_ylim(0, max(max(a[2]) for a in arms) * 1.25)
fig.tight_layout(); fig.savefig(OUT / "wtb2-pretraining-arms.png", bbox_inches="tight"); plt.close(fig)
print("wrote wtb2-pretraining-arms.png")


# ---- F5: asymmetry vanishes without velocity (ratio over in-domain) ----
import json as _json
N3 = pathlib.Path("results/nivel3")
_d = _json.load(open(N3 / "asymmetry_decompose.json"))
_IN = {"basket": 0.158, "soccer": 0.101}  # in-domain refs (same architecture)


def _ratios(v):
    return _d[v]["soccer_to_basket"] / _IN["basket"], _d[v]["basket_to_soccer"] / _IN["soccer"]


_variants = [("Full model\n(positions + velocity)", _ratios("full")),
             ("Positions only\n(velocity removed)", _ratios("pos_only"))]
fig, ax = plt.subplots(figsize=(8.6, 5.0))
x = np.arange(2); w = 0.36
s2b = [v[1][0] for v in _variants]; b2s = [v[1][1] for v in _variants]
ax.bar(x - w / 2, s2b, w, color=AMBER, edgecolor="white", label="soccer -> basketball")
ax.bar(x + w / 2, b2s, w, color=TEAL, edgecolor="white", label="basketball -> soccer")
for i in range(2):
    ax.text(i - w / 2, s2b[i] + .04, f"{s2b[i]:.2f}x", ha="center", fontweight="bold")
    ax.text(i + w / 2, b2s[i] + .04, f"{b2s[i]:.2f}x", ha="center", fontweight="bold")
    # gap bracket BELOW the bars (clear of the legend/labels)
    ax.annotate("", xy=(i - w / 2, 0.14), xytext=(i + w / 2, 0.14),
                arrowprops=dict(arrowstyle="<->", color="#666", lw=1.2))
    ax.text(i, 0.22, f"gap {abs(s2b[i]-b2s[i]):.2f}", ha="center", fontsize=10, color="#666")
ax.axhline(1.0, color=RED, ls="--", lw=1.5)
ax.text(0.5, 1.06, "perfect transfer (= in-domain error)", color=RED, fontsize=9, ha="center")
ax.set_xticks(x); ax.set_xticklabels([v[0] for v in _variants])
ax.set_ylabel("Zero-shot error / in-domain error\n(1.0 = transfers perfectly)")
ax.set_ylim(0, 2.8)
ax.set_title("Remove the velocity channel and the asymmetry vanishes")
ax.legend(framealpha=0.9, loc="upper left")
fig.tight_layout(); fig.savefig(OUT / "wtb2-asymmetry-velocity.png", bbox_inches="tight"); plt.close(fig)
print("wrote wtb2-asymmetry-velocity.png")
