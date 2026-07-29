"""Figures for blog article 3 (Level 3 — what the black box computes).

Reads results/nivel3/{geometry,tda,informational}.json and writes wtb3-*.png into
the personal-website public/blog folder. Palette matches articles 1 & 2.

Usage: uv run python scripts/make_figures_n3.py
"""
from __future__ import annotations

import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OUT = pathlib.Path("/Users/javieraguilarmartin1/Documents/repos/personal-website/public/blog")
N3 = pathlib.Path("results/nivel3")
TEAL, AMBER, GRAPHITE, RED = "#0f9b8e", "#e8973a", "#3a4149", "#c0392b"
plt.rcParams.update({"font.size": 12, "figure.dpi": 150, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.axisbelow": True})

geo = json.loads((N3 / "geometry.json").read_text())
tda = json.loads((N3 / "tda.json").read_text())
inf = json.loads((N3 / "informational.json").read_text())

# ---- F1: interpretable geometry recovers the deep net ----
# refs from Level 2 (same splits): centroid untrained baseline, deep = the tiny net
REF = {"soccer": {"centroid": 0.231, "deep": 0.101}, "basket": {"centroid": 0.227, "deep": 0.158}}
fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.8), sharey=True)
for ax, sport, nice in [(axes[0], "soccer", "Soccer"), (axes[1], "basket", "Basketball")]:
    r = REF[sport]
    bars = [("Centroid\n(untrained)", r["centroid"], GRAPHITE),
            ("~10 geometric\nfeatures + GBM", geo[sport]["geo"], TEAL),
            ("Deep net\n(the black box)", r["deep"], AMBER)]
    x = np.arange(len(bars))
    ax.bar(x, [b[1] for b in bars], 0.62, color=[b[2] for b in bars], edgecolor="white")
    for xi, b in zip(x, bars):
        ax.text(xi, b[1] + 0.004, f"{b[1]:.3f}", ha="center", fontweight="bold", color="#333")
    ax.set_xticks(x); ax.set_xticklabels([b[0] for b in bars], fontsize=10)
    ax.set_title(f"{nice}  ·  interpretable geometry recovers {geo[sport]['recovery']:.0%}", fontsize=11.5)
axes[0].set_ylabel("median localization error\n(lower is better)")
fig.suptitle("The black box is (mostly) reading geometry", fontsize=14, fontweight="bold")
fig.tight_layout(); fig.savefig(OUT / "wtb3-geometry-recovers.png", bbox_inches="tight"); plt.close(fig)
print("wrote wtb3-geometry-recovers.png")

# ---- F2: which geometry carries it ----
imp = geo["importance_soccer"]
NAMES = {"vel_centroid": "velocity-weighted centroid", "fastest": "fastest player",
         "spread": "team spread", "home_centroid": "home-team centroid",
         "densest": "densest cluster", "team_contact": "closest opposing pair",
         "converge": "velocity convergence point", "mean_speed": "mean speed",
         "away_centroid": "away-team centroid", "centroid": "plain centroid",
         "max_speed": "max speed"}
items = [(NAMES.get(k, k), v) for k, v in imp.items()][:8][::-1]
fig, ax = plt.subplots(figsize=(8.4, 4.8))
y = np.arange(len(items))
cols = [TEAL if it[0].startswith("velocity") else GRAPHITE for it in items]
ax.barh(y, [it[1] for it in items], color=cols, edgecolor="white")
for yi, it in zip(y, items):
    ax.text(it[1] + 0.002, yi, f"{it[1]:+.3f}", va="center", fontsize=10, color="#333")
ax.set_yticks(y); ax.set_yticklabels([it[0] for it in items], fontsize=10)
ax.set_xlabel("permutation importance (Δ median error when shuffled)")
ax.set_title("One feature dominates: where the running mass is heading", fontsize=12.5, fontweight="bold")
ax.set_xlim(0, max(v for _, v in items) * 1.18)
fig.tight_layout(); fig.savefig(OUT / "wtb3-which-geometry.png", bbox_inches="tight"); plt.close(fig)
print("wrote wtb3-which-geometry.png")

# ---- F3: topology adds nothing ----
fig, ax = plt.subplots(figsize=(8.2, 4.8))
groups = ["Soccer", "Basketball"]
x = np.arange(len(groups)); w = 0.26
series = [("Geometry only", GRAPHITE, [tda["soccer"]["geo"], tda["basket"]["geo"]]),
          ("Topology only", RED, [tda["soccer"]["tda"], tda["basket"]["tda"]]),
          ("Geometry + topology", TEAL, [tda["soccer"]["geo_tda"], tda["basket"]["geo_tda"]])]
for i, (lab, c, vals) in enumerate(series):
    b = ax.bar(x + (i - 1) * w, vals, w, color=c, edgecolor="white", label=lab)
    for xi, v in zip(x + (i - 1) * w, vals):
        ax.text(xi, v + 0.004, f"{v:.3f}", ha="center", fontsize=9, color="#333")
ax.set_xticks(x); ax.set_xticklabels(groups)
ax.set_ylabel("median localization error\n(lower is better)")
ax.set_title("Persistent homology adds nothing over plain geometry", fontsize=12.5, fontweight="bold")
ax.legend(loc="upper right", fontsize=10)
ax.set_ylim(0, 0.30)
fig.tight_layout(); fig.savefig(OUT / "wtb3-topology-nothing.png", bbox_inches="tight"); plt.close(fig)
print("wrote wtb3-topology-nothing.png")

# ---- F4: the blind spot (two panels) ----
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.2, 4.6))
# panel 1: global calibration (error by declared-uncertainty quintile)
q = inf["calibration_quintiles"]
a1.plot(range(1, 6), q, "o-", color=TEAL, lw=2.2, ms=8)
a1.set_xticks(range(1, 6))
a1.set_xlabel("model's declared uncertainty (quintile, low → high)")
a1.set_ylabel("actual median error")
a1.set_title(f"Globally calibrated\n(corr {inf['corr_std_err']:+.2f}: more doubt → more error)", fontsize=11)
# panel 2: the blind spot — tangled vs loose
tight_err, loose_err = 0.087, 0.106       # coupling Q1 vs Q4 (from run log)
tight_std, loose_std = inf["coupling_tight_std"], inf["coupling_loose_std"]
x = np.arange(2); w = 0.36
a2.bar(x - w/2, [tight_err, loose_err], w, color=AMBER, edgecolor="white", label="actual error")
a2.bar(x + w/2, [tight_std, loose_std], w, color=GRAPHITE, edgecolor="white", label="declared uncertainty")
for xi, v in zip(x - w/2, [tight_err, loose_err]):
    a2.text(xi, v + 0.002, f"{v:.3f}", ha="center", fontsize=9, color="#333")
for xi, v in zip(x + w/2, [tight_std, loose_std]):
    a2.text(xi, v + 0.002, f"{v:.3f}", ha="center", fontsize=9, color="#333")
a2.set_xticks(x); a2.set_xticklabels(["ball tangled\nin the crowd", "ball loose\nin open space"])
a2.set_title("The blind spot: error ↑ but the model gets\nMORE confident on loose balls", fontsize=11)
a2.legend(loc="upper left", fontsize=9.5)
a2.set_xlim(-0.55, 1.75)
a2.annotate("", xy=(1.46, loose_err), xytext=(1.46, loose_std),
            arrowprops=dict(arrowstyle="<->", color=RED, lw=1.8))
a2.text(1.52, (loose_err + loose_std) / 2, "gap", color=RED, fontsize=9, va="center")
fig.tight_layout(); fig.savefig(OUT / "wtb3-blind-spot.png", bbox_inches="tight"); plt.close(fig)
print("wrote wtb3-blind-spot.png")
