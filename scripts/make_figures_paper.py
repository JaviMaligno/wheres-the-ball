"""Paper figures — all from the SCALED results (12 matches / n=260), not the blog's
2-game numbers. Writes to paper/figures/.

  fig1_vlm.pdf        VLMs vs camera-center in the far bin (the contrast)
  fig2_core.pdf       centroid / deep / interpretable-geometry across 12 matches
  fig3_mechanism.pdf  velocity ablation, still vs moving, across 12 matches

Usage: uv run python scripts/make_figures_paper.py
"""
from __future__ import annotations

import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

N3 = pathlib.Path("results/nivel3")
F1 = pathlib.Path("results/fase1")
FIG = pathlib.Path("paper/figures"); FIG.mkdir(parents=True, exist_ok=True)
TEAL, AMBER, GRAPHITE, RED = "#0f9b8e", "#e8973a", "#3a4149", "#c0392b"
plt.rcParams.update({"font.size": 11, "figure.dpi": 150, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.axisbelow": True, "savefig.bbox": "tight"})

# ---- fig1: trained specialist vs VLMs vs camera-center, same far items ----
vlm = json.loads((F1 / "paper_vlm_benchmark.json").read_text())
bridge = json.loads((F1 / "paper_bridge.json").read_text())
fig, ax = plt.subplots(figsize=(6.8, 3.7))
# specialist first (the trained head-to-head from B3), then the VLMs
sp_lo = (bridge["specialist_far_win"] * 100 - bridge["specialist_far_win_ci"][0])
sp_hi = (bridge["specialist_far_win_ci"][1] - bridge["specialist_far_win"] * 100)
bars = [("Specialist\n(trained)", bridge["specialist_far_win"] * 100, sp_lo, sp_hi, TEAL)]
for name, key in [("GPT-5.4", "gpt"), ("Llama-4-\nMaverick", "llama4")]:
    far = vlm.get(key, {}).get("far")
    if far:
        bars.append((name, far["win"] * 100, (far["win"] - far["win_ci"][0]) * 100,
                     (far["win_ci"][1] - far["win"]) * 100, GRAPHITE))
x = np.arange(len(bars))
ax.bar(x, [b[1] for b in bars], 0.55, color=[b[4] for b in bars], edgecolor="white",
       yerr=[[b[2] for b in bars], [b[3] for b in bars]], capsize=5, error_kw={"ecolor": "#333", "lw": 1.2})
ax.axhline(50, color=RED, ls="--", lw=1.6)
ax.text(len(bars) - 0.5, 51.5, "chance / camera-center", color=RED, fontsize=9, ha="right")
for xi, b in zip(x, bars):
    ax.text(xi, b[1] + b[3] + 2.5, f"{b[1]:.0f}%", ha="center", fontweight="bold", color="#333")
ax.set_xticks(x); ax.set_xticklabels([b[0] for b in bars])
ax.set_ylabel("win-rate vs camera center\n(off-center balls, same items)")
ax.set_ylim(0, 85)
ax.set_title("A tiny trained model beats the camera on off-center balls; VLMs don't",
             fontsize=11, fontweight="bold")
fig.savefig(FIG / "fig1_vlm.pdf"); fig.savefig(FIG / "fig1_vlm.png"); plt.close(fig)
print("wrote fig1_vlm")

# ---- fig2: centroid / tuned deep / geometry across 12 matches (mean ± std) ----
# deep is the TUNED multi-seed DeepSets (paper_deep_tuned.json), not the old lightly-trained one.
tuned = json.loads((N3 / "paper_deep_tuned.json").read_text())
cen = np.array([r["centroid"] for r in tuned]); deep = np.array([r["deep_tuned"] for r in tuned])
geo = np.array([r["geo"] for r in tuned])
recovery = np.array([r["recovery"] for r in tuned])
fig, ax = plt.subplots(figsize=(6.0, 3.6))
vals = [("Centroid\n(untrained)", cen, GRAPHITE), ("Tuned Deep Sets\n(black box)", deep, AMBER),
        ("Interpretable\ngeometry", geo, TEAL)]
x = np.arange(len(vals))
ax.bar(x, [v.mean() for _, v, _ in vals], 0.55, color=[c for _, _, c in vals],
       edgecolor="white", yerr=[v.std() for _, v, _ in vals], capsize=5, error_kw={"ecolor": "#333", "lw": 1.2})
for xi, (_, v, _) in zip(x, vals):
    ax.text(xi, v.mean() + v.std() + 0.006, f"{v.mean():.3f}", ha="center", fontweight="bold", color="#333")
ax.set_xticks(x); ax.set_xticklabels([n for n, _, _ in vals])
ax.set_ylabel("median error (12 matches)")
ax.set_ylim(0, 0.26)
ax.set_title(f"Interpretable geometry recovers {recovery.mean():.0%} of the deep model",
             fontsize=11.5, fontweight="bold")
fig.savefig(FIG / "fig2_core.pdf"); fig.savefig(FIG / "fig2_core.png"); plt.close(fig)
print("wrote fig2_core")

# ---- fig3: velocity ablation, still vs moving, 12 matches ----
st = json.loads((N3 / "paper_scale_stillness.json").read_text())
sf = np.array([r["still_full"] for r in st]); sp = np.array([r["still_pos"] for r in st])
mf = np.array([r["move_full"] for r in st]); mp = np.array([r["move_pos"] for r in st])
fig, ax = plt.subplots(figsize=(6.4, 3.6))
groups = ["ball still\n(<2 m/s)", "ball moving"]
x = np.arange(len(groups)); w = 0.36
ax.bar(x - w/2, [sf.mean(), mf.mean()], w, yerr=[sf.std(), mf.std()], capsize=4,
       color=TEAL, edgecolor="white", label="full (uses velocity)", error_kw={"ecolor": "#333"})
ax.bar(x + w/2, [sp.mean(), mp.mean()], w, yerr=[sp.std(), mp.std()], capsize=4,
       color=GRAPHITE, edgecolor="white", label="positions only", error_kw={"ecolor": "#333"})
for i, (f, p) in enumerate([(sf, sp), (mf, mp)]):
    ax.text(x[i], max(f.mean(), p.mean()) + 0.012, f"+{p.mean()-f.mean():.3f}", ha="center",
            color=RED, fontsize=9, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(groups)
ax.set_ylabel("median error (12 matches)")
ax.set_ylim(0, 0.21); ax.legend(loc="lower center", fontsize=9, ncol=2)
ax.set_title("Velocity is the signal — useless when the ball is still", fontsize=11.5, fontweight="bold")
fig.savefig(FIG / "fig3_mechanism.pdf"); fig.savefig(FIG / "fig3_mechanism.png"); plt.close(fig)
print("wrote fig3_mechanism")
