"""Generate the article figures from the Phase 1 data (into personal-website/public/blog).

Fig 1: correlation pred-GT per model (x & y) with 95% bootstrap CI — who has signal.
Fig 2: the camera-bias trap — far-bin win-rate vs center at n=14 (looked like 64%)
       vs n=40 (55%, CI includes 50%), with binomial CIs.
Fig 3: multi-view ablation — corr_y across single/temporal/shuffled/lastonly for GPT
       vs Opus (GPT aggregates views, Opus gets diluted; order is irrelevant).
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
FASE1 = pathlib.Path("results/fase1")
rng = np.random.default_rng(0)
TEAL, AMBER, SLATE, GRAPHITE = "#0f9b8e", "#e8973a", "#6b7a8f", "#3a4149"
plt.rcParams.update({"font.size": 12, "figure.dpi": 150,
                     "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True})

man = {i["id"]: i for i in json.loads((FASE1 / "manifest.json").read_text())["items"]}
single = json.loads((FASE1 / "predictions.json").read_text())


def xy(d):
    return (d["x"], d["y"]) if isinstance(d, dict) and d.get("x") is not None and "error" not in d else None


def corr_ci(src, key, axis):
    P, G = [], []
    for i in man:
        p = xy((src.get(i, {}) or {}).get(key))
        if p:
            P.append(p[axis]); G.append((man[i]["gt"]["x"], man[i]["gt"]["y"])[axis])
    P, G = np.array(P), np.array(G)
    r = np.corrcoef(P, G)[0, 1]
    bs = [np.corrcoef(P[idx], G[idx])[0, 1] for idx in (rng.integers(0, len(P), len(P)) for _ in range(5000))]
    return r, np.percentile(bs, 2.5), np.percentile(bs, 97.5)


# ---- Fig 1: correlation per model ----
models = [("claude_opus", "Claude Opus 4.8", AMBER), ("gpt", "GPT-5.4", TEAL),
          ("qwen", "Qwen2.5-VL 7B\n(open)", SLATE), ("claude", "Claude Sonnet 4.6", GRAPHITE)]
fig, ax = plt.subplots(figsize=(8.4, 4.6))
w = 0.38
xs = np.arange(len(models))
for j, (axis, lbl, hatch) in enumerate([(0, "horizontal (x)", None), (1, "vertical (y)", "///")]):
    rs, los, his = [], [], []
    for key, _, _ in models:
        r, lo, hi = corr_ci(single, key, axis)
        rs.append(r); los.append(r - lo); his.append(hi - r)
    ax.bar(xs + (j - 0.5) * w, rs, w, yerr=[los, his], capsize=4,
           color=[m[2] for m in models], edgecolor="white", hatch=hatch,
           alpha=0.75 if j else 1.0, label=lbl,
           error_kw={"ecolor": "#333", "lw": 1.2})
ax.axhline(0, color="#333", lw=1)
ax.set_xticks(xs); ax.set_xticklabels([m[1] for m in models])
ax.set_ylabel("Correlation of prediction with true ball position")
ax.set_title("Does the model track where the ball is?  (n=92, 95% CI)")
ax.legend(title="axis", loc="upper right", framealpha=0.9)
ax.text(0.01, -0.22, "Bars crossing 0 = no reliable signal.", transform=ax.transAxes,
        fontsize=10, color="#666")
fig.tight_layout()
fig.savefig(OUT / "wheres-the-ball-correlation.png", bbox_inches="tight")
plt.close(fig)
print("wrote wheres-the-ball-correlation.png")


# ---- Fig 2: the camera-bias trap ----
def binom_ci(k, n):
    p = k / n
    se = math.sqrt(p * (1 - p) / n)
    return p, 1.96 * se


fig, ax = plt.subplots(figsize=(7.2, 4.4))
pts = [("n=14\n(first look)", 9, 14, AMBER), ("n=40\n(scaled)", 22, 40, TEAL)]
for i, (lbl, k, n, c) in enumerate(pts):
    p, e = binom_ci(k, n)
    ax.bar(i, p * 100, 0.5, yerr=e * 100, capsize=6, color=c, edgecolor="white",
           error_kw={"ecolor": "#333", "lw": 1.4})
    ax.text(i, p * 100 + e * 100 + 2, f"{p*100:.0f}%", ha="center", fontsize=12, fontweight="bold")
ax.axhline(50, color="#c0392b", lw=1.6, ls="--")
ax.text(1.42, 51.5, "chance (50%)", color="#c0392b", fontsize=10, ha="right")
ax.set_xticks([0, 1]); ax.set_xticklabels([p[0] for p in pts])
ax.set_ylabel("GPT-5.4 beats the 'center' baseline\non off-center balls (%)")
ax.set_ylim(0, 100)
ax.set_title("How a small sample almost fooled us")
fig.tight_layout()
fig.savefig(OUT / "wheres-the-ball-sample-trap.png", bbox_inches="tight")
plt.close(fig)
print("wrote wheres-the-ball-sample-trap.png")


# ---- Fig 3: multi-view ablation (corr_y) ----
conds = [("single", "1 frame"), ("temporal", "4 frames\n(in order)"),
         ("shuffled", "4 frames\n(shuffled)"), ("lastonly", "target frame\n(seq. format)")]
srcs = {"single": single}
for c, _ in conds[1:]:
    f = FASE1 / (f"predictions_temporal.json" if c == "temporal" else f"predictions_temporal_{c}.json")
    srcs[c] = json.loads(f.read_text())
fig, ax = plt.subplots(figsize=(8.4, 4.6))
xs = np.arange(len(conds))
for key, lbl, c in [("gpt", "GPT-5.4", TEAL), ("claude_opus", "Claude Opus 4.8", AMBER)]:
    ys = [corr_ci(srcs[cond], key, 1)[0] for cond, _ in conds]
    ax.plot(xs, ys, "o-", color=c, lw=2.4, ms=9, label=lbl)
ax.set_xticks(xs); ax.set_xticklabels([c[1] for c in conds])
ax.set_ylabel("Correlation with ball's vertical position")
ax.set_title("Nobody reads motion — it's a multi-view effect\n(shuffling the frames changes nothing)")
ax.legend(loc="best", framealpha=0.9)
ax.axhline(0, color="#333", lw=0.8)
fig.tight_layout()
fig.savefig(OUT / "wheres-the-ball-multiview.png", bbox_inches="tight")
plt.close(fig)
print("wrote wheres-the-ball-multiview.png")


# ---- Fig 4: informed prompt — neutral vs informed (mean correlation), slope chart ----
def mean_corr(key):
    return (corr_ci(single, key, 0)[0] + corr_ci(single, key, 1)[0]) / 2


rows = [("gpt", "gpt_informed", "GPT-5.4", TEAL),
        ("claude_opus", "claude_opus_informed", "Claude Opus 4.8", AMBER),
        ("claude", "claude_informed", "Claude Sonnet 4.6", GRAPHITE)]
fig, ax = plt.subplots(figsize=(7.6, 4.8))
for base, inf, lbl, c in rows:
    n, i = mean_corr(base), mean_corr(inf)
    ax.plot([0, 1], [n, i], "-o", color=c, lw=2.6, ms=10)
    ax.annotate(f"{lbl}", (1, i), xytext=(8, 0), textcoords="offset points",
                va="center", fontsize=11, color=c, fontweight="bold")
    ax.annotate(f"{n:.2f}", (0, n), xytext=(-8, 0), textcoords="offset points",
                va="center", ha="right", fontsize=10, color=c)
ax.set_xlim(-0.15, 1.65)
ax.set_xticks([0, 1]); ax.set_xticklabels(["neutral prompt", "informed prompt\n(names sport + tactics)"])
ax.set_ylabel("Mean correlation with true ball position")
ax.set_title("Does telling the model the rules help?\nOnly the model with room to grow (GPT)")
ax.axhline(0, color="#333", lw=0.8)
fig.tight_layout()
fig.savefig(OUT / "wheres-the-ball-informed.png", bbox_inches="tight")
plt.close(fig)
print("wrote wheres-the-ball-informed.png")
