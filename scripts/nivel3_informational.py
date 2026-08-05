"""Level-3 informational strand — p(ball | players): when is the ball inferable?

A Mixture Density Network (DeepSets backbone -> K-Gaussian mixture head) predicts the
full posterior over ball position instead of a point. From it:

  1. Calibration: does the model's declared spread track its actual error?
     (bin eval items by predicted std, plot mean error per bin.)
  2. Inferability map: field grid -> median error per cell (heatmap). Where do the
     players determine the ball, and where don't they?
  3. H4: posterior spread vs ball-mass coupling (tangled play = tight; loose = wide).

Soccer field coords (Metrica g1+g2 train... eval g2 held out; + SkillCorner train).
Local, ~$0. Figures -> personal-website/public/blog for potential article use.

Usage: uv run python scripts/nivel3_informational.py
"""
from __future__ import annotations

import json
import math
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from wheres_the_ball.data.field_tracking import (load_metrica_trajectories,
                                                 load_skillcorner_trajectories)
from nivel2_v1_temporal import collate, D_IN

OUT = pathlib.Path("results/nivel3")
FIG = pathlib.Path("/Users/javieraguilarmartin1/Documents/repos/personal-website/public/blog")
K = 5  # mixture components
torch.manual_seed(0)
plt.rcParams.update({"font.size": 12, "figure.dpi": 150})


class MDN(nn.Module):
    def __init__(self, d_in=D_IN, d_h=64, k=K):
        super().__init__()
        self.k = k
        self.phi = nn.Sequential(nn.Linear(d_in, d_h), nn.ReLU(),
                                 nn.Linear(d_h, d_h), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(2 * d_h, d_h), nn.ReLU())
        self.pi = nn.Linear(d_h, k)
        self.mu = nn.Linear(d_h, k * 2)
        self.logsig = nn.Linear(d_h, k * 2)

    def forward(self, players, mask):
        h = self.phi(players)
        m = mask.unsqueeze(-1)
        pooled = torch.cat([(h * m).sum(1) / m.sum(1).clamp(min=1),
                            h.masked_fill(~mask.unsqueeze(-1), -1e9).max(1).values], -1)
        z = self.head(pooled)
        pi = torch.log_softmax(self.pi(z), -1)          # [B,K]
        mu = self.mu(z).view(-1, self.k, 2)             # [B,K,2]
        sig = self.logsig(z).view(-1, self.k, 2).clamp(-6, 2).exp() + 1e-3
        return pi, mu, sig


def nll(pi, mu, sig, y):
    y = y.unsqueeze(1)                                   # [B,1,2]
    log_g = (-0.5 * (((y - mu) / sig) ** 2).sum(-1)
             - torch.log(sig).sum(-1) - math.log(2 * math.pi))  # [B,K]
    return -(torch.logsumexp(pi + log_g, -1)).mean()


def summarize(pi, mu, sig):
    """Return point pred (mixture mean), predictive std, and modal separation."""
    w = pi.exp()                                         # [B,K]
    mean = (w.unsqueeze(-1) * mu).sum(1)                 # [B,2]
    # mixture variance = E[var] + var[E]
    var = (w.unsqueeze(-1) * (sig ** 2 + (mu - mean.unsqueeze(1)) ** 2)).sum(1)
    std = var.sqrt().mean(-1)                            # [B] scalar spread
    # modal separation: distance between the two heaviest components' means
    top2 = w.topk(2, dim=-1).indices
    m0 = torch.gather(mu, 1, top2[:, :1].unsqueeze(-1).expand(-1, 1, 2)).squeeze(1)
    m1 = torch.gather(mu, 1, top2[:, 1:].unsqueeze(-1).expand(-1, 1, 2)).squeeze(1)
    sep = (m0 - m1).norm(dim=-1)
    return mean, std, sep


def coupling(players):  # dist(ball-less): here players only; caller passes ball
    pass


def main() -> None:
    base = pathlib.Path("data/sample-data/data")
    print("Loading soccer trajectories…")
    train = (list(load_metrica_trajectories(base / "Sample_Game_1"))
             + list(load_skillcorner_trajectories("data/opendata/data/matches/1886347"))
             + list(load_skillcorner_trajectories("data/opendata/data/matches/1899585")))
    test = list(load_metrica_trajectories(base / "Sample_Game_2"))
    print(f"train={len(train)} test={len(test)}")

    model = MDN()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    idx = np.arange(len(train))
    for ep in range(20):
        np.random.default_rng(ep).shuffle(idx)
        tot = 0.0
        for i in range(0, len(idx), 256):
            batch = [train[j] for j in idx[i:i + 256]]
            P, M, Y = collate(batch)
            loss = nll(*model(P, M), Y)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(batch)
        if (ep + 1) % 5 == 0:
            print(f"epoch {ep+1}/20 nll={tot/len(idx):.4f}")

    P, M, Y = collate(test)
    with torch.no_grad():
        pi, mu, sig = model(P, M)
    mean, std, sep = summarize(pi, mu, sig)
    mean, std, sep, Yn = mean.numpy(), std.numpy(), sep.numpy(), Y.numpy()
    err = np.linalg.norm(mean - Yn, axis=1)
    print(f"\npoint error median={np.median(err):.4f} (MDN mean)")

    # ball-mass coupling per item (dist ball to speed-weighted player centroid)
    coup = []
    for pl, b in test:
        pos, vel = pl[:, [16, 17]], pl[:, [18, 19]]
        w = np.linalg.norm(vel, axis=1) + 1e-6
        c = (pos * w[:, None]).sum(0) / w.sum()
        coup.append(float(np.linalg.norm(b - c)))
    coup = np.array(coup)

    # --- 1. calibration: error vs declared spread (quintiles) ---
    print("\n1) Calibración — error real por quintil de incertidumbre declarada:")
    q = np.quantile(std, [.2, .4, .6, .8])
    bins = np.digitize(std, q)
    calib = []
    for k in range(5):
        e = err[bins == k]
        calib.append(float(np.median(e)))
        print(f"   quintil {k+1} (std↑): error mediano={np.median(e):.3f}  (n={len(e)})")
    corr_es = float(np.corrcoef(std, err)[0, 1])
    print(f"   corr(std declarada, error real) = {corr_es:+.2f}")

    # --- 3. H4: spread vs coupling ---
    print("\n3) H4 — incertidumbre vs acoplamiento balón-masa:")
    corr_sc = float(np.corrcoef(coup, std)[0, 1])
    tight = std[coup < np.median(coup)]; loose = std[coup >= np.median(coup)]
    print(f"   std mediana: juego trenzado={np.median(tight):.3f}  balón suelto={np.median(loose):.3f}")
    print(f"   corr(acoplamiento, std) = {corr_sc:+.2f}")

    # robustness (a): actual ERROR (not declared std) vs coupling quartiles
    print("   [a] error REAL por cuartil de acoplamiento (bajo=trenzado → alto=suelto):")
    qc = np.quantile(coup, [.25, .5, .75])
    cb = np.digitize(coup, qc)
    for k in range(4):
        print(f"       Q{k+1}: error mediano={np.median(err[cb==k]):.3f} (n={int((cb==k).sum())})")
    corr_ce = float(np.corrcoef(coup, err)[0, 1])
    print(f"       corr(acoplamiento, error real) = {corr_ce:+.2f}")

    # robustness (b): de-confound field position — repeat within distance-to-goal bands
    d2goal = np.minimum(Yn[:, 0], 1 - Yn[:, 0])  # length axis is Yn[:,0]; goals at 0 and 1
    print("   [b] corr(acoplamiento, std) DENTRO de bandas de distancia-a-portería:")
    edges = np.quantile(d2goal, [0, 1/3, 2/3, 1.0])
    for k in range(3):
        m = (d2goal >= edges[k]) & (d2goal <= edges[k + 1])
        if m.sum() > 50:
            r = float(np.corrcoef(coup[m], std[m])[0, 1])
            print(f"       banda {k+1} (d2goal {edges[k]:.2f}-{edges[k+1]:.2f}): corr={r:+.2f} (n={int(m.sum())})")

    # --- 2. inferability map (heatmap of median error over a real football pitch) ---
    gx, gy = 12, 8
    xi = np.clip((Yn[:, 0] * gx).astype(int), 0, gx - 1)
    yi = np.clip((Yn[:, 1] * gy).astype(int), 0, gy - 1)
    grid = np.full((gy, gx), np.nan)
    for cx in range(gx):
        for cy in range(gy):
            e = err[(xi == cx) & (yi == cy)]
            if len(e) >= 15:
                grid[cy, cx] = np.median(e)
    np.save(OUT / "inferability_grid.npy", grid)  # for redraw without retraining

    def draw_pitch(ax, c="#0b3d1e", lw=1.6):
        """FIFA pitch markings in normalized coords: x=length (goal-to-goal), y=width."""
        import matplotlib.patches as mp
        # perimeter + halfway line
        ax.add_patch(mp.Rectangle((0, 0), 1, 1, fill=False, ec=c, lw=lw))
        ax.plot([0.5, 0.5], [0, 1], color=c, lw=lw)
        # centre circle (9.15 m) + spot — normalized radii differ per axis
        ax.add_patch(mp.Ellipse((0.5, 0.5), 2 * 9.15 / 105, 2 * 9.15 / 68, fill=False, ec=c, lw=lw))
        ax.plot(0.5, 0.5, "o", color=c, ms=2.5)
        pen_d, pen_w = 16.5 / 105, 40.32 / 68      # penalty box
        six_d, six_w = 5.5 / 105, 18.32 / 68       # six-yard box
        goal_w = 7.32 / 68
        for x0, sgn in ((0.0, 1), (1.0, -1)):
            ax.add_patch(mp.Rectangle((x0, 0.5 - pen_w / 2), sgn * pen_d, pen_w, fill=False, ec=c, lw=lw))
            ax.add_patch(mp.Rectangle((x0, 0.5 - six_w / 2), sgn * six_d, six_w, fill=False, ec=c, lw=lw))
            ax.add_patch(mp.Rectangle((x0, 0.5 - goal_w / 2), sgn * -0.012, goal_w, fill=False, ec=c, lw=lw))
            ax.plot(x0 + sgn * 11 / 105, 0.5, "o", color=c, ms=2.5)  # penalty spot

    fig, ax = plt.subplots(figsize=(9.0, 6.2))
    im = ax.imshow(grid, origin="upper", cmap="RdYlGn_r", vmin=0.05, vmax=0.20,
                   extent=[0, 1, 1, 0], aspect=68 / 105, interpolation="bilinear", alpha=0.9)
    draw_pitch(ax)
    ax.set_xlim(-0.01, 1.01); ax.set_ylim(1.01, -0.01)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Where is the ball inferable from the players?\n"
                 "median localization error per pitch zone — greener = better determined", fontsize=12)
    ax.text(0.02, 0.5, "goal", rotation=90, va="center", ha="left", fontsize=9, color="#0b3d1e")
    ax.text(0.98, 0.5, "goal", rotation=90, va="center", ha="right", fontsize=9, color="#0b3d1e")
    fig.colorbar(im, ax=ax, label="median localization error", shrink=0.62, pad=0.02)
    fig.tight_layout(); fig.savefig(FIG / "wtb3-inferability-map.png", bbox_inches="tight", dpi=150); plt.close(fig)
    print("\nwrote wtb3-inferability-map.png (over pitch)")

    # bootstrap CIs for the headline sign-contradiction (loose ball: error↑ but std↓)
    print("\n4) Bootstrap 95% CI (n=2000 resamples) del hallazgo de signo opuesto:")
    rng = np.random.default_rng(0)
    bce, bcs, bgap = [], [], []
    for _ in range(2000):
        s = rng.integers(0, len(err), len(err))
        bce.append(np.corrcoef(coup[s], err[s])[0, 1])
        bcs.append(np.corrcoef(coup[s], std[s])[0, 1])
        hi, lo = coup[s] >= np.median(coup[s]), coup[s] < np.median(coup[s])
        bgap.append(np.median(err[s][hi]) - np.median(err[s][lo]))
    ci = lambda a: (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))
    print(f"   corr(acoplamiento, error real) = {np.mean(bce):+.2f}  CI={ci(bce)}")
    print(f"   corr(acoplamiento, std declar.) = {np.mean(bcs):+.2f}  CI={ci(bcs)}")
    print(f"   error(suelto) - error(trenzado) = {np.mean(bgap):+.3f}  CI={ci(bgap)}  (>0 ⇒ suelto más difícil)")

    res = {"point_error_median": float(np.median(err)),
           "ci_corr_coupling_error": ci(bce), "ci_corr_coupling_std": ci(bcs),
           "ci_error_gap_loose_minus_tight": ci(bgap),
           "calibration_quintiles": calib, "corr_std_err": corr_es,
           "coupling_tight_std": float(np.median(tight)),
           "coupling_loose_std": float(np.median(loose)), "corr_coupling_std": corr_sc}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "informational.json").write_text(json.dumps(res, indent=2))
    print(f"Saved {OUT/'informational.json'}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "scripts")
    main()
