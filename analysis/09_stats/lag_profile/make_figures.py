# --- replication-package prelude (added by build_package.py) ------------------
# The original research scripts pointed at an absolute project root. In this
# package the repository locates itself, so it runs from any checkout location.
from pathlib import Path as _RepoPath
_REPO = str(_RepoPath(__file__).resolve().parents[3])
# ------------------------------------------------------------------------------
#!/usr/bin/env python3
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = (_REPO + "/analysis/09_stats/lag_profile")

INK = "#1F2430"; MUTED = "#6B7280"
BLUE = "#3056D3"; BLUE_FILL = "#3056D3"
ORANGE = "#D9640D"; GREY_BAND = "#9AA1AC"

plt.rcParams.update({
    "font.size": 9.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.linewidth": 0.8, "font.family": "Helvetica",
})

def style(ax):
    ax.spines[["top", "right"]].set_visible(False)

# ------------------------------------------------------------- Figure 1
titles = {
    "original": "Original measures (full corpus $\\times$ weighted 34-token index)",
    "primary": "Audited measures (climate-clean coverage $\\times$ cleaned index)",
}
fig, axes = plt.subplots(2, 1, figsize=(8.2, 7.2), sharex=True)
for ax, name in zip(axes, ["original", "primary"]):
    p = pd.read_csv(f"{OUT}/focal_profile_{name}.csv")
    band = p["band_global"].iloc[0]
    band_sec = p["band_secondary_12_20"].iloc[0]

    # negative-lag region separated
    ax.axvspan(-26.5, 0, color="#000000", alpha=0.045, zorder=0, lw=0)
    ax.axvline(0, color=MUTED, lw=0.8, zorder=1)
    # McCombs 12-20 week window
    ax.axvspan(12, 20, color=GREY_BAND, alpha=0.22, zorder=0, lw=0)
    # global max-stat null band
    ax.axhspan(-band, band, color=GREY_BAND, alpha=0.14, zorder=0, lw=0)
    ax.axhline(band, color=MUTED, lw=0.9, ls=(0, (4, 3)), zorder=2)
    ax.axhline(-band, color=MUTED, lw=0.9, ls=(0, (4, 3)), zorder=2)
    ax.axhline(0, color=MUTED, lw=0.6, zorder=1)
    # bootstrap ribbon + r(L)
    ax.fill_between(p.L, p.ci_lo, p.ci_hi, color=BLUE_FILL, alpha=0.16,
                    lw=0, zorder=2, label="95% block-bootstrap CI (block = 13)")
    ax.plot(p.L, p.r, color=BLUE, lw=2.0, zorder=4, label=r"$\rho(L)$")
    # vertical line at L=25
    ax.axvline(25, color=ORANGE, lw=1.1, ls=":", zorder=3)
    # annotations at 12,16,20,25
    for L in (12, 16, 20, 25):
        r = p.loc[p.L == L, "r"].iloc[0]
        ax.plot(L, r, "o", ms=5, color=ORANGE, mec="white", mew=0.9, zorder=5)
        if L == 25:
            xy, ha = (L + 4.5, r + 0.10), "left"
        else:
            xy, ha = (L - 8.2, r + 0.085), "left"
        ax.annotate(f"r({L}) = {r:.2f}", (L, r), xytext=xy, ha=ha,
                    fontsize = 8, color=INK,
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
    ax.text(24.4, -0.33, "L = 25\n(featured pooled-\nmodel lead)", fontsize=7.5,
            color=ORANGE, ha="right", va="bottom")
    ax.text(16, 0.705, "McCombs\n12–20 wk", fontsize=7.5, color=MUTED,
            ha="center", va="top")
    ax.text(-25.5, 0.55, "G leads M\n(reverse direction)", fontsize=7.5,
            color=MUTED, ha="left", va="top")
    ax.text(51.5, band + 0.015, f"95% global null band  ±{band:.3f}",
            fontsize=7.5, color=MUTED, ha="right", va="bottom")
    ax.set_ylim(-0.35, 0.72)
    ax.set_ylabel(r"$\rho(L)=\mathrm{corr}(M_t,\,G_{t+L})$")
    ax.set_title(titles[name], fontsize=10, loc="left", color=INK, pad=8)
    style(ax)
axes[0].legend(loc="upper left", bbox_to_anchor=(0.0, 1.0), frameon=False,
               fontsize=8, handlelength=1.6)
axes[1].set_xlabel("Lead L (weeks; positive = coverage leads search interest)")
axes[1].set_xticks(np.arange(-26, 53, 13))
fig.tight_layout()
fig.savefig(f"{OUT}/fig_lag_profile.png", dpi=300)
fig.savefig(f"{OUT}/fig_lag_profile.pdf")
plt.close(fig)

# ------------------------------------------------------------- Figure 2: forest
inf = pd.read_csv(f"{OUT}/influence.csv")
fig, axes = plt.subplots(1, 2, figsize=(9.2, 7.6), sharey=True)
short = {"original": "Original pair\n(m_all46 × idx_orig34_weighted)",
         "primary": "Primary pair\n(Audited measures × )"}
order = None
for ax, name in zip(axes, ["original", "primary"]):
    s = inf[inf.pair == name].reset_index(drop=True)
    full = s.loc[s.exclusion == "none (full sample)", "r"].iloc[0]
    Lpk = int(s.L_peak.iloc[0])
    s = s[s.exclusion != "none (full sample)"].reset_index(drop=True)
    if order is None:
        order = list(s.exclusion)
    ypos = np.arange(len(s))[::-1]
    ax.axvline(0, color=MUTED, lw=0.6)
    ax.axvline(full, color=BLUE, lw=1.3, ls="--", zorder=2)
    for y, (_, row) in zip(ypos, s.iterrows()):
        hl = row.exclusion == "drop 2021H2"
        c = ORANGE if hl else INK
        if hl:
            ax.axhspan(y - 0.5, y + 0.5, color=ORANGE, alpha=0.10, lw=0)
        ax.plot([full, row.r], [y, y], color=c, lw=0.8, alpha=0.5, zorder=3)
        ax.plot(row.r, y, "o", ms=6 if hl else 4.5, color=c,
                mec="white", mew=0.8, zorder=4)
        if hl:
            ax.annotate(f"{row.r:.2f}", (row.r, y), xytext=(row.r, y + 0.75),
                        fontsize=8, color=ORANGE, ha="center", fontweight="bold")
    ax.text(full, len(order) + 0.4, f"full sample\nr({Lpk}) = {full:.2f}", fontsize=8,
            color=BLUE, ha="center", va="bottom")
    ax.set_yticks(ypos)
    ax.set_yticklabels(order, fontsize=8)
    ax.set_ylim(-1, len(order) + 2.4)
    ax.set_xlim(0.0, 0.62)
    ax.set_xlabel(f"r at peak lead L = {Lpk} under exclusion")
    ax.set_title(short[name], fontsize=9.5, loc="left", color=INK, pad=6)
    style(ax)
fig.suptitle("Influence diagnostics: peak-lead correlation under sample exclusions "
             "(pairs dropped if either endpoint falls in window)",
             fontsize=10, x=0.02, ha="left", color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.965))
fig.savefig(f"{OUT}/fig_influence_forest.png", dpi=300)
fig.savefig(f"{OUT}/fig_influence_forest.pdf")
plt.close(fig)
print("figures saved")
