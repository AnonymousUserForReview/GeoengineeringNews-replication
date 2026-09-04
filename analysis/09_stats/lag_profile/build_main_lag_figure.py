"""Build main-text Figure 2: the corrected lead--lag profile (primary specification).

Reads the archived profile in focal_profile_primary.csv and writes the figure
used by the manuscript. Correlations and the null band are labelled to three
decimals, because at two decimals r(20) = 0.354 and the band 0.346 both print
as 0.35 and the comparison the text makes becomes unreadable.

Run from the repository root.
"""

from __future__ import annotations
# --- replication-package prelude (added by build_package.py) ------------------
# The original research scripts pointed at an absolute project root. In this
# package the repository locates itself, so it runs from any checkout location.
from pathlib import Path as _RepoPath
_REPO = str(_RepoPath(__file__).resolve().parents[3])
# ------------------------------------------------------------------------------

from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = _RepoPath(_REPO)
PROFILE = ROOT / "analysis/09_stats/lag_profile/focal_profile_primary.csv"
GRAPHS = ROOT / "manuscript/src/graphs-Round3"

LINE = "#1f3b73"
RIBBON = "#9db6dd"
MUTED = "#6f6f6f"
WINDOW = "#d8a657"

LABELLED = [12, 16, 20, 25, 26]


def main() -> None:
    profile = pd.read_csv(PROFILE)
    band = float(profile["band_global"].iloc[0])
    leads = profile["L"].to_numpy()

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    fig, ax = plt.subplots(figsize=(9.2, 4.6))

    ax.axvspan(12, 20, color=WINDOW, alpha=0.16, zorder=0, lw=0)
    ax.text(16, 0.655, "McCombs\n12–20 wk", ha="center", va="top",
            fontsize=8, color="#8a6d3b")

    ax.axhspan(-band, band, color="#d9d9d9", alpha=0.30, zorder=0, lw=0)
    for sign in (1, -1):
        ax.axhline(sign * band, color=MUTED, lw=0.9, ls=(0, (4, 3)), zorder=2)
    ax.text(51.5, band + 0.012, f"95% global null band  ±{band:.3f}",
            ha="right", va="bottom", fontsize=8, color=MUTED)

    ax.fill_between(leads, profile["ci_lo"], profile["ci_hi"],
                    color=RIBBON, alpha=0.45, lw=0, zorder=1,
                    label="95% block-bootstrap CI")
    ax.plot(leads, profile["r"], color=LINE, lw=2.0, zorder=3)
    ax.axhline(0, color="black", lw=0.8, zorder=2)
    ax.axvline(0, color="#bbbbbb", lw=0.8, zorder=1)

    # family-corrected threshold for the pre-specified 12-20-week window
    window_thr = float(profile["band_secondary_12_20"].iloc[0])
    ax.plot([12, 20], [window_thr, window_thr], color="#8a6d3b", lw=1.1,
            ls=(0, (2, 2)), zorder=2)
    ax.annotate(f"window threshold {window_thr:.3f}", xy=(13, window_thr),
                xytext=(4, 0.27), ha="right", va="center", fontsize=7.5,
                color="#8a6d3b",
                arrowprops={"arrowstyle": "-", "color": "#8a6d3b", "lw": 0.7})

    lookup = dict(zip(profile["L"], profile["r"]))
    ci_lo = dict(zip(profile["L"], profile["ci_lo"]))
    ci_hi = dict(zip(profile["L"], profile["ci_hi"]))
    for lead in LABELLED:
        value = lookup[lead]
        ax.plot([lead], [value], "o", color=LINE, ms=4.5, zorder=4)
        if lead == 26:
            text = (f"peak r(26) = {value:.3f}\n"
                    f"95% CI [{ci_lo[lead]:.2f}, {ci_hi[lead]:.2f}]")
            offset, ha = (0, 14), "center"
        elif lead == 25:
            ax.annotate(f"r(25) = {value:.3f}", xy=(lead, value), xytext=(33, 0.46),
                        ha="left", va="center", fontsize=8, color="#1a1a1a", zorder=5,
                        arrowprops={"arrowstyle": "-", "color": "#999999", "lw": 0.7})
            continue
        else:
            text, offset, ha = f"r({lead}) = {value:.3f}", (0, -16), "center"
        ax.annotate(text, xy=(lead, value), xytext=offset,
                    textcoords="offset points", ha=ha, fontsize=8,
                    color="#1a1a1a", zorder=5)

    neg = profile[profile["L"] < 0]
    neg_row = neg.loc[neg["r"].abs().idxmax()]
    ax.annotate(f"largest reverse-direction value\n"
                f"|r| = {abs(neg_row['r']):.3f} at L = {int(neg_row['L'])} (inside the band)",
                xy=(neg_row["L"], neg_row["r"]), xytext=(-14, 0.47),
                ha="center", va="bottom", fontsize=7.5, color="#555555",
                arrowprops={"arrowstyle": "-", "color": "#999999", "lw": 0.8})
    ax.annotate("G leads M\n(reverse direction)", xy=(-18, -0.16),
                ha="center", va="center", fontsize=8, color="#777777")

    ax.set_xlim(-26, 52)
    ax.set_ylim(-0.30, 0.72)
    n_leads = len(profile)
    ax.set_xlabel(f"Lead L (weeks; {n_leads} leads scanned from -26 to 52; "
                  "positive = media leads search interest)")
    ax.set_ylabel(r"$\rho(L) = \mathrm{corr}(M_t,\, G_{t+L})$")
    ax.legend(frameon=False, fontsize=8, loc="upper left")

    fig.tight_layout()
    fig.savefig(GRAPHS / "fig_lag_profile_main.pdf")
    fig.savefig(GRAPHS / "fig_lag_profile_main.png", dpi=300)
    plt.close(fig)
    print(f"wrote fig_lag_profile_main.pdf (band ±{band:.3f})")


if __name__ == "__main__":
    main()
