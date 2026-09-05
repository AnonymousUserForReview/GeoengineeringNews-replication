"""Build main-text Figure 1: the corrected lead--lag profile (primary specification).

Reads the archived profile in focal_profile_primary.csv and writes the figure
used by the manuscript. The figure carries only what the text describes: the
curve, its bootstrap ribbon, the null band, the pre-specified 12-20-week
window, and two labelled points (the peak and the upper edge of the window).
Correlations and the band are printed to three decimals because r(20) = 0.354
and the band 0.346 both round to 0.35.

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


def main() -> None:
    profile = pd.read_csv(PROFILE)
    band = float(profile["band_global"].iloc[0])
    leads = profile["L"].to_numpy()
    r = dict(zip(profile["L"], profile["r"]))
    lo = dict(zip(profile["L"], profile["ci_lo"]))
    hi = dict(zip(profile["L"], profile["ci_hi"]))

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    fig, ax = plt.subplots(figsize=(9.2, 4.4))

    # the pre-specified theory window
    ax.axvspan(12, 20, color=WINDOW, alpha=0.16, zorder=0, lw=0)
    ax.text(16, 0.665, "agenda-setting window\n12–20 weeks", ha="center", va="top",
            fontsize=8, color="#8a6d3b")

    # what chance alone produces anywhere in the scan
    ax.axhspan(-band, band, color="#d9d9d9", alpha=0.30, zorder=0, lw=0)
    for sign in (1, -1):
        ax.axhline(sign * band, color=MUTED, lw=0.9, ls=(0, (4, 3)), zorder=2)
    ax.text(51.5, -band - 0.012, f"±{band:.3f}: the largest correlation chance alone\nproduces anywhere in this scan (95% of shuffles)",
            ha="right", va="top", fontsize=7.5, color=MUTED, linespacing=1.2)

    ax.fill_between(leads, profile["ci_lo"], profile["ci_hi"],
                    color=RIBBON, alpha=0.45, lw=0, zorder=1,
                    label="range of estimates on resampled blocks of weeks (95%)")
    ax.plot(leads, profile["r"], color=LINE, lw=2.0, zorder=3)
    ax.axhline(0, color="black", lw=0.8, zorder=2)
    ax.axvline(0, color="#bbbbbb", lw=0.8, zorder=1)

    # the two points the text reads off the figure
    ax.plot([26], [r[26]], "o", color=LINE, ms=5.5, zorder=4)
    ax.text(27.2, r[26] + 0.02,
            f"peak: L = 26 weeks, r = {r[26]:.3f}\n95% interval [{lo[26]:.2f}, {hi[26]:.2f}]",
            ha="left", va="bottom", fontsize=8, color="#1a1a1a", zorder=5)
    ax.plot([20], [r[20]], "o", color=LINE, ms=4.5, zorder=4)
    ax.text(19.5, r[20] - 0.06, f"r(20) = {r[20]:.3f}", ha="center", va="top",
            fontsize=8, color="#1a1a1a", zorder=5)

    # direction of the lead, spelled out at the two ends of the axis
    ax.text(-25, -0.265, "← search interest leads coverage", ha="left", va="center",
            fontsize=8, color="#777777")
    ax.text(51, -0.265, "coverage leads search interest →", ha="right", va="center",
            fontsize=8, color="#777777")

    ax.set_xlim(-26, 52)
    ax.set_ylim(-0.47, 0.72)
    ax.set_xlabel("Lead L (weeks between a week of coverage and the week of search interest it is compared with)")
    ax.set_ylabel("Correlation r(L)")
    ax.legend(frameon=False, fontsize=8, loc="upper left")

    fig.tight_layout()
    fig.savefig(GRAPHS / "fig_lag_profile_main.pdf")
    fig.savefig(GRAPHS / "fig_lag_profile_main.png", dpi=300)
    plt.close(fig)
    print(f"wrote fig_lag_profile_main.pdf (band ±{band:.3f}, peak r(26) = {r[26]:.3f})")


if __name__ == "__main__":
    main()
