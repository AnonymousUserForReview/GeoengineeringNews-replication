"""Build main-text Figure 3: influence diagnostics for the peak-lead correlation.

Panel A re-estimates r(26) after excluding each calendar quarter (all pairs with
either endpoint inside the excluded window removed), in chronological order.
Panel B shows the whole-period exclusions. The null band is labelled to three
decimals so it can be compared with the point estimates, which is the
comparison the panel exists to support.

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
INFLUENCE = ROOT / "analysis/09_stats/lag_profile/influence.csv"
PROFILE = ROOT / "analysis/09_stats/lag_profile/focal_profile_primary.csv"
GRAPHS = ROOT / "manuscript/src/graphs-Round3"

# Quarters carrying the AR6/COP26 coverage wave, and the quarters 26 weeks downstream
# in which the search response those pairs are built from falls.
MEDIA_QUARTERS = {"2021Q3", "2021Q4"}
SEARCH_QUARTERS = {"2022Q1", "2022Q2"}
PANDEMIC_QUARTERS = {"2020Q1", "2020Q2", "2020Q3", "2020Q4"}

FULL = "none (full sample)"
WHOLE_PERIOD = ["drop 2018", "drop 2019", "drop 2020", "drop 2021", "drop 2022", "drop 2021H2"]
WHOLE_LABELS = {
    "drop 2018": "2018", "drop 2019": "2019", "drop 2020": "2020 (COVID year)",
    "drop 2021": "2021", "drop 2022": "2022", "drop 2021H2": "Jul–Dec 2021",
}


def main() -> None:
    influence = pd.read_csv(INFLUENCE)
    influence = influence[influence["pair"] == "primary"]
    band = float(pd.read_csv(PROFILE)["band_global"].iloc[0])
    baseline = float(influence.loc[influence["exclusion"] == FULL, "r"].iloc[0])

    quarters = influence[influence["exclusion"].str.match(r"drop \d{4}Q\d")].copy()
    quarters["label"] = quarters["exclusion"].str.replace("drop ", "", regex=False)
    quarters = quarters.sort_values("label").reset_index(drop=True)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, (ax_a, ax_b) = plt.subplots(
        2, 1, figsize=(9.0, 6.4), gridspec_kw={"height_ratios": (1.55, 1.0)})

    # ---- Panel A: leave-one-quarter-out, chronological -------------------------
    for i, row in quarters.iterrows():
        if row["label"] in PANDEMIC_QUARTERS:
            ax_a.axvspan(i - 0.5, i + 0.5, color="#2ca25f", alpha=0.10, zorder=0, lw=0)
    colors = ["#d95f02" if q in MEDIA_QUARTERS else "#d7191c" if q in SEARCH_QUARTERS
              else "#4C78A8" for q in quarters["label"]]
    ax_a.axhspan(-band, band, color="#cccccc", alpha=0.35, zorder=0, lw=0)
    ax_a.axhline(band, color="#6f6f6f", lw=0.9, ls=(0, (4, 3)), zorder=2)
    ax_a.axhline(baseline, color="#333333", lw=1.0, ls=":", zorder=2)
    ax_a.text(-0.3, baseline + 0.008, f"full sample r(26) = {baseline:.3f}",
              ha="left", va="bottom", fontsize=7.5, color="#333333")
    ax_a.text(len(quarters) - 0.6, band - 0.012, f"95% null band ±{band:.3f}",
              ha="right", va="top", fontsize=7.5, color="#6f6f6f")
    ax_a.scatter(range(len(quarters)), quarters["r"], c=colors, s=46,
                 edgecolor="white", lw=0.6, zorder=3)
    ax_a.set_xticks(range(len(quarters)))
    ax_a.set_xticklabels(quarters["label"], rotation=60, ha="right", fontsize=7.5)
    ax_a.set_ylabel("$r(26)$ with quarter excluded")
    ax_a.set_ylim(0.0, 0.56)
    ax_a.set_title("A. Leave-one-quarter-out, in chronological order",
                   loc="left", fontweight="bold")
    handles = [
        plt.Line2D([], [], marker="o", ls="", color="#d95f02", label="AR6 / COP26 media quarters"),
        plt.Line2D([], [], marker="o", ls="", color="#d7191c", label="search quarters 26 weeks later"),
        plt.Line2D([], [], marker="o", ls="", color="#4C78A8", label="other quarters"),
        plt.Line2D([], [], marker="s", ls="", color="#2ca25f", alpha=0.35, label="pandemic year"),
    ]
    ax_a.legend(handles=handles, frameon=False, fontsize=7.5, ncol=2, loc="lower left")

    # ---- Panel B: whole-period exclusions --------------------------------------
    rows = [influence.loc[influence["exclusion"] == e].iloc[0] for e in WHOLE_PERIOD]
    values = [float(r["r"]) for r in rows]
    labels = [WHOLE_LABELS[e] for e in WHOLE_PERIOD]
    bar_colors = ["#d7191c" if v < band else "#4C78A8" for v in values]
    ax_b.axvspan(-band, band, color="#cccccc", alpha=0.35, zorder=0, lw=0)
    ax_b.axvline(band, color="#6f6f6f", lw=0.9, ls=(0, (4, 3)), zorder=2)
    ax_b.axvline(baseline, color="#333333", lw=1.0, ls=":", zorder=2)
    ax_b.barh(range(len(values)), values, color=bar_colors, height=0.6, zorder=3)
    for i, v in enumerate(values):
        ax_b.text(v + 0.008, i, f"{v:.3f}", va="center", fontsize=8)
    ax_b.set_yticks(range(len(values)))
    ax_b.set_yticklabels(labels, fontsize=8)
    ax_b.invert_yaxis()
    ax_b.set_xlim(0, 0.60)
    ax_b.set_xlabel("$r(26)$ with the period excluded")
    ax_b.set_title("B. Whole-period exclusions (red: falls inside the null band)",
                   loc="left", fontweight="bold")

    fig.tight_layout()
    fig.savefig(GRAPHS / "fig_influence_main.pdf")
    fig.savefig(GRAPHS / "fig_influence_main.png", dpi=300)
    plt.close(fig)
    print(f"wrote fig_influence_main.pdf (band ±{band:.3f}, baseline {baseline:.3f})")


if __name__ == "__main__":
    main()
