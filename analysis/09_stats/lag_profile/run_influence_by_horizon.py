"""Influence diagnostics at every horizon from 19 to 26 weeks, and Figure 2.

For each lead L in 19..26 the correlation between climate coverage (m_climate_clean_pooled_z)
and the search index (idx_cleaned_weighted) is re-estimated after removing every pair of
weeks with either endpoint in an excluded window: each calendar quarter, each calendar
year, and July-December 2021. The exclusion logic is identical to run_lag_profile.py, and
the L = 26 column is asserted against influence.csv from that script.

Run from the repository root. Writes influence_by_horizon.csv next to this script and
manuscript/src/graphs-Round3/fig_influence_main.pdf.
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
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402

ROOT = _RepoPath(_REPO)
OUT = ROOT / "analysis/09_stats/lag_profile"
GRAPHS = ROOT / "manuscript/src/graphs-Round3"
M_COL, G_COL = "m_climate_clean_pooled_z", "idx_cleaned_weighted"
HORIZONS = list(range(19, 27))
MEDIA_Q = {"2021Q3", "2021Q4"}
SEARCH_Q = {"2022Q1", "2022Q2"}


def corr(x, y):
    if len(x) < 10 or x.std() == 0 or y.std() == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def main() -> None:
    master = pd.read_csv(ROOT / "analysis/08_rebuild/weekly_series_master.csv", parse_dates=["Week"])
    weeks = master["Week"].to_numpy()
    m, g = master[M_COL].to_numpy(float), master[G_COL].to_numpy(float)
    band = float(pd.read_csv(OUT / "focal_profile_primary.csv")["band_global"].iloc[0])
    years = pd.DatetimeIndex(weeks).year
    yr = np.clip(years, 2018, 2022)
    qtr_raw = pd.PeriodIndex(pd.DatetimeIndex(weeks), freq="Q")
    lo, hi = pd.Period("2018Q1", "Q").ordinal, pd.Period("2022Q4", "Q").ordinal
    qtr = np.array([str(pd.Period(ordinal=min(max(p.ordinal, lo), hi), freq="Q")) for p in qtr_raw])
    h2 = (weeks >= np.datetime64("2021-07-01")) & (weeks <= np.datetime64("2021-12-31"))
    quarters = sorted(set(qtr))
    periods = [f"drop {y}" for y in range(2018, 2023)] + ["drop 2021H2"]

    rows = []
    for L in HORIZONS:
        n = len(m) - L
        t = np.arange(n)
        te = t + L
        ok = ~np.isnan(m[t]) & ~np.isnan(g[te])
        x, y, t_, te_ = m[t][ok], g[te][ok], t[ok], te[ok]
        rows.append({"L": L, "exclusion": "none (full sample)", "r": corr(x, y), "n": len(x)})
        def excl(mask, label):
            keep = ~(mask[t_] | mask[te_])
            rows.append({"L": L, "exclusion": label, "r": corr(x[keep], y[keep]), "n": int(keep.sum())})
        for y_ in range(2018, 2023):
            excl(yr == y_, f"drop {y_}")
        for q in quarters:
            excl(qtr == q, f"drop {q}")
        excl(h2, "drop 2021H2")
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "influence_by_horizon.csv", index=False)

    # assert the L = 26 column against the archived diagnostics
    old = pd.read_csv(OUT / "influence.csv")
    old = old[old["pair"] == "primary"].set_index("exclusion")["r"]
    new26 = df[df.L == 26].set_index("exclusion")["r"]
    diffs = {k: abs(old[k] - new26[k]) for k in old.index if k in new26.index}
    worst = max(diffs.values())
    assert worst < 1e-6, f"L=26 mismatch vs influence.csv: {worst}"
    print(f"L = 26 reproduces influence.csv (max abs diff {worst:.2e}); band ±{band:.3f}")

    # ---------------- Figure 2: two panels of r(h) after exclusion, horizons as rows
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13.0, 5.4), gridspec_kw={"width_ratios": [20, 6.5]})
    base = {L: df[(df.L == L) & (df.exclusion == "none (full sample)")]["r"].iloc[0] for L in HORIZONS}
    norm = TwoSlopeNorm(vmin=0.0, vcenter=band, vmax=0.6)
    cmap = plt.get_cmap("RdBu")

    def panel(axis, labels, title, xlabels):
        mat = np.array([[df[(df.L == L) & (df.exclusion == lab)]["r"].iloc[0] for lab in labels] for L in HORIZONS])
        axis.imshow(mat, cmap=cmap, norm=norm, aspect="auto")
        for i, L in enumerate(HORIZONS):
            for j in range(len(labels)):
                v = mat[i, j]
                axis.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.2,
                          color="white" if (v < 0.22 or v > 0.5) else "black",
                          fontweight="bold" if v < band else "normal")
        axis.set_yticks(range(len(HORIZONS)))
        axis.set_yticklabels([f"h = {L}" for L in HORIZONS])
        axis.set_xticks(range(len(labels)))
        axis.set_xticklabels(xlabels, rotation=90, fontsize=8)
        axis.set_title(title, loc="left", fontweight="bold")
        for s in axis.spines.values():
            s.set_visible(False)
        return mat

    qlabels = [f"drop {q}" for q in quarters]
    matA = panel(ax, qlabels, "A. Correlation after leaving out one quarter, at each horizon",
                 [q + ("*" if q in MEDIA_Q else "\u2020" if q in SEARCH_Q else "") for q in quarters])
    matB = panel(bx, periods, "B. Leaving out a whole year",
                 ["2018", "2019", "2020 (COVID)", "2021", "2022", "Jul–Dec 2021"])
    fig.subplots_adjust(left=0.06, right=0.99, top=0.9, bottom=0.17, wspace=0.28)
    fig.savefig(GRAPHS / "fig_influence_main.pdf")
    fig.savefig(GRAPHS / "fig_influence_main.png", dpi=300)
    plt.close(fig)
    below_A = int((matA < band).sum()); below_B = int((matB < band).sum())
    print(f"cells below {band:.3f}: quarters {below_A} of {matA.size}; periods {below_B} of {matB.size}")
    # which quarters push any horizon below the level
    hits = {}
    for j, q in enumerate(quarters):
        hs = [HORIZONS[i] for i in range(len(HORIZONS)) if matA[i, j] < band]
        if hs: hits[q] = hs
    print("quarters whose removal pushes r(h) below the level, by horizon:", hits)
    for j, p in enumerate(periods):
        hs = [HORIZONS[i] for i in range(len(HORIZONS)) if matB[i, j] < band]
        print(f"  {p}: below at {hs}")


if __name__ == "__main__":
    main()
