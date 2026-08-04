"""Cross-retrieval stability check for the Google Trends term series.

Two independent retrievals of the same terms, window, and geography exist on disk:

  * monthly  : data/03_google_trends/gt_monthly_worldwide__<slug>.csv   (retrieved 2025-09-18)
  * weekly   : data/07_weekly_series/gt_weekly_36tokens_2018_2022.csv   (separate retrieval)

Both cover 2018-2022 Worldwide. Aggregating the weekly series to calendar months
and correlating against the independently retrieved monthly series gives a
stability statistic that requires no new collection.

This is deliberately NOT presented as an ICC reliability table: the two retrievals
differ in temporal granularity as well as in retrieval date, so month-level
aggregation removes within-month variation that the weekly request resolves. For
dense series the statistic is therefore a lower bound on agreement; for sparse
series disagreement conflates query-sampling noise with integer quantization.
Both limitations are reported alongside the numbers.

Outputs (analysis/12_cross_retrieval/):
  cross_retrieval_stability.csv   per-term Pearson/Spearman agreement + volume stats
  cross_retrieval_summary.json    headline statistics used in the manuscript
  fig_cross_retrieval.pdf/.png    figure: agreement against series density
"""

from __future__ import annotations
# --- replication-package prelude (added by build_package.py) ------------------
# The original research scripts pointed at an absolute project root. In this
# package the repository locates itself, so it runs from any checkout location.
from pathlib import Path as _RepoPath
_REPO = str(_RepoPath(__file__).resolve().parents[2])
# ------------------------------------------------------------------------------

import glob
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analysis" / "12_cross_retrieval"
WEEKLY = ROOT / "data" / "07_weekly_series" / "gt_weekly_36tokens_2018_2022.csv"
MONTHLY_GLOB = str(ROOT / "data" / "03_google_trends" / "gt_monthly_worldwide__*.csv")
WINDOW = ("2018-01-01", "2022-12-01")
MIN_MONTHS = 24


def _read_monthly(path: str) -> tuple[str, pd.DataFrame] | None:
    """Return (term, monthly frame) for one Google Trends monthly export."""
    lines = Path(path).read_text(encoding="utf-8-sig").splitlines()
    header = re.match(r"Month,(.+): \(Worldwide\)", lines[2]) if len(lines) > 2 else None
    if header is None:
        return None
    term = header.group(1).strip()
    frame = pd.read_csv(path, skiprows=2)
    frame.columns = ["month", "monthly"]
    frame["month"] = pd.to_datetime(frame["month"])
    # Google renders suppressed low volume as "<1"; treat it as the midpoint.
    frame["monthly"] = pd.to_numeric(
        frame["monthly"].astype(str).str.replace("<1", "0.5", regex=False),
        errors="coerce",
    )
    return term, frame


def compute() -> pd.DataFrame:
    weekly = pd.read_csv(WEEKLY)
    weekly["Week"] = pd.to_datetime(weekly["Week"])

    rows: list[dict[str, object]] = []
    for path in sorted(glob.glob(MONTHLY_GLOB)):
        parsed = _read_monthly(path)
        if parsed is None:
            continue
        term, monthly = parsed
        if term not in weekly.columns:
            continue

        week_frame = weekly[["Week", term]].copy()
        week_frame["month"] = week_frame["Week"].dt.to_period("M").dt.to_timestamp()
        aggregated = (
            week_frame.groupby("month")[term]
            .mean()
            .reset_index()
            .rename(columns={term: "weekly_aggregated"})
        )

        joined = monthly.merge(aggregated, on="month").dropna()
        joined = joined[(joined["month"] >= WINDOW[0]) & (joined["month"] <= WINDOW[1])]

        weekly_values = week_frame[term].to_numpy(dtype=float)
        zero_share = float(np.mean(weekly_values == 0))
        record: dict[str, object] = {
            "term": term,
            "n_months": int(len(joined)),
            "weekly_mean": float(np.mean(weekly_values)),
            "weekly_zero_week_share": zero_share,
        }

        estimable = (
            len(joined) >= MIN_MONTHS
            and joined["weekly_aggregated"].std() > 0
            and joined["monthly"].std() > 0
        )
        if estimable:
            record["pearson_r"] = float(
                np.corrcoef(joined["weekly_aggregated"], joined["monthly"])[0, 1]
            )
            record["spearman_rho"] = float(
                joined[["weekly_aggregated", "monthly"]].corr(method="spearman").iloc[0, 1]
            )
            record["estimable"] = True
        else:
            record["pearson_r"] = np.nan
            record["spearman_rho"] = np.nan
            record["estimable"] = False
        rows.append(record)

    return pd.DataFrame(rows).sort_values("pearson_r", ascending=False, na_position="last")


def summarize(table: pd.DataFrame) -> dict[str, object]:
    ok = table[table["estimable"]]
    dense = ok[ok["weekly_zero_week_share"] < 0.40]
    sparse = ok[ok["weekly_zero_week_share"] >= 0.40]
    return {
        "terms_compared": int(len(ok)),
        "terms_non_estimable": int((~table["estimable"]).sum()),
        "months_per_term": int(ok["n_months"].max()) if len(ok) else 0,
        "median_pearson_r": round(float(ok["pearson_r"].median()), 4),
        "median_spearman_rho": round(float(ok["spearman_rho"].median()), 4),
        "n_at_or_above_0.90": int((ok["pearson_r"] >= 0.90).sum()),
        "n_below_0.70": int((ok["pearson_r"] < 0.70).sum()),
        "dense_terms_zero_share_lt_40pct": {
            "n": int(len(dense)),
            "median_pearson_r": round(float(dense["pearson_r"].median()), 4) if len(dense) else None,
            "min_pearson_r": round(float(dense["pearson_r"].min()), 4) if len(dense) else None,
        },
        "sparse_terms_zero_share_ge_40pct": {
            "n": int(len(sparse)),
            "median_pearson_r": round(float(sparse["pearson_r"].median()), 4) if len(sparse) else None,
            "max_pearson_r": round(float(sparse["pearson_r"].max()), 4) if len(sparse) else None,
        },
        "retrieval_a": "monthly Worldwide exports, retrieved 2025-09-18",
        "retrieval_b": "weekly Worldwide series, separate retrieval",
        "statistic": "Pearson/Spearman correlation between the independently retrieved "
        "monthly series and the weekly series aggregated to calendar-month means",
        "not_an_icc": "Granularity differs between retrievals, so this is a stability "
        "check, not an intraclass reliability coefficient.",
    }


def make_figure(table: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ok = table[table["estimable"]].copy()
    dense = ok[ok["weekly_zero_week_share"] < 0.40]
    sparse = ok[ok["weekly_zero_week_share"] >= 0.40]

    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    ax.axhline(0.9, ls="--", lw=0.9, color="#8a8a8a")
    ax.text(0.995, 0.905, "r = 0.90", fontsize=8, color="#8a8a8a", ha="right")
    ax.axhline(0.0, lw=0.8, color="#555555")
    ax.scatter(
        dense["weekly_zero_week_share"], dense["pearson_r"],
        s=52, color="#3b64ad", edgecolor="white", lw=0.6, zorder=3,
        label=f"dense (<40% zero weeks), n={len(dense)}",
    )
    ax.scatter(
        sparse["weekly_zero_week_share"], sparse["pearson_r"],
        s=52, color="#CC79A7", edgecolor="white", lw=0.6, zorder=3,
        label=f"sparse (≥40% zero weeks), n={len(sparse)}",
    )
    for _, row in ok.iterrows():
        if row["pearson_r"] < 0.70 or row["term"] in {"geoengineering"}:
            ax.annotate(
                row["term"], xy=(row["weekly_zero_week_share"], row["pearson_r"]),
                xytext=(4, -2), textcoords="offset points", fontsize=7, color="#2b2b2b",
            )
    ax.set_xlabel("Share of weeks with a value of exactly zero (weekly retrieval)")
    ax.set_ylabel("Cross-retrieval agreement (Pearson r)")
    ax.set_ylim(-0.15, 1.05)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_cross_retrieval.pdf")
    fig.savefig(OUT / "fig_cross_retrieval.png", dpi=300)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    table = compute()
    table.to_csv(OUT / "cross_retrieval_stability.csv", index=False)
    summary = summarize(table)
    (OUT / "cross_retrieval_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    make_figure(table)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
