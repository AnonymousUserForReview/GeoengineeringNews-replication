"""Build main-text Figure 4: topic-coverage main effects under the corrected specification.

The earlier version of this figure was drawn from coefficients estimated without
the post-February-2022 regime indicator, so its energy estimate (3.27) did not
match Table 4 (2.836). This script recomputes every topic's HAC and moving-block
bootstrap interval under the specification actually reported and draws the
figure from those values.

Run from the repository root. Writes the figure and coefs_h25_cleaned_corrected.csv.
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
import statsmodels.api as sm

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = _RepoPath(_REPO)
OUT = ROOT / "analysis/09_stats/pooled_model"
GRAPHS = ROOT / "manuscript/src/graphs-Round3"

CATS = ["art", "disaster", "economy", "education", "energy", "medical",
        "nature", "politics", "pollution", "religion", "society", "technology"]
COLS = (["G1", "G2"] + [f"sal_{c}" for c in CATS]
        + ["en_ix", "oth_ix", "s1", "c1", "s2", "c2", "covid", "ukr"])
H = 25
BLOCK = 25
NBOOT = 1000
SEED = 42


def design() -> pd.DataFrame:
    master = pd.read_csv(ROOT / "analysis/08_rebuild/weekly_series_master.csv",
                         parse_dates=["Week"])
    ps = pd.read_csv(ROOT / "data/07_weekly_series/weekly_topic_salience_by_category.csv")
    ps["Week"] = pd.to_datetime(ps["0"])
    pt = pd.read_csv(ROOT / "data/07_weekly_series/weekly_topic_sentiment_by_category.csv")
    pt["Week"] = pd.to_datetime(pt["0"])
    df = master.merge(ps[["Week"] + CATS].rename(columns={c: f"sal_{c}" for c in CATS}), on="Week")
    df = df.merge(pt[["Week"] + CATS].rename(columns={c: f"ton_{c}" for c in CATS}), on="Week")
    for c in [c for c in df.columns if c.startswith(("sal_", "ton_"))]:
        df[c] = (df[c] - df[c].mean()) / df[c].std()

    d = df.copy()
    G = "idx_cleaned_weighted"
    d["y"] = d[G].shift(-H)
    d["G1"] = d[G]
    d["G2"] = d[G].shift(1)
    woy = d["Week"].dt.isocalendar().week.astype(float)
    d["s1"], d["c1"] = np.sin(2 * np.pi * woy / 52), np.cos(2 * np.pi * woy / 52)
    d["s2"], d["c2"] = np.sin(4 * np.pi * woy / 52), np.cos(4 * np.pi * woy / 52)
    d["covid"] = ((d.Week >= "2020-03-08") & (d.Week <= "2021-12-26")).astype(float)
    d["ukr"] = (d.Week >= "2022-02-20").astype(float)
    d["en_ix"] = d["sal_energy"] * d["ton_energy"]
    d["oth_ix"] = np.mean([d[f"sal_{c}"] * d[f"ton_{c}"] for c in CATS if c != "energy"], axis=0)
    return d.dropna(subset=["y", "G2"]).reset_index(drop=True)


def main() -> None:
    d = design()
    X = sm.add_constant(d[COLS])
    fit = sm.OLS(d["y"], X).fit(cov_type="HAC", cov_kwds={"maxlags": max(H, 25)})
    hac = fit.conf_int()

    rng = np.random.default_rng(SEED)
    n = len(d)
    draws = {f"sal_{c}": [] for c in CATS}
    for _ in range(NBOOT):
        idx = np.concatenate(
            [np.arange(s, s + BLOCK) % n for s in rng.integers(0, n, n // BLOCK + 1)])[:n]
        dd = d.iloc[idx]
        try:
            rr = sm.OLS(dd["y"], sm.add_constant(dd[COLS])).fit()
        except Exception:
            continue
        for c in CATS:
            draws[f"sal_{c}"].append(rr.params[f"sal_{c}"])

    rows = []
    for c in CATS:
        key = f"sal_{c}"
        arr = np.array(draws[key])
        rows.append({
            "topic": c,
            "coef": float(fit.params[key]),
            "HAC_lo": float(hac.loc[key, 0]), "HAC_hi": float(hac.loc[key, 1]),
            "boot_lo": float(np.percentile(arr, 2.5)),
            "boot_hi": float(np.percentile(arr, 97.5)),
            "p": float(fit.pvalues[key]),
        })
    table = pd.DataFrame(rows).sort_values("coef", ascending=False).reset_index(drop=True)
    table.to_csv(OUT / "coefs_h25_cleaned_corrected.csv", index=False)
    print(table.round(3).to_string(index=False))

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    ys = np.arange(len(table))[::-1]
    ax.axvline(0, color="black", lw=0.9, zorder=1)
    for y, (_, r) in zip(ys, table.iterrows()):
        is_energy = r["topic"] == "energy"
        ax.plot([r["HAC_lo"], r["HAC_hi"]], [y, y], lw=5.0,
                color="#b9c9e4" if not is_energy else "#f3c99a", solid_capstyle="butt", zorder=2)
        ax.plot([r["boot_lo"], r["boot_hi"]], [y, y], lw=2.0,
                color="#1f3b73" if not is_energy else "#d95f02", solid_capstyle="butt", zorder=3)
        ax.plot([r["coef"]], [y], "o", ms=5,
                color="#1f3b73" if not is_energy else "#d95f02", zorder=4)
    # print every estimate so the figure can be read without the table
    import json as _json
    spec = _json.loads((OUT / "corrected_final_spec.json").read_text())
    rank1 = float(spec["bootstrap_h25_cleaned"]["rank1_share"])
    for y, (_, r) in zip(ys, table.iterrows()):
        right = max(r["HAC_hi"], r["boot_hi"]) + 0.1
        if r["topic"] == "energy":
            label = (f"{r['coef']:.2f}  [{r['boot_lo']:.2f}, {r['boot_hi']:.2f}]\n"
                     f"first in {100 * rank1:.1f}% of bootstrap resamples")
            ax.text(right, y, label, va="center", ha="left", fontsize=7.5,
                    color="#b34700")
        else:
            ax.text(right, y, f"{r['coef']:.2f}", va="center", ha="left",
                    fontsize=7.5, color="#333333")
    ax.set_xlim(right=max(table["HAC_hi"].max(), table["boot_hi"].max()) + 2.0)
    ax.set_yticks(ys)
    ax.set_yticklabels([t.capitalize() for t in table["topic"]])
    ax.set_xlabel("Standardized coefficient on topic coverage, $h = 25$ "
                  "(index points per one-SD change in coverage)")
    ax.set_title("Pooled dynamic topic model, corrected specification",
                 loc="left", fontweight="bold")
    handles = [
        plt.Line2D([], [], lw=5, color="#b9c9e4", label="HAC 95% CI"),
        plt.Line2D([], [], lw=2, color="#1f3b73", label="moving-block bootstrap 95% CI"),
        plt.Line2D([], [], lw=2, color="#d95f02", label="energy"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=8, loc="upper center",
              bbox_to_anchor=(0.5, -0.13), ncol=3)
    fig.tight_layout()
    fig.savefig(GRAPHS / "fig_topic_coefficients_main.pdf")
    fig.savefig(GRAPHS / "fig_topic_coefficients_main.png", dpi=300)
    plt.close(fig)
    print("\nwrote fig_topic_coefficients_main.pdf")


if __name__ == "__main__":
    main()
