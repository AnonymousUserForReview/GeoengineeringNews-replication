"""Pooled dynamic topic model across the band of horizons h = 19..26.

The lead--lag scan (RQ1) clears its chance band at L = 19, 20, 22, 24, 25 and 26;
21 and 23 sit just inside it. Single-horizon estimates of the pooled model are
noisy week to week (both series have little week-to-week memory), so the paper
reports the model at every horizon from 19 to 26 and describes the energy
result as a band rather than as a value at one horizon.

For each horizon this script fits the corrected specification (lagged index,
twelve topic-coverage series, two tone interactions, seasonality, pandemic and
post-February-2022 indicators; Newey-West standard errors with maxlags =
max(h, 25)) and records: every topic coefficient with HAC and plain OLS
standard errors and p-values; energy's rank; a moving-block bootstrap (1,000
resamples, block = max(h, 25)) for the energy coefficient and the share of
resamples in which energy ranks first; the energy-by-tone interaction with its
bootstrap interval and the Wald test against the pooled other-topic
interaction; the nested block ladder (adjusted R^2 and expanding-window
out-of-sample RMSE for baseline, +topic volume, +topic tone, +interactions) and
the shifted-tone benchmark (200 circular shifts of the tone block).

Run from the repository root. Writes horizon_band_19_26.json and
horizon_band_19_26.csv next to this script.
"""

from __future__ import annotations
# --- replication-package prelude (added by build_package.py) ------------------
# The original research scripts pointed at an absolute project root. In this
# package the repository locates itself, so it runs from any checkout location.
from pathlib import Path as _RepoPath
_REPO = str(_RepoPath(__file__).resolve().parents[3])
# ------------------------------------------------------------------------------

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

warnings.filterwarnings("ignore")
ROOT = _RepoPath(_REPO)
OUT = ROOT / "analysis/09_stats/pooled_model"
CATS = ["art", "disaster", "economy", "education", "energy", "medical",
        "nature", "politics", "pollution", "religion", "society", "technology"]
BASE = ["G1", "G2", "s1", "c1", "s2", "c2", "covid", "ukr"]
SAL = [f"sal_{c}" for c in CATS]
TON = [f"ton_{c}" for c in CATS]
IX = ["en_ix", "oth_ix"]
COLS = ["G1", "G2"] + SAL + IX + ["s1", "c1", "s2", "c2", "covid", "ukr"]
HORIZONS = list(range(19, 27))
G = "idx_cleaned_weighted"
NBOOT, NBENCH = 1000, 200


def load() -> pd.DataFrame:
    master = pd.read_csv(ROOT / "analysis/08_rebuild/weekly_series_master.csv", parse_dates=["Week"])
    ps = pd.read_csv(ROOT / "data/07_weekly_series/weekly_topic_salience_by_category.csv")
    ps["Week"] = pd.to_datetime(ps["0"])
    pt = pd.read_csv(ROOT / "data/07_weekly_series/weekly_topic_sentiment_by_category.csv")
    pt["Week"] = pd.to_datetime(pt["0"])
    df = master.merge(ps[["Week"] + CATS].rename(columns={c: f"sal_{c}" for c in CATS}), on="Week")
    df = df.merge(pt[["Week"] + CATS].rename(columns={c: f"ton_{c}" for c in CATS}), on="Week")
    for c in SAL + TON:
        df[c] = (df[c] - df[c].mean()) / df[c].std()
    return df


def design(df: pd.DataFrame, h: int) -> pd.DataFrame:
    d = df.copy()
    d["y"] = d[G].shift(-h)
    d["G1"], d["G2"] = d[G], d[G].shift(1)
    w = d["Week"].dt.isocalendar().week.astype(float)
    d["s1"], d["c1"] = np.sin(2 * np.pi * w / 52), np.cos(2 * np.pi * w / 52)
    d["s2"], d["c2"] = np.sin(4 * np.pi * w / 52), np.cos(4 * np.pi * w / 52)
    d["covid"] = ((d.Week >= "2020-03-08") & (d.Week <= "2021-12-26")).astype(float)
    d["ukr"] = (d.Week >= "2022-02-20").astype(float)
    d["en_ix"] = d["sal_energy"] * d["ton_energy"]
    d["oth_ix"] = np.mean([d[f"sal_{c}"] * d[f"ton_{c}"] for c in CATS if c != "energy"], axis=0)
    return d.dropna(subset=["y", "G2"]).reset_index(drop=True)


def oos_rmse(d: pd.DataFrame, cols: list[str]) -> float:
    n = len(d)
    folds = [(int(n * f), int(n * (f + 0.1))) for f in [0.5, 0.6, 0.7, 0.8, 0.9]]
    errs = []
    for a, b in folds:
        tr, te = d.iloc[:a], d.iloc[a:b]
        fc = [c for c in cols if tr[c].std() > 0]
        f = sm.OLS(tr["y"], sm.add_constant(tr[fc])).fit()
        errs.extend((te["y"] - f.predict(sm.add_constant(te[fc], has_constant="add"))) ** 2)
    return float(np.sqrt(np.mean(errs)))


def main() -> None:
    df = load()
    rows, full = [], {}
    for h in HORIZONS:
        d = design(df, h)
        X = sm.add_constant(d[COLS])
        hac = sm.OLS(d["y"], X).fit(cov_type="HAC", cov_kwds={"maxlags": max(h, 25)})
        ols = sm.OLS(d["y"], X).fit()
        ranks = sorted(CATS, key=lambda c: -abs(hac.params[f"sal_{c}"]))
        # bootstrap: energy coefficient, rank-1 share, energy-by-tone interaction
        rng = np.random.default_rng(42)
        n, bl = len(d), max(h, 25)
        b_en, b_ix, b_diff, first = [], [], [], 0
        for _ in range(NBOOT):
            idx = np.concatenate([np.arange(s, s + bl) % n for s in rng.integers(0, n, n // bl + 1)])[:n]
            dd = d.iloc[idx]
            try:
                rr = sm.OLS(dd["y"], sm.add_constant(dd[COLS])).fit()
            except Exception:
                continue
            b_en.append(rr.params["sal_energy"])
            b_ix.append(rr.params["en_ix"])
            b_diff.append(rr.params["en_ix"] - rr.params["oth_ix"])
            sal = {c: abs(rr.params[f"sal_{c}"]) for c in CATS}
            first += max(sal, key=sal.get) == "energy"
        wald = float(hac.t_test("en_ix - oth_ix = 0").pvalue)
        # nested ladder and shifted-tone benchmark
        ladder = {}
        for name, cols in (("baseline", BASE), ("volume", BASE + SAL), ("tone", BASE + SAL + TON), ("interactions", BASE + SAL + TON + IX)):
            m = sm.OLS(d["y"], sm.add_constant(d[cols])).fit(cov_type="HAC", cov_kwds={"maxlags": max(h, 25)})
            ladder[name] = {"adjR2": round(float(m.rsquared_adj), 4), "oos_rmse": round(oos_rmse(d, cols), 3)}
        rng2 = np.random.default_rng(11)
        bench = []
        for _ in range(NBENCH):
            dd = d.copy()
            k = int(rng2.integers(5, n - 5))
            for c in TON:
                dd[c] = np.roll(d[c].to_numpy(), k)
            bench.append(oos_rmse(dd, BASE + SAL + TON))
        row = {
            "h": h, "N": n,
            "energy_b": round(float(hac.params["sal_energy"]), 3),
            "energy_se_hac": round(float(hac.bse["sal_energy"]), 3),
            "energy_p_hac": round(float(hac.pvalues["sal_energy"]), 4),
            "energy_se_ols": round(float(ols.bse["sal_energy"]), 3),
            "energy_p_ols": round(float(ols.pvalues["sal_energy"]), 4),
            "energy_boot_ci": [round(float(np.percentile(b_en, 2.5)), 3), round(float(np.percentile(b_en, 97.5)), 3)],
            "rank1_share": round(first / len(b_en), 3),
            "energy_rank": ranks.index("energy") + 1,
            "runner_up": ranks[1] if ranks[0] == "energy" else ranks[0],
            "adjR2": round(float(hac.rsquared_adj), 3),
            "en_ix_b": round(float(hac.params["en_ix"]), 3),
            "en_ix_p_hac": round(float(hac.pvalues["en_ix"]), 4),
            "en_ix_boot_ci": [round(float(np.percentile(b_ix, 2.5)), 3), round(float(np.percentile(b_ix, 97.5)), 3)],
            "wald_p_en_eq_oth": round(wald, 4),
            "ladder": ladder,
            "benchmark": {"mean": round(float(np.mean(bench)), 3),
                          "ci": [round(float(np.percentile(bench, 2.5)), 3), round(float(np.percentile(bench, 97.5)), 3)]},
        }
        rows.append(row)
        full[h] = {c: {"b": round(float(hac.params[f"sal_{c}"]), 3),
                       "se_hac": round(float(hac.bse[f"sal_{c}"]), 3),
                       "p_hac": round(float(hac.pvalues[f"sal_{c}"]), 4)} for c in CATS}
        full[h].update({k: {"b": round(float(hac.params[k]), 3), "se_hac": round(float(hac.bse[k]), 3),
                            "p_hac": round(float(hac.pvalues[k]), 4)} for k in ["G1", "G2", "en_ix", "oth_ix", "covid", "ukr"]})
        print(f"h={h}: energy {row['energy_b']:.2f} (HAC se {row['energy_se_hac']:.2f}, OLS se {row['energy_se_ols']:.2f}) "
              f"boot {row['energy_boot_ci']} rank1 {row['rank1_share']:.2f} rank {row['energy_rank']} | "
              f"ladder OOS {[ladder[k]['oos_rmse'] for k in ladder]} bench {row['benchmark']['mean']:.2f}", flush=True)

    # band summaries
    eb = [r["energy_b"] for r in rows]
    summary = {
        "horizons": HORIZONS,
        "energy_b_min": min(eb), "energy_b_max": max(eb), "energy_b_mean": round(float(np.mean(eb)), 3),
        "energy_positive_at": sum(b > 0 for b in eb),
        "energy_first_at": sum(r["energy_rank"] == 1 for r in rows),
        "energy_sig05_hac_at": sum(r["energy_p_hac"] < 0.05 for r in rows),
        "energy_sig05_ols_at": sum(r["energy_p_ols"] < 0.05 for r in rows),
        "en_ix_boot_includes_zero_at": sum(r["en_ix_boot_ci"][0] <= 0 <= r["en_ix_boot_ci"][1] for r in rows),
        "volume_improves_oos_at": sum(r["ladder"]["volume"]["oos_rmse"] < r["ladder"]["baseline"]["oos_rmse"] for r in rows),
        "tone_improves_oos_at": sum(r["ladder"]["tone"]["oos_rmse"] < r["ladder"]["volume"]["oos_rmse"] for r in rows),
        "tone_within_benchmark_at": sum(r["benchmark"]["ci"][0] <= r["ladder"]["tone"]["oos_rmse"] <= r["benchmark"]["ci"][1] for r in rows),
    }
    # per-topic band: mean, min, max coefficient and number of horizons ranked first
    topic_band = {}
    for c in CATS:
        bs = [full[h][c]["b"] for h in HORIZONS]
        firsts = sum(1 for h in HORIZONS if max(CATS, key=lambda k: abs(full[h][k]["b"])) == c)
        topic_band[c] = {"mean": round(float(np.mean(bs)), 3), "min": min(bs), "max": max(bs), "first_at": firsts}
    payload = {"summary": summary, "by_horizon": rows, "coefficients": full, "topic_band": topic_band}
    (OUT / "horizon_band_19_26.json").write_text(json.dumps(payload, indent=1) + "\n")
    pd.DataFrame([{k: v for k, v in r.items() if k not in ("ladder", "benchmark")} for r in rows]).to_csv(OUT / "horizon_band_19_26.csv", index=False)
    print("\nSUMMARY", json.dumps(summary))
    print("TOPIC BAND", json.dumps(topic_band))


if __name__ == "__main__":
    main()
