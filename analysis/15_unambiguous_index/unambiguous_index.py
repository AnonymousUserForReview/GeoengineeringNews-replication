"""The unambiguous-term index: definition, lead--lag test, and pooled topic model.

The composite index weights terms by their indexed search-result counts, which
places 76% of the weight on six terms whose queries do not unambiguously denote
climate intervention (`bio-energy`, `air capture`, `space sunshade`,
`space shade`, `climate engineering`, `carbon storage`). This script builds the
contrasting construction --- an equal-weighted mean of the standardized series
for the remaining, unambiguous terms --- and re-runs both headline analyses
against it, each against its own scan-corrected null band.

The rule is stated explicitly so the construction is reproducible:
  start from the 26 cleaned tokens, drop the six ambiguous terms above (and the
  `bioenergy` spelling variant of `bio-energy`), standardize each remaining
  series, and take the unweighted mean.

Run from the repository root. Writes unambiguous_index_results.json.
"""

from __future__ import annotations
# --- replication-package prelude (added by build_package.py) ------------------
# The original research scripts pointed at an absolute project root. In this
# package the repository locates itself, so it runs from any checkout location.
from pathlib import Path as _RepoPath
_REPO = str(_RepoPath(__file__).resolve().parents[2])
# ------------------------------------------------------------------------------

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = _RepoPath(_REPO)
OUT = ROOT / "analysis" / "15_unambiguous_index"

UMBRELLA = ["geoengineering", "geoengineer", "climate engineering", "climate engineer"]
SRM = ["aerosol injection", "stratospheric aerosol injection", "albedo modification",
       "high albedo", "high-albedo", "marine cloud brightening", "ocean mirror",
       "space shade", "space sunshade"]
CDR = ["air capture", "carbon capture", "carbon storage", "carbon sequestration",
       "biochar", "enhanced weathering", "ocean fertilization", "ocean fertilisation",
       "ocean iron fertilization", "ocean iron fertilisation", "methane removal",
       "afforestation", "reforestation", "bioenergy", "bio-energy"]
CLEANED = UMBRELLA + SRM + CDR

AMBIGUOUS = ["bio-energy", "bioenergy", "air capture", "space sunshade",
             "space shade", "climate engineering", "carbon storage"]

CATS = ["art", "disaster", "economy", "education", "energy", "medical",
        "nature", "politics", "pollution", "religion", "society", "technology"]
COLS = (["G1", "G2"] + [f"sal_{c}" for c in CATS]
        + ["en_ix", "oth_ix", "s1", "c1", "s2", "c2", "covid", "ukr"])

LEADS = range(-26, 53)
N_SHIFT = 2000
SEED = 42


def zscore(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return (x - x.mean()) / x.std() if x.std() > 0 else x * 0.0


def lag_profile(media: np.ndarray, index: np.ndarray) -> dict[int, float]:
    out = {}
    for lead in LEADS:
        if lead >= 0:
            a, b = media[: len(media) - lead], index[lead:]
        else:
            a, b = media[-lead:], index[: len(index) + lead]
        out[lead] = float(np.corrcoef(a, b)[0, 1])
    return out


def max_stat_band(media: np.ndarray, index: np.ndarray) -> float:
    rng = np.random.default_rng(SEED)
    n = len(index)
    maxima = np.empty(N_SHIFT)
    for i in range(N_SHIFT):
        shifted = np.roll(index, int(rng.integers(53, n - 53)))
        maxima[i] = max(abs(r) for r in lag_profile(media, shifted).values())
    return float(np.percentile(maxima, 95))


def pooled_model(index: np.ndarray, weeks: pd.Series, h: int = 25):
    sal = pd.read_csv(ROOT / "data/07_weekly_series/weekly_topic_salience_by_category.csv")
    sal["Week"] = pd.to_datetime(sal["0"])
    ton = pd.read_csv(ROOT / "data/07_weekly_series/weekly_topic_sentiment_by_category.csv")
    ton["Week"] = pd.to_datetime(ton["0"])

    df = pd.DataFrame({"Week": weeks, "idx": index})
    df = df.merge(sal[["Week"] + CATS].rename(columns={c: f"sal_{c}" for c in CATS}), on="Week")
    df = df.merge(ton[["Week"] + CATS].rename(columns={c: f"ton_{c}" for c in CATS}), on="Week")
    for c in [c for c in df.columns if c.startswith(("sal_", "ton_"))]:
        df[c] = (df[c] - df[c].mean()) / df[c].std()

    d = df.copy()
    d["y"] = d["idx"].shift(-h)
    d["G1"] = d["idx"]
    d["G2"] = d["idx"].shift(1)
    woy = d["Week"].dt.isocalendar().week.astype(float)
    d["s1"], d["c1"] = np.sin(2 * np.pi * woy / 52), np.cos(2 * np.pi * woy / 52)
    d["s2"], d["c2"] = np.sin(4 * np.pi * woy / 52), np.cos(4 * np.pi * woy / 52)
    d["covid"] = ((d.Week >= "2020-03-08") & (d.Week <= "2021-12-26")).astype(float)
    d["ukr"] = (d.Week >= "2022-02-20").astype(float)
    d["en_ix"] = d["sal_energy"] * d["ton_energy"]
    d["oth_ix"] = np.mean([d[f"sal_{c}"] * d[f"ton_{c}"] for c in CATS if c != "energy"], axis=0)
    d = d.dropna(subset=["y", "G2"]).reset_index(drop=True)

    fit = sm.OLS(d["y"], sm.add_constant(d[COLS])).fit(
        cov_type="HAC", cov_kwds={"maxlags": max(h, 25)})
    mains = {c: fit.params[f"sal_{c}"] for c in CATS}
    rank1 = max(mains, key=lambda k: abs(mains[k]))
    order = sorted(CATS, key=lambda c: -abs(mains[c]))
    return {
        "n": int(fit.nobs),
        "energy_b": round(float(fit.params["sal_energy"]), 3),
        "energy_se": round(float(fit.bse["sal_energy"]), 3),
        "energy_p": round(float(fit.pvalues["sal_energy"]), 4),
        "rank1_topic": rank1,
        "energy_rank": order.index("energy") + 1,
        "rank1_b": round(float(mains[rank1]), 3),
        "adjR2": round(float(fit.rsquared_adj), 4),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    gt = pd.read_csv(ROOT / "data/07_weekly_series/gt_weekly_36tokens_2018_2022.csv")
    gt["Week"] = pd.to_datetime(gt["Week"])
    master = pd.read_csv(ROOT / "analysis/08_rebuild/weekly_series_master.csv",
                         parse_dates=["Week"])
    media = master["m_climate_clean_pooled_z"].to_numpy(dtype=float)

    terms = [t for t in CLEANED if t in gt.columns and t not in AMBIGUOUS]
    index = np.mean([zscore(gt[t]) for t in terms], axis=0)

    profile = lag_profile(media, index)
    positive = {k: v for k, v in profile.items() if k > 0}
    peak = max(positive, key=lambda k: positive[k])
    band = max_stat_band(media, index)
    lag1 = float(np.corrcoef(index[:-1], index[1:])[0, 1])

    payload = {
        "definition": "equal-weighted mean of standardized series for cleaned tokens "
                      "excluding the six ambiguous high-weight terms",
        "terms": terms,
        "n_terms": len(terms),
        "excluded_as_ambiguous": AMBIGUOUS,
        "lead_lag": {
            "peak_lead_weeks": int(peak),
            "peak_r": round(positive[peak], 4),
            "max_stat_band_95": round(band, 4),
            "clears_band": bool(abs(positive[peak]) > band),
            "lag1_autocorrelation": round(lag1, 3),
            "r_at_26": round(profile[26], 4),
        },
        "pooled_model_h25": pooled_model(index, gt["Week"]),
        "n_circular_shifts": N_SHIFT,
        "seed": SEED,
    }
    (OUT / "unambiguous_index_results.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
