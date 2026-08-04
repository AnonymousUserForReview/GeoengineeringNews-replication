"""Is the energy result lexical overlap between the topic and the index?

The composite index places 21.9% of its weight on `bio-energy` and further
weight on carbon-capture, carbon-storage and air-capture terms, all of which are
energy infrastructure. A referee will ask whether energy-coverage volume ranks
first simply because the outcome is largely an energy-search index.

This script re-estimates the corrected pooled specification (identical to
run_corrected_pooled_model.py) against indices with the energy-adjacent terms
removed. If the energy coefficient survives removal of the terms that share its
subject matter, the ranking is not an artifact of lexical overlap.

Run from the repository root. Writes energy_lexical_overlap.json.
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
OUT = ROOT / "analysis" / "14_lexical_overlap"

CATS = ["art", "disaster", "economy", "education", "energy", "medical",
        "nature", "politics", "pollution", "religion", "society", "technology"]
HORIZONS = [12, 16, 20, 25, 26]
BLOCK = 25
NBOOT = 1000
SEED = 42

UMBRELLA = ["geoengineering", "geoengineer", "climate engineering", "climate engineer"]
SRM = ["aerosol injection", "stratospheric aerosol injection", "albedo modification",
       "high albedo", "high-albedo", "marine cloud brightening", "ocean mirror",
       "space shade", "space sunshade"]
CDR = ["air capture", "carbon capture", "carbon storage", "carbon sequestration",
       "biochar", "enhanced weathering", "ocean fertilization", "ocean fertilisation",
       "ocean iron fertilization", "ocean iron fertilisation", "methane removal",
       "afforestation", "reforestation", "bioenergy", "bio-energy"]
CLEANED = UMBRELLA + SRM + CDR

BIOENERGY = ["bioenergy", "bio-energy"]
ENERGY_INFRA = BIOENERGY + ["carbon capture", "carbon storage", "air capture"]

VARIANTS = {
    "cleaned weighted (primary)": CLEANED,
    "drop bio-energy": [t for t in CLEANED if t not in BIOENERGY],
    "drop all energy-infrastructure terms": [t for t in CLEANED if t not in ENERGY_INFRA],
}

COLS = (["G1", "G2"] + [f"sal_{c}" for c in CATS]
        + ["en_ix", "oth_ix", "s1", "c1", "s2", "c2", "covid", "ukr"])


def load_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float]]:
    gt = pd.read_csv(ROOT / "data/07_weekly_series/gt_weekly_36tokens_2018_2022.csv")
    gt["Week"] = pd.to_datetime(gt["Week"])
    sal = pd.read_csv(ROOT / "data/07_weekly_series/weekly_topic_salience_by_category.csv")
    sal["Week"] = pd.to_datetime(sal["0"])
    ton = pd.read_csv(ROOT / "data/07_weekly_series/weekly_topic_sentiment_by_category.csv")
    ton["Week"] = pd.to_datetime(ton["0"])
    raw = pd.read_csv(ROOT / "data/01_search_terms/geoengineering_weight.csv",
                      header=None, names=["token", "w"], encoding="utf-8-sig")
    raw["w"] = raw["w"].str.replace(",", "").astype(float)
    return gt, sal, ton, dict(zip(raw["token"].str.strip(), raw["w"]))


def assemble(gt, sal, ton, weights, terms):
    """Reproduce the corrected specification's design frame for one index variant."""
    cols = [c for c in terms if c in gt.columns and c in weights]
    w = np.array([weights[c] for c in cols], dtype=float)
    index = (gt[cols].to_numpy(dtype=float) * w).sum(1) / w.sum()

    df = pd.DataFrame({"Week": gt["Week"], "idx": index})
    df = df.merge(sal[["Week"] + CATS].rename(columns={c: f"sal_{c}" for c in CATS}), on="Week")
    df = df.merge(ton[["Week"] + CATS].rename(columns={c: f"ton_{c}" for c in CATS}), on="Week")
    for c in [c for c in df.columns if c.startswith(("sal_", "ton_"))]:
        df[c] = (df[c] - df[c].mean()) / df[c].std()
    return df, cols, float(w.sum())


def build(df: pd.DataFrame, h: int) -> pd.DataFrame:
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
    return d.dropna(subset=["y", "G2"]).reset_index(drop=True)


def bootstrap(d: pd.DataFrame) -> tuple[list[float], float]:
    rng = np.random.default_rng(SEED)
    n = len(d)
    draws, rank1 = [], 0
    for _ in range(NBOOT):
        idx = np.concatenate(
            [np.arange(s, s + BLOCK) % n for s in rng.integers(0, n, n // BLOCK + 1)]
        )[:n]
        dd = d.iloc[idx]
        try:
            fit = sm.OLS(dd["y"], sm.add_constant(dd[COLS])).fit()
        except Exception:
            continue
        draws.append(fit.params["sal_energy"])
        mains = {c: abs(fit.params[f"sal_{c}"]) for c in CATS}
        if max(mains, key=mains.get) == "energy":
            rank1 += 1
    return (
        [round(float(np.percentile(draws, 2.5)), 3), round(float(np.percentile(draws, 97.5)), 3)],
        round(rank1 / len(draws), 3),
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    gt, sal, ton, weights = load_frames()
    primary_weight = sum(weights[c] for c in CLEANED if c in gt.columns and c in weights)

    payload = {
        "description": "Energy-topic result under indices stripped of energy-adjacent terms",
        "specification": "identical to analysis/09_stats/pooled_model/run_corrected_pooled_model.py",
        "block": BLOCK, "nboot": NBOOT, "seed": SEED, "variants": {},
    }

    for name, terms in VARIANTS.items():
        df, used, total_w = assemble(gt, sal, ton, weights, terms)
        rows = []
        for h in HORIZONS:
            d = build(df, h)
            fit = sm.OLS(d["y"], sm.add_constant(d[COLS])).fit(
                cov_type="HAC", cov_kwds={"maxlags": max(h, 25)}
            )
            mains = {c: fit.params[f"sal_{c}"] for c in CATS}
            rank1 = max(mains, key=lambda k: abs(mains[k]))
            order = sorted(CATS, key=lambda c: -abs(mains[c]))
            entry = {
                "h": h,
                "energy_b": round(float(fit.params["sal_energy"]), 3),
                "energy_se": round(float(fit.bse["sal_energy"]), 3),
                "energy_p": round(float(fit.pvalues["sal_energy"]), 4),
                "rank1_topic": rank1,
                "energy_rank": order.index("energy") + 1,
                "n": int(fit.nobs),
            }
            if h == 25:
                entry["energy_boot_ci"], entry["energy_rank1_share"] = bootstrap(d)
            rows.append(entry)
            print(f"{name:38s} h={h:2d} b={entry['energy_b']:6.3f} "
                  f"p={entry['energy_p']:.4f} rank1={rank1:10s} energy_rank={entry['energy_rank']}")
        payload["variants"][name] = {
            "n_terms": len(used),
            "weight_share_of_primary": round(total_w / primary_weight, 4),
            "horizons": rows,
        }
        print()

    (OUT / "energy_lexical_overlap.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {OUT / 'energy_lexical_overlap.json'}")


if __name__ == "__main__":
    main()
