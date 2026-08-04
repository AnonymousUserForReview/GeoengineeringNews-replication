"""Sensitivity of the lead--lag result to the climate-token retention rule.

The climate-coverage predictor retains nine audited climate tokens and drops
`natural disaster`, whose matched articles are only 17% climate-related. That
threshold is a judgment call: `overfishing` (0.37) and `climate change` (0.40)
also fall below one half on the same criterion. This script re-estimates the
primary lead--lag result under alternative retention rules so the sensitivity is
reported rather than asserted.

For each coverage variant the peak positive-lead correlation with the cleaned
weighted index is computed together with that variant's own two-sided 95%
max-statistic band from circular shifts of the index.

Outputs analysis/13_coverage_sensitivity/coverage_token_sensitivity.json
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

ROOT = _RepoPath(_REPO)
OUT = ROOT / "analysis" / "13_coverage_sensitivity"

CLEANED_GEO = [
    "geoengineering", "geoengineer", "climate engineering", "climate engineer",
    "aerosol injection", "stratospheric aerosol injection", "albedo modification",
    "high albedo", "high-albedo", "marine cloud brightening", "ocean mirror",
    "space shade", "space sunshade",
    "air capture", "carbon capture", "carbon storage", "carbon sequestration",
    "biochar", "enhanced weathering", "ocean fertilization", "ocean fertilisation",
    "ocean iron fertilization", "ocean iron fertilisation", "methane removal",
    "afforestation", "reforestation", "bioenergy", "bio-energy",
]

BASELINE = [
    "biodiversity loss", "deforestation", "sea level rise", "plastic pollution",
    "ocean acidification", "air pollution", "global warming", "overfishing",
    "climate change",
]

VARIANTS = {
    "primary (natural disaster dropped)": BASELINE,
    "natural disaster retained": BASELINE + ["natural disaster"],
    "also drop overfishing (p_cli 0.37)": [t for t in BASELINE if t != "overfishing"],
    "also drop climate change (p_cli 0.40)": [t for t in BASELINE if t != "climate change"],
    "drop all three below 0.50": [
        t for t in BASELINE if t not in {"overfishing", "climate change"}
    ],
    "majority-climate rule (p_cli > 0.50 only)": [
        "biodiversity loss", "deforestation", "sea level rise", "plastic pollution",
        "ocean acidification", "air pollution", "global warming",
    ],
}

MAX_LEAD = 52
MIN_LEAD = -26
N_SHIFT = 4000
SEED = 42


def has_any(cell: str, tokens: set[str]) -> bool:
    return bool(set(cell.split("|")) & tokens) if cell else False


def build_coverage(prov: pd.DataFrame, keep: list[str], weeks: pd.Series) -> np.ndarray:
    """Pooled, per-outlet standardized weekly count for a climate token set."""
    keep_set = set(keep)
    sel = prov[
        prov["climate_tokens"].apply(lambda c: has_any(c, keep_set))
        & ~prov["is_geo_specific"]
    ].copy()
    # Sunday-start weeks, matching the Google Trends anchor used in the rebuild.
    sel["wkstart"] = sel["date"].dt.to_period("W-SAT").dt.start_time
    table = sel.groupby(["wkstart", "outlet"]).size().unstack(fill_value=0)
    table = table.reindex(weeks.to_numpy(), fill_value=0)

    pooled = []
    for outlet in ("bbc", "nytimes"):
        counts = (
            table[outlet].to_numpy(dtype=float)
            if outlet in table.columns
            else np.zeros(len(table))
        )
        pooled.append((counts - counts.mean()) / counts.std())
    return np.mean(pooled, axis=0)


def lag_profile(media: np.ndarray, index: np.ndarray) -> dict[int, float]:
    profile = {}
    for lead in range(MIN_LEAD, MAX_LEAD + 1):
        if lead >= 0:
            a, b = media[: len(media) - lead], index[lead:]
        else:
            a, b = media[-lead:], index[: len(index) + lead]
        profile[lead] = float(np.corrcoef(a, b)[0, 1])
    return profile


def max_stat_band(media: np.ndarray, index: np.ndarray, rng: np.random.Generator) -> float:
    n = len(index)
    low, high = 53, n - 53
    maxima = np.empty(N_SHIFT)
    for i in range(N_SHIFT):
        shifted = np.roll(index, int(rng.integers(low, high)))
        maxima[i] = max(abs(r) for r in lag_profile(media, shifted).values())
    return float(np.percentile(maxima, 95))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    master = pd.read_csv(ROOT / "analysis" / "08_rebuild" / "weekly_series_master.csv")
    master["Week"] = pd.to_datetime(master["Week"])
    weeks = master["Week"]
    index = master["idx_cleaned_weighted"].to_numpy(dtype=float)

    prov = pd.read_parquet(
        ROOT / "analysis" / "06_token_audit" / "matched_articles_with_text.parquet"
    )
    prov["date"] = pd.to_datetime(
        prov["date"].astype(str).str.slice(0, 10), format="%Y-%m-%d", errors="coerce"
    )
    prov = prov.dropna(subset=["date"]).drop_duplicates(subset=["outlet", "text"], keep="first")
    geo_set = set(CLEANED_GEO)
    prov["is_geo_specific"] = prov["geo_tokens"].apply(lambda c: has_any(c, geo_set))

    rng = np.random.default_rng(SEED)
    results = []
    for name, keep in VARIANTS.items():
        media = build_coverage(prov, keep, weeks)
        profile = lag_profile(media, index)
        positive = {lead: r for lead, r in profile.items() if lead > 0}
        peak_lead = max(positive, key=lambda lead: positive[lead])
        band = max_stat_band(media, index, rng)
        results.append(
            {
                "variant": name,
                "n_tokens": len(keep),
                "peak_lead_weeks": int(peak_lead),
                "peak_r": round(positive[peak_lead], 4),
                "max_stat_band_95": round(band, 4),
                "clears_band": bool(abs(positive[peak_lead]) > band),
                "r_at_26": round(profile[26], 4),
                "corr_with_primary_series": None,
            }
        )
        print(
            f"{name:44s} k={len(keep):2d} peak L={peak_lead:2d} r={positive[peak_lead]:.3f} "
            f"band=±{band:.3f} {'CLEARS' if abs(positive[peak_lead]) > band else 'does not clear'}"
        )

    baseline_media = build_coverage(prov, BASELINE, weeks)
    for row, (name, keep) in zip(results, VARIANTS.items()):
        row["corr_with_primary_series"] = round(
            float(np.corrcoef(baseline_media, build_coverage(prov, keep, weeks))[0, 1]), 4
        )

    payload = {
        "description": "Lead-lag sensitivity to the climate-token retention rule",
        "index": "idx_cleaned_weighted",
        "n_circular_shifts": N_SHIFT,
        "seed": SEED,
        "variants": results,
    }
    (OUT / "coverage_token_sensitivity.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {OUT / 'coverage_token_sensitivity.json'}")


if __name__ == "__main__":
    main()
