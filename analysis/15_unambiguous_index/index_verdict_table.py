"""Peak lead--lag estimate and scan-corrected verdict for each index construction.

The paper's measurement argument is that a scan-corrected verdict is a joint
property of an estimate and the series' persistence, not of the estimate alone.
This script makes that claim checkable: for each construction of the outcome it
reports the peak positive-lead correlation with the same climate-coverage
series, that construction's own max-statistic null band, its lag-1
autocorrelation, and whether the peak clears.

Run from the repository root. Writes index_verdict_table.json and .csv.
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

import unambiguous_index as ui

ROOT = _RepoPath(_REPO)
OUT = ROOT / "analysis" / "15_unambiguous_index"
N_SHIFT = 2000
SEED = 42
LEADS = range(-26, 53)


def profile(media: np.ndarray, index: np.ndarray) -> dict[int, float]:
    out = {}
    for lead in LEADS:
        if lead >= 0:
            a, b = media[: len(media) - lead], index[lead:]
        else:
            a, b = media[-lead:], index[: len(index) + lead]
        out[lead] = float(np.corrcoef(a, b)[0, 1])
    return out


def band(media: np.ndarray, index: np.ndarray) -> float:
    rng = np.random.default_rng(SEED)
    n = len(index)
    maxima = np.empty(N_SHIFT)
    for i in range(N_SHIFT):
        shifted = np.roll(index, int(rng.integers(53, n - 53)))
        maxima[i] = max(abs(r) for r in profile(media, shifted).values())
    return float(np.percentile(maxima, 95))


def main() -> None:
    master = pd.read_csv(ROOT / "analysis/08_rebuild/weekly_series_master.csv",
                         parse_dates=["Week"])
    gt = pd.read_csv(ROOT / "data/07_weekly_series/gt_weekly_36tokens_2018_2022.csv")
    gt["Week"] = pd.to_datetime(gt["Week"])
    media = master["m_climate_clean_pooled_z"].to_numpy(dtype=float)

    unambiguous = [t for t in ui.CLEANED if t in gt.columns and t not in ui.AMBIGUOUS]

    constructions = {
        "cleaned, result-count weighted (primary)": master["idx_cleaned_weighted"].to_numpy(float),
        "cleaned, first principal component": master["idx_cleaned_pca"].to_numpy(float),
        "cleaned, equal weight": master["idx_cleaned_equal"].to_numpy(float),
        "unambiguous-19, equal weight on standardized series":
            np.mean([ui.zscore(gt[t]) for t in unambiguous], axis=0),
        "strict umbrella-only, equal weight": master["idx_strict_umbrella_equal"].to_numpy(float),
        "geoengineering alone": gt["geoengineering"].to_numpy(float),
    }

    rows = []
    for name, index in constructions.items():
        prof = profile(media, index)
        positive = {k: v for k, v in prof.items() if k > 0}
        peak = max(positive, key=lambda k: positive[k])
        b = band(media, index)
        rows.append({
            "construction": name,
            "peak_lead_weeks": int(peak),
            "peak_r": round(positive[peak], 3),
            "null_band_95": round(b, 3),
            "lag1_autocorrelation": round(float(np.corrcoef(index[:-1], index[1:])[0, 1]), 2),
            "clears_band": bool(abs(positive[peak]) > b),
        })
        print(f"{name:54s} L={peak:2d} r={positive[peak]:.3f} band=±{b:.3f} "
              f"lag1={rows[-1]['lag1_autocorrelation']:.2f} "
              f"{'CLEARS' if rows[-1]['clears_band'] else 'does not clear'}")

    table = pd.DataFrame(rows)
    table.to_csv(OUT / "index_verdict_table.csv", index=False)
    clearing = [r for r in rows if r["clears_band"]]
    payload = {
        "description": "Peak estimate and scan-corrected verdict by index construction",
        "media_series": "m_climate_clean_pooled_z",
        "n_circular_shifts": N_SHIFT,
        "seed": SEED,
        "n_constructions": len(rows),
        "n_clearing": len(clearing),
        "peak_r_range_excluding_single_term": [
            min(r["peak_r"] for r in rows if r["construction"] != "geoengineering alone"),
            max(r["peak_r"] for r in rows if r["construction"] != "geoengineering alone"),
        ],
        "null_band_range": [min(r["null_band_95"] for r in rows),
                            max(r["null_band_95"] for r in rows)],
        "rows": rows,
    }
    (OUT / "index_verdict_table.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\n{len(clearing)}/{len(rows)} constructions clear their own band")


if __name__ == "__main__":
    main()
