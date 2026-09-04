"""Is the energy topic's first rank a property of one index construction?

The composite index is weighted by indexed search-result counts, a rule that
concentrates weight on generic terms. If the leading rank of energy-coverage
volume were an artifact of that weighting, it should disappear under other
constructions of the outcome. This script re-estimates the corrected pooled
specification at the featured horizon against seven constructions spanning the
plausible range: the two weighted composites, two equal-weighted indices built
only from unambiguous terms, the strict umbrella-only index, an equal-weighted
index over all cleaned tokens, and the single term `geoengineering`.

Run from the repository root. Writes index_construction_robustness.json.
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
import os as _os
HORIZONS = [int(_os.environ['IDX_H'])] if _os.environ.get('IDX_H') else list(range(19, 27))
HORIZON = HORIZONS[0]

UMBRELLA_ONLY = ["geoengineering", "geoengineer", "climate engineering", "climate engineer"]


def run(HORIZON: int) -> None:
    gt = pd.read_csv(ROOT / "data/07_weekly_series/gt_weekly_36tokens_2018_2022.csv")
    gt["Week"] = pd.to_datetime(gt["Week"])
    master = pd.read_csv(ROOT / "analysis/08_rebuild/weekly_series_master.csv",
                         parse_dates=["Week"])

    unambiguous = [t for t in ui.CLEANED if t in gt.columns and t not in ui.AMBIGUOUS]
    cleaned = [t for t in ui.CLEANED if t in gt.columns]
    umbrella = [t for t in UMBRELLA_ONLY if t in gt.columns]

    constructions = {
        "cleaned weighted (primary)": master["idx_cleaned_weighted"].to_numpy(float),
        "original 34-token weighted": master["idx_orig34_weighted"].to_numpy(float),
        "unambiguous-19, equal weight on standardized series":
            np.mean([ui.zscore(gt[t]) for t in unambiguous], axis=0),
        "unambiguous-19, equal weight on raw series":
            gt[unambiguous].mean(1).to_numpy(float),
        "all cleaned tokens, equal weight": gt[cleaned].mean(1).to_numpy(float),
        "strict umbrella-only": gt[umbrella].mean(1).to_numpy(float),
        "geoengineering alone": gt["geoengineering"].to_numpy(float),
    }

    rows = []
    for name, index in constructions.items():
        result = ui.pooled_model(index, gt["Week"], h=HORIZON)
        result["construction"] = name
        rows.append(result)
        print(f"{name:52s} b={result['energy_b']:>7} p={result['energy_p']:<8} "
              f"rank1={result['rank1_topic']:<10} energy_rank={result['energy_rank']}")

    energy_first = sum(1 for r in rows if r["energy_rank"] == 1)
    significant = sum(1 for r in rows if r["energy_p"] < 0.05)
    payload = {
        "description": "Energy salience rank at the featured horizon across index constructions",
        "horizon": HORIZON,
        "specification": "identical to analysis/09_stats/pooled_model/run_corrected_pooled_model.py",
        "n_constructions": len(rows),
        "energy_first_ranked_in": energy_first,
        "energy_significant_in": significant,
        "results": rows,
    }
    (OUT / f"index_construction_robustness_h{HORIZON}.json").write_text(json.dumps(payload, indent=2) + "\n")
    if HORIZON == 25:
        (OUT / "index_construction_robustness.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nenergy first-ranked in {energy_first}/{len(rows)} constructions; "
          f"p < .05 in {significant}/{len(rows)}")


if __name__ == "__main__":
    for _h in HORIZONS:
        run(_h)
