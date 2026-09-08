"""Build the SI tone-interaction horizon figure from the corrected specification.

The earlier version of this figure plotted the uncorrected estimates (no regime
indicator, fixed HAC bandwidth) and covered only four horizons, which
contradicted the corrected values reported in the main text. This rebuild reads
corrected_tone_interaction.json and covers all five reported horizons for both
index constructions.

Run from the repository root.
"""

from __future__ import annotations
# --- replication-package prelude (added by build_package.py) ------------------
# The original research scripts pointed at an absolute project root. In this
# package the repository locates itself, so it runs from any checkout location.
from pathlib import Path as _RepoPath
_REPO = str(_RepoPath(__file__).resolve().parents[3])
# ------------------------------------------------------------------------------

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = _RepoPath(_REPO)
DATA = ROOT / "analysis/09_stats/pooled_model/corrected_tone_interaction.json"
GRAPHS = ROOT / "manuscript/src/graphs-Round3"

HORIZONS = [12, 16, 20, 25, 26]
SERIES = [
    ("Cleaned index", "h{h}", "#1f3b73", -0.28),
    ("Original 34-token index", "idx_orig34_weighted|h{h}", "#CC79A7", 0.28),
]


def main() -> None:
    data = json.loads(DATA.read_text())

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    ax.axhline(0, color="black", lw=0.9, zorder=1)

    positions = np.arange(len(HORIZONS), dtype=float)
    for label, pattern, color, offset in SERIES:
        xs, points, lows, highs = [], [], [], []
        for i, h in enumerate(HORIZONS):
            entry = data[pattern.format(h=h)]
            lo, hi = entry["en_ix_bootCI"]
            xs.append(positions[i] + offset)
            points.append(entry["en_ix_b"])
            lows.append(entry["en_ix_b"] - lo)
            highs.append(hi - entry["en_ix_b"])
        ax.errorbar(xs, points, yerr=[lows, highs], fmt="o", color=color,
                    capsize=3.5, markersize=5, lw=1.4, label=label, zorder=3)

    ax.set_xticks(positions)
    ax.set_xticklabels([f"L = {h}" for h in HORIZONS])
    ax.set_xlabel("Lead L (weeks)")
    ax.set_ylabel(r"Energy $\times$ tone coefficient $\hat\theta_1$")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.text(0.995, 0.02,
            "Moving-block bootstrap 95% intervals (block length = max(L, 25))",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7.5, color="#666666")

    fig.tight_layout()
    fig.savefig(GRAPHS / "fig_energy_horizon.pdf")
    fig.savefig(GRAPHS / "fig_energy_horizon.png", dpi=300)
    plt.close(fig)
    print("wrote fig_energy_horizon.pdf (corrected spec, 5 horizons, both indices)")


if __name__ == "__main__":
    main()
