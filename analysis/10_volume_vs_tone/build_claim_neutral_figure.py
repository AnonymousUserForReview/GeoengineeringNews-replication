#!/usr/bin/env python3
"""Build the tone-split figure for the volume-versus-tone analysis.

The numerical inputs are the archived diagnostic outputs. This script changes
presentation only: it does not recalculate any result.
"""

from __future__ import annotations
# --- replication-package prelude (added by build_package.py) ------------------
# The original research scripts pointed at an absolute project root. In this
# package the repository locates itself, so it runs from any checkout location.
from pathlib import Path as _RepoPath
_REPO = str(_RepoPath(__file__).resolve().parents[2])
# ------------------------------------------------------------------------------

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
GRAPH_ROOT = ROOT / "manuscript/src/graphs-Round3"
CURVE_PATH = HERE / "dr_curve.npz"
NOISE_PATH = HERE / "noise_benchmark.npz"
OOS_PATH = HERE / "nested_oos_idx_cleaned_weighted_h25.csv"
OUTPUT_PDF = GRAPH_ROOT / "fig_tone_split.pdf"
OUTPUT_PNG = GRAPH_ROOT / "fig_tone_split.png"
PROVENANCE_PATH = HERE / "claim_neutral_figure_provenance.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    curves = np.load(CURVE_PATH)
    leads = np.arange(53)

    rp = np.asarray(curves["rp"], dtype=float)
    rn = np.asarray(curves["rn"], dtype=float)
    delta = np.asarray(curves["dr"], dtype=float)
    lower = np.asarray(curves["lo"], dtype=float)
    upper = np.asarray(curves["hi"], dtype=float)
    if not all(values.shape == (53,) for values in (rp, rn, delta, lower, upper)):
        raise ValueError("tone-profile inputs must each have 53 leads")
    if not np.allclose(delta, rp - rn, atol=1e-12, rtol=1e-12):
        raise ValueError("stored difference curve does not equal rp-rn")

    clearing = np.asarray(curves["clearing_leads"], dtype=int)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    figure = plt.figure(figsize=(11.6, 5.2), constrained_layout=True)
    grid = figure.add_gridspec(2, 2, width_ratios=(1.55, 1.0), height_ratios=(2, 1))
    profile_axis = figure.add_subplot(grid[0, 0])
    difference_axis = figure.add_subplot(grid[1, 0], sharex=profile_axis)
    rmse_axis = figure.add_subplot(grid[:, 1])

    pos_color = "#009E73"
    neg_color = "#CC79A7"
    for lead in clearing:
        profile_axis.axvspan(lead - 0.5, lead + 0.5, color="#4C78A8", alpha=0.13, zorder=0)
    profile_axis.plot(
        leads, rp, color=pos_color, linewidth=2.0, label="Positive-tone volume"
    )
    profile_axis.plot(
        leads, rn, color=neg_color, linewidth=2.0, label="Negative-tone volume"
    )
    profile_axis.axhline(0, color="black", linewidth=0.8)
    band = float(curves["band_primary"])
    profile_axis.axhline(band, color="#777777", linestyle="--", linewidth=0.9)
    profile_axis.text(
        51.5,
        0.36,
        f"±{band:.3f}: largest correlation\nchance alone produces",
        ha="right",
        va="bottom",
        color="#666666",
        fontsize=8,
    )
    profile_axis.text(
        22.5,
        0.055,
        "leads at which coverage and search interest line up",
        ha="center",
        va="bottom",
        color="#4C78A8",
        fontsize=7.5,
    )
    # print the values the text quotes at the peak lead and at the largest gap
    peak = 26
    profile_axis.plot([peak], [rp[peak]], "o", color=pos_color, ms=4, zorder=4)
    profile_axis.plot([peak], [rn[peak]], "o", color=neg_color, ms=4, zorder=4)
    profile_axis.text(
        27.5, 0.47,
        f"L = 26: negative {rn[peak]:.2f}, positive {rp[peak]:.2f}",
        ha="left", va="bottom", fontsize=7.5, color="#333333",
    )
    profile_axis.set_ylim(-0.05, 0.55)
    profile_axis.set_ylabel(r"Correlation with search index, $r(L)$")
    profile_axis.set_title(
        "A. Tone-split profiles and uncertainty", loc="left", fontweight="bold"
    )
    profile_axis.legend(frameon=False, loc="upper left")
    profile_axis.tick_params(labelbottom=False)

    difference_axis.fill_between(
        leads, lower, upper, color="#BDBDBD", alpha=0.42, linewidth=0
    )
    difference_axis.plot(leads, delta, color="#333333", linewidth=1.7)
    difference_axis.axhline(0, color="black", linewidth=0.8)
    for lead in clearing:
        difference_axis.axvspan(lead - 0.5, lead + 0.5, color="#4C78A8", alpha=0.13, zorder=0)
    difference_axis.plot([peak], [delta[peak]], "o", color="#333333", ms=4, zorder=4)
    difference_axis.text(
        52, 0.145,
        f"L = 26: $\\Delta r$ = {delta[peak]:.2f}, range [{lower[peak]:.2f}, {upper[peak]:+.2f}]",
        ha="right", va="center", fontsize=7.5, color="#333333",
    )
    gap = int(np.argmax(np.abs(delta)))
    difference_axis.plot([gap], [delta[gap]], "o", color="#333333", ms=4, zorder=4)
    difference_axis.text(
        gap + 1.5, delta[gap] - 0.005,
        f"largest gap {abs(delta[gap]):.2f} at L = {gap}",
        ha="left", va="center", fontsize=7.5, color="#333333",
    )
    difference_axis.set_ylim(-0.22, 0.17)
    difference_axis.set_ylabel(r"$\Delta r(L)$")
    difference_axis.set_xlabel("Lead L (weeks; positive = media leads search)")
    difference_axis.text(
        0.01,
        0.95,
        "shaded: range of the gap on resampled blocks of weeks (95%)",
        transform=difference_axis.transAxes,
        ha="left",
        va="top",
        color="#555555",
        fontsize=8,
    )

    band = json.loads((ROOT / "analysis/09_stats/pooled_model/horizon_band_19_26.json").read_text())
    rows = band["by_horizon"]
    H = [r["h"] for r in rows]
    # change in prediction error when the twelve tone series are added to the model that
    # already has the twelve volume series: real tone against tone shifted in time (no information)
    real = [r["ladder"]["tone"]["oos_rmse"] - r["ladder"]["volume"]["oos_rmse"] for r in rows]
    fake_m = [r["benchmark"]["mean"] - r["ladder"]["volume"]["oos_rmse"] for r in rows]
    fake_lo = [r["benchmark"]["ci"][0] - r["ladder"]["volume"]["oos_rmse"] for r in rows]
    fake_hi = [r["benchmark"]["ci"][1] - r["ladder"]["volume"]["oos_rmse"] for r in rows]
    for i, h in enumerate(H):
        rmse_axis.plot([h, h], [fake_lo[i], fake_hi[i]], color="#9e9e9e", lw=7, solid_capstyle="butt",
                       zorder=1, label="tone shifted in time, carrying no information (95% of 200 draws)" if i == 0 else None)
        rmse_axis.plot([h - 0.22, h + 0.22], [fake_m[i], fake_m[i]], color="#616161", lw=1.4, zorder=2)
    rmse_axis.plot(H, real, "o", color="#E69F00", ms=8, mec="#8a6100", mew=1.0, zorder=3,
                   label="the real tone series")
    rmse_axis.axhline(0, color="black", lw=1.0, zorder=2)
    rmse_axis.text(18.55, 0.06, "worse", ha="left", va="bottom", fontsize=7.5, color="#555555")
    rmse_axis.text(18.55, -0.06, "better", ha="left", va="top", fontsize=7.5, color="#555555")
    inside = sum(lo <= r <= hi for r, lo, hi in zip(real, fake_lo, fake_hi))
    rmse_axis.text(0.5, 0.97, f"the real tone series falls inside the no-information range\nat {inside} of the 8 leads",
                   transform=rmse_axis.transAxes, ha="center", va="top", fontsize=8, color="#333333")
    rmse_axis.set_xticks(H)
    rmse_axis.set_xlim(18.4, 26.6)
    rmse_axis.set_xlabel("Lead L (weeks)")
    rmse_axis.set_ylabel("Change in prediction error when tone is added\n(index points; above zero = worse predictions)")
    rmse_axis.legend(frameon=False, fontsize=7.5, loc="lower center", bbox_to_anchor=(0.5, -0.30))
    rmse_axis.set_title(
        "B. Does tone help predict unseen weeks?", loc="left", fontweight="bold"
    )

    figure.suptitle(
        "Coverage volume and tone: does tone carry the association?",
        fontsize=11,
        fontweight="bold",
    )
    figure.savefig(OUTPUT_PDF, bbox_inches="tight")
    figure.savefig(OUTPUT_PNG, dpi=240, bbox_inches="tight")
    plt.close(figure)

    provenance = {
        "schema_version": "1.0.0",
        "status": "presentation_only",
        "claim_neutral_rebuild": True,
        "inputs": {
            path.relative_to(ROOT).as_posix(): _sha256(path)
            for path in (CURVE_PATH, ROOT / "analysis/09_stats/pooled_model/horizon_band_19_26.json")
        },
        "outputs": {
            path.relative_to(ROOT).as_posix(): {
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in (OUTPUT_PDF, OUTPUT_PNG)
        },
    }
    PROVENANCE_PATH.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
