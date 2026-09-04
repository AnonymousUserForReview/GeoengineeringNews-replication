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
    figure = plt.figure(figsize=(10.2, 4.9), constrained_layout=True)
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
        0.355,
        f"95% scan band  ±{band:.3f}",
        ha="right",
        va="bottom",
        color="#666666",
        fontsize=8,
    )
    profile_axis.text(
        22.5,
        0.055,
        "leads clearing the band",
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
        f"at L = 26:  negative-tone r = {rn[peak]:.2f},  positive-tone r = {rp[peak]:.2f}",
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
        27.5, 0.145,
        f"at L = 26:  $\\Delta r$ = {delta[peak]:.2f}, band [{lower[peak]:.2f}, {upper[peak]:+.2f}]",
        ha="left", va="center", fontsize=7.5, color="#333333",
    )
    gap = int(np.argmax(np.abs(delta)))
    difference_axis.plot([gap], [delta[gap]], "o", color="#333333", ms=4, zorder=4)
    difference_axis.text(
        gap + 1.5, delta[gap] - 0.01,
        f"largest gap {abs(delta[gap]):.2f} at L = {gap}",
        ha="left", va="top", fontsize=7.5, color="#333333",
    )
    difference_axis.set_ylim(-0.22, 0.17)
    difference_axis.set_ylabel(r"$\Delta r(L)$")
    difference_axis.set_xlabel("Lead L (weeks; positive = media leads search)")
    difference_axis.text(
        0.01,
        0.95,
        "Joint block-bootstrap 95% interval (shaded)",
        transform=difference_axis.transAxes,
        ha="left",
        va="top",
        color="#555555",
        fontsize=8,
    )

    band = json.loads((ROOT / "analysis/09_stats/pooled_model/horizon_band_19_26.json").read_text())
    rows = band["by_horizon"]; H = [r["h"] for r in rows]
    lo = [r["benchmark"]["ci"][0] for r in rows]; hi = [r["benchmark"]["ci"][1] for r in rows]
    rmse_axis.fill_between(H, lo, hi, color="#BDBDBD", alpha=0.45, lw=0, label="shifted-tone benchmark, 95% interval")
    for key, col, lab, mk in (("baseline", "#999999", "baseline", "s"), ("volume", "#4C72B0", "+ topic volume", "o"), ("tone", "#E69F00", "+ topic tone", "D")):
        vals = [r["ladder"][key]["oos_rmse"] for r in rows]
        rmse_axis.plot(H, vals, marker=mk, color=col, lw=1.4, ms=5, label=lab)
    inside = sum(l <= r["ladder"]["tone"]["oos_rmse"] <= u for r, l, u in zip(rows, lo, hi))
    better = sum(r["ladder"]["volume"]["oos_rmse"] < r["ladder"]["baseline"]["oos_rmse"] for r in rows)
    rmse_axis.text(0.02, 0.97, f"tone model inside the benchmark interval at {inside} of 8 horizons\n"
                   f"topic volume lowers the error at {better} of 8 horizons",
                   transform=rmse_axis.transAxes, ha="left", va="top", fontsize=7.5, color="#333333")
    rmse_axis.set_xticks(H)
    rmse_axis.set_xlabel("Horizon h (weeks)")
    rmse_axis.set_ylabel("Out-of-sample RMSE (lower is better)")
    rmse_axis.legend(frameon=False, fontsize=7.5, loc="lower right")
    rmse_axis.set_title(
        "B. Prediction error by horizon", loc="left", fontweight="bold"
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
