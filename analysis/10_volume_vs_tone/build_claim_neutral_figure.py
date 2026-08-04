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
    noise = np.load(NOISE_PATH)
    oos = pd.read_csv(OOS_PATH)
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
    rmse = oos["OOS_RMSE"].to_numpy(dtype=float)
    if rmse.shape != (4,) or not np.isfinite(rmse).all():
        raise ValueError("expected four finite OOS RMSE rows")
    shifted = np.asarray(noise["noise"], dtype=float)
    if shifted.shape != (200,) or not np.isfinite(shifted).all():
        raise ValueError("expected 200 finite shifted-tone benchmark draws")

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
    difference_axis.set_ylabel(r"$\Delta r(L)$")
    difference_axis.set_xlabel("Lead L (weeks; positive = media leads search)")
    difference_axis.text(
        1,
        0.93,
        "Joint block-bootstrap 95% interval",
        transform=difference_axis.transAxes,
        ha="right",
        va="top",
        color="#555555",
        fontsize=8,
    )

    bar_colors = ["#999999", "#4C72B0", "#E69F00", "#CC79A7"]
    labels = ["Baseline", "+ topic\nvolume", "+ topic\ntone", "+ tone\ninteractions"]
    positions = np.arange(4)
    bars = rmse_axis.bar(positions, rmse, color=bar_colors, width=0.62)
    for bar, value in zip(bars, rmse, strict=True):
        rmse_axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.07,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    shifted_mean = float(shifted.mean())
    shifted_low, shifted_high = np.quantile(shifted, [0.025, 0.975])
    benchmark_x = 2.35
    rmse_axis.errorbar(
        benchmark_x,
        shifted_mean,
        yerr=[[shifted_mean - shifted_low], [shifted_high - shifted_mean]],
        fmt="D",
        color="#333333",
        capsize=4,
        markersize=5,
        zorder=5,
    )
    rmse_axis.annotate(
        "shifted-tone\nbenchmark",
        xy=(benchmark_x, shifted_mean),
        xytext=(1.55, 14.0),
        arrowprops={"arrowstyle": "-", "color": "#555555"},
        ha="center",
        va="bottom",
        fontsize=8,
    )
    rmse_axis.set_xticks(positions, labels)
    rmse_axis.set_ylabel("Out-of-sample RMSE (lower is better)")
    rmse_axis.set_ylim(10.0, max(16.2, float(rmse.max()) + 0.6))
    rmse_axis.set_title(
        "B. Tone block versus shifted benchmark", loc="left", fontweight="bold"
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
            for path in (CURVE_PATH, NOISE_PATH, OOS_PATH)
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
