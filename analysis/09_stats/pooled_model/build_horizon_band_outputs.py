"""Figure 3, Table 4 and the SI horizon tables from the h = 19..26 band.

Reads horizon_band_19_26.json (run_horizon_band.py), the lead--lag profile,
the lexical-overlap results and the per-horizon index-construction results,
and writes:
  manuscript/src/graphs-Round3/fig_topic_coefficients_main.pdf   (Figure 3, two panels)
  manuscript/src/generated/pooled_band_table.tex                 (Table 4 body)
  manuscript/src/generated/si_topic_band_table.tex               (SI: 12 topics x 8 horizons)
  manuscript/src/generated/si_ladder_band_table.tex              (SI: prediction ladder by horizon)
  analysis/09_stats/pooled_model/band_numbers.json               (numbers quoted in the text)
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
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = _RepoPath(_REPO)
PM = ROOT / "analysis/09_stats/pooled_model"
GRAPHS = ROOT / "manuscript/src/graphs-Round3"
GEN = ROOT / "manuscript/src/generated"
CATS = ["art", "disaster", "economy", "education", "energy", "medical",
        "nature", "politics", "pollution", "religion", "society", "technology"]


def main() -> None:
    band = json.loads((PM / "horizon_band_19_26.json").read_text())
    rows = band["by_horizon"]
    H = [r["h"] for r in rows]
    prof = pd.read_csv(ROOT / "analysis/09_stats/lag_profile/focal_profile_primary.csv")
    rL = dict(zip(prof.L, prof.r))
    band_global = float(prof.band_global.iloc[0])
    clears = {h: abs(rL[h]) > band_global for h in H}

    # ---------- Figure 3: two panels ----------
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.0, 4.6), gridspec_kw={"width_ratios": [1.15, 1.0]})
    for h in H:
        if clears[h]:
            ax.axvspan(h - 0.5, h + 0.5, color="#4C78A8", alpha=0.10, zorder=0, lw=0)
    ax.axhline(0, color="black", lw=0.8, zorder=1)
    for r in rows:
        lo, hi = r["energy_boot_ci"]
        ax.plot([r["h"], r["h"]], [lo, hi], color="#d95f02", lw=2.0, zorder=2)
        ax.plot([r["h"]], [r["energy_b"]], "o", color="#d95f02", ms=6, zorder=3)
        ax.text(r["h"], hi + 0.12, f"{r['energy_b']:.2f}", ha="center", va="bottom", fontsize=8, color="#1a1a1a")
    ax.axhline(band["summary"]["energy_b_mean"], color="#d95f02", lw=0.9, ls=(0, (4, 3)), zorder=1)
    ax.text(23.5, band["summary"]["energy_b_mean"] + 0.06, f"mean {band['summary']['energy_b_mean']:.2f}",
            ha="center", va="bottom", fontsize=8, color="#b34700")
    ax.text(19.5, -1.55, "shaded: horizons circled in Figure 1 (r above 0.346)",
            ha="left", va="center", fontsize=7.5, color="#4C78A8")
    ax.set_xticks(H)
    ax.set_xlim(18.4, 26.6)
    ax.set_ylim(-1.8, 4.2)
    ax.set_xlabel("Horizon h (weeks between coverage and search interest)")
    ax.set_ylabel("Index points per one-SD rise in energy coverage")
    ax.set_title("A. Energy coefficient at each horizon, 19 to 26 weeks", loc="left", fontweight="bold")

    tb = band["topic_band"]
    order = sorted(CATS, key=lambda c: -tb[c]["mean"])
    ys = np.arange(len(order))[::-1]
    bx.axvline(0, color="black", lw=0.8, zorder=1)
    for y, c in zip(ys, order):
        col = "#d95f02" if c == "energy" else "#1f3b73"
        bx.plot([tb[c]["min"], tb[c]["max"]], [y, y], color=col, lw=2.0, alpha=0.55, zorder=2)
        bx.plot([tb[c]["mean"]], [y], "o", color=col, ms=6, zorder=3)
        ba = band.get("band_average", {})
        if c == "energy" and ba:
            lo_, hi_ = ba["energy_mean_ci"]
            bx.plot([lo_, hi_], [y + 0.28, y + 0.28], color=col, lw=1.2, zorder=2)
            bx.plot([lo_, lo_], [y + 0.2, y + 0.36], color=col, lw=1.2); bx.plot([hi_, hi_], [y + 0.2, y + 0.36], color=col, lw=1.2)
        label = f"{tb[c]['mean']:.2f}" + (f", first at {tb[c]['first_at']} of 8" if tb[c]["first_at"] else "")

        bx.text(tb[c]["max"] + 0.12, y, label, va="center", ha="left", fontsize=7.5,
                color="#b34700" if c == "energy" else "#333333")
    ba = band.get("band_average", {})
    if ba:
        fig.text(0.995, 0.012, f"Energy, mean over the eight horizons: {ba['energy_mean_over_horizons']:.2f}; range on resampled weeks "
                               f"[{ba['energy_mean_ci'][0]:.2f}, {ba['energy_mean_ci'][1]:.2f}] (thin whisker); largest mean of the twelve in "
                               f"{100*ba['share_energy_largest_mean']:.0f}% of resamples",
                 ha="right", va="bottom", fontsize=7.5, color="#b34700")
    bx.set_yticks(ys)
    bx.set_yticklabels([c.capitalize() for c in order])
    bx.set_xlim(-1.6, 3.6)
    bx.set_xlabel("Mean over the eight horizons (dot) and range across them (line)")
    bx.set_title("B. All twelve topics over the same horizons", loc="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(GRAPHS / "fig_topic_coefficients_main.pdf")
    fig.savefig(GRAPHS / "fig_topic_coefficients_main.png", dpi=300)
    plt.close(fig)

    # ---------- Table 4: the band, one row per horizon ----------
    def p(x):
        return "$<$.0001" if x < 1e-4 else f"{x:.3f}".lstrip("0")
    lines = [r"\begin{tabular}{lccccc}", r"\toprule",
             r"Horizon $h$ & $r(h)$ & Energy coefficient & s.e. & Range on resampled weeks & Rank of energy \\",
             r"\midrule"]
    for r in rows:
        star = r"$^{\ast}$" if clears[r["h"]] else ""
        lines.append(f"{r['h']}{star} & {rL[r['h']]:.3f} & {r['energy_b']:.2f} & {r['energy_se_hac']:.2f} & "
                     f"[{r['energy_boot_ci'][0]:.2f}, {r['energy_boot_ci'][1]:.2f}] & {r['energy_rank']} \\\\")
    ba = band.get("band_average", {})
    if ba:
        lines.append(r"\midrule")
        lines.append(f"Mean over the eight & & {ba['energy_mean_over_horizons']:.2f} & & [{ba['energy_mean_ci'][0]:.2f}, {ba['energy_mean_ci'][1]:.2f}] & 1 \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (GEN / "pooled_band_table.tex").write_text("\n".join(lines) + "\n")

    # ---------- SI: all topics x horizons ----------
    co = band["coefficients"]
    lines = [r"\begin{tabular}{l" + "c" * len(H) + "}", r"\toprule",
             "Topic & " + " & ".join(f"$h={h}$" for h in H) + r" \\", r"\midrule"]
    for c in order:
        cells = []
        for h in H:
            e = co[str(h)][c]
            s = f"{e['b']:.2f}"
            if e["p_hac"] < 0.01: s += r"\sym{***}"
            elif e["p_hac"] < 0.05: s += r"\sym{**}"
            elif e["p_hac"] < 0.10: s += r"\sym{*}"
            cells.append(s)
        lines.append(f"{c} & " + " & ".join(cells) + r" \\")
    lines.append(r"\midrule")
    for key, name in (("en_ix", r"Energy $\times$ tone"), ("oth_ix", r"Pooled other $\times$ tone"),
                      ("covid", "Pandemic indicator"), ("ukr", "Post-Feb-2022 indicator")):
        cells = []
        for h in H:
            e = co[str(h)][key]
            s = f"{e['b']:.2f}"
            if e["p_hac"] < 0.01: s += r"\sym{***}"
            elif e["p_hac"] < 0.05: s += r"\sym{**}"
            elif e["p_hac"] < 0.10: s += r"\sym{*}"
            cells.append(s)
        lines.append(f"{name} & " + " & ".join(cells) + r" \\")
    lines.append("Energy s.e., Newey--West & " + " & ".join(f"{r['energy_se_hac']:.2f}" for r in rows) + r" \\")
    lines.append("Energy s.e., plain OLS & " + " & ".join(f"{r['energy_se_ols']:.2f}" for r in rows) + r" \\")
    lines.append("Adj.\\ $R^2$ & " + " & ".join(f"{r['adjR2']:.3f}" for r in rows) + r" \\")
    lines.append("$N$ (weeks) & " + " & ".join(str(r["N"]) for r in rows) + r" \\")
    lines += [r"\bottomrule",
              r"\multicolumn{" + str(len(H) + 1) + r"}{l}{\footnotesize \sym{*} $p<0.10$, \sym{**} $p<0.05$, \sym{***} $p<0.01$ (HAC).}\\",
              r"\end{tabular}"]
    (GEN / "si_topic_band_table.tex").write_text("\n".join(lines) + "\n")

    # ---------- SI: ladder by horizon ----------
    lines = [r"\begin{tabular}{lcccccccc}", r"\toprule",
             "& " + " & ".join(f"$h={h}$" for h in H) + r" \\", r"\midrule"]
    for key, name in (("baseline", "Baseline adj.\\ $R^2$"), ("volume", "+ topic volume"), ("tone", "+ topic tone"), ("interactions", "+ interactions")):
        lines.append(f"{name} & " + " & ".join(f"{r['ladder'][key]['adjR2']:.3f}" for r in rows) + r" \\")
    lines.append(r"\midrule")
    for key, name in (("baseline", "Baseline OOS RMSE"), ("volume", "+ topic volume"), ("tone", "+ topic tone"), ("interactions", "+ interactions")):
        lines.append(f"{name} & " + " & ".join(f"{r['ladder'][key]['oos_rmse']:.2f}" for r in rows) + r" \\")
    lines.append("Shifted-tone benchmark & " + " & ".join(f"{r['benchmark']['mean']:.2f}" for r in rows) + r" \\")
    lines.append("Benchmark 95\\% interval & " + " & ".join(f"[{r['benchmark']['ci'][0]:.1f}, {r['benchmark']['ci'][1]:.1f}]" for r in rows) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (GEN / "si_ladder_band_table.tex").write_text("\n".join(lines) + "\n")

    # ---------- numbers for the text ----------
    lex = json.loads((ROOT / "analysis/14_lexical_overlap/energy_lexical_overlap.json").read_text())
    lex_summary = {}
    for name, v in lex["variants"].items():
        hs = [x for x in v["horizons"] if 19 <= x["h"] <= 26]
        lex_summary[name] = {"weight_share": v.get("weight_share_of_primary"),
                             "first_at": sum(x["energy_rank"] == 1 for x in hs), "positive_at": sum(x["energy_b"] > 0 for x in hs),
                             "sig05_at": sum(x["energy_p"] < 0.05 for x in hs),
                             "b_range": [min(x["energy_b"] for x in hs), max(x["energy_b"] for x in hs)]}
    idx = {}
    for h in H:
        r = json.loads((ROOT / f"analysis/15_unambiguous_index/index_construction_robustness_h{h}.json").read_text())
        cons = r.get("constructions", r.get("results", []))
        idx[h] = {"first": r.get("energy_first_ranked_in"), "n": len(cons), "sig05": sum(c.get("energy_p", 1) < 0.05 for c in cons)}
    # ---------- SI: vocabulary-overlap test by horizon ----------
    lines = [r"\begin{tabular}{l" + "c" * len(H) + "}", r"\toprule",
             "Index variant & " + " & ".join(f"$h={h}$" for h in H) + r" \\", r"\midrule"]
    for name, v in lex["variants"].items():
        cells = []
        for h in H:
            x = next(x for x in v["horizons"] if x["h"] == h)
            s_ = f"{x['energy_b']:.2f}"
            if x["energy_p"] < 0.01: s_ += r"\sym{***}"
            elif x["energy_p"] < 0.05: s_ += r"\sym{**}"
            elif x["energy_p"] < 0.10: s_ += r"\sym{*}"
            s_ += f" ({x['energy_rank']})"
            cells.append(s_)
        share = f" ({100 * v['weight_share_of_primary']:.0f}\\% of weight)" if v.get("weight_share_of_primary") and v["weight_share_of_primary"] < 1 else ""
        lines.append(f"{name}{share} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule",
              r"\multicolumn{" + str(len(H) + 1) + r"}{l}{\footnotesize Energy coefficient (HAC stars: \sym{*} $p<0.10$, \sym{**} $p<0.05$, \sym{***} $p<0.01$) and, in parentheses, energy's rank among the twelve topics.}\\",
              r"\end{tabular}"]
    (GEN / "si_lexical_band_table.tex").write_text("\n".join(lines) + "\n")

    numbers = {"summary": band["summary"], "topic_band": band["topic_band"],
               "by_horizon": [{k: v for k, v in r.items() if k != "ladder"} | {"ladder": r["ladder"]} for r in rows],
               "rL": {h: round(rL[h], 3) for h in H}, "clears": clears, "band_global": round(band_global, 3),
               "lexical_overlap": lex_summary, "index_constructions": idx, "band_average": band.get("band_average", {})}
    (PM / "band_numbers.json").write_text(json.dumps(numbers, indent=1) + "\n")
    print("Figure 3, Table 4, SI tables and band_numbers.json written")
    print("lexical:", json.dumps(lex_summary))
    print("index constructions:", json.dumps(idx))


if __name__ == "__main__":
    main()
