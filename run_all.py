#!/usr/bin/env python3
"""End-to-end orchestrator for the GeoengineeringNews replication package.

Runs every pipeline stage in dependency order, checking that each stage's
required inputs exist before launching it and failing with an actionable
message when they do not. See README.md for how to obtain each input file.

Usage:
  python run_all.py                 # run everything
  python run_all.py --list          # show stages and their input status
  python run_all.py --stages lag_profile,pooled
  python run_all.py --verify        # after a run: check key numbers vs the paper
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent

# (stage, scripts, required inputs, notes). "cwd=repo" scripts read paths
# relative to the repository root; all others are location-independent.
STAGES = [
    (
        "rebuild",
        ["analysis/08_rebuild/rebuild.py"],
        [
            "data/01_search_terms/geoengineering_weight.csv",
            "data/07_weekly_series/gt_weekly_36tokens_2018_2022.csv",
            "analysis/06_token_audit/matched_articles_with_text.parquet",
            "analysis/06_token_audit/audit_article_codes.csv",
        ],
        "builds weekly_series_master.csv; downloads sentence-transformer weights on first run",
    ),
    (
        "lag_profile",
        ["analysis/09_stats/lag_profile/run_lag_profile.py"],
        ["analysis/08_rebuild/weekly_series_master.csv"],
        "scan-corrected lead-lag profiles, null bands, influence diagnostics",
    ),
    (
        "lag_figures",
        [
            "analysis/09_stats/lag_profile/make_figures.py",
            "analysis/09_stats/lag_profile/build_main_lag_figure.py",
            "analysis/09_stats/lag_profile/build_main_influence_figure.py",
        ],
        ["analysis/09_stats/lag_profile/focal_profile_primary.csv"],
        "main-text Figures 2-3 and the SI two-panel comparisons",
    ),
    (
        "adl",
        ["analysis/09_stats/adl/run_adl.py"],
        [
            "analysis/08_rebuild/weekly_series_master.csv",
            "data/07_weekly_series/weekly_topic_salience_by_category.csv",
        ],
        "dynamic (ADL) models, multipliers, event terms",
    ),
    (
        "pooled",
        [
            "analysis/09_stats/pooled_model/run_corrected_pooled_model.py",
            "analysis/09_stats/pooled_model/run_corrected_tone_interaction.py",
            "analysis/09_stats/pooled_model/run_horizon_band.py",
            "analysis/09_stats/pooled_model/run_pooled_model.py",
            "analysis/09_stats/pooled_model/build_topic_coefficient_figure.py",
            "analysis/09_stats/pooled_model/build_tone_horizon_figure.py",
        ],
        [
            "analysis/08_rebuild/weekly_series_master.csv",
            "data/07_weekly_series/weekly_topic_salience_by_category.csv",
            "data/07_weekly_series/weekly_topic_sentiment_by_category.csv",
            "data/07_weekly_series/gt_weekly_36tokens_2018_2022.csv",
        ],
        "Table 4 and the RQ2/RQ3 inferential results; Figure 4; SI horizon figure",
    ),
    (
        "tone",
        [
            "analysis/10_volume_vs_tone/run.py",
            "analysis/10_volume_vs_tone/strengthen_fig5.py",
        ],
        [
            "analysis/08_rebuild/weekly_series_master.csv",
            "analysis/06_token_audit/matched_articles_with_text.parquet",
            "analysis/10_volume_vs_tone/climate_clean_tones.csv",
            "analysis/09_stats/lag_profile/focal_profile_primary.csv",
        ],
        "nested decomposition, tone-split permutations, Figure 5 (tone scores reused; "
        "re-scoring instead needs models/news_sentiment_bert_retrained_seed42)",
    ),
    (
        "cross_retrieval",
        ["analysis/12_cross_retrieval/cross_retrieval_stability.py"],
        ["data/07_weekly_series/gt_weekly_36tokens_2018_2022.csv"],
        "needs data/03_google_trends/gt_monthly_worldwide__*.csv as well",
    ),
    (
        "coverage_sensitivity",
        ["analysis/13_coverage_sensitivity/coverage_token_sensitivity.py"],
        [
            "analysis/08_rebuild/weekly_series_master.csv",
            "analysis/06_token_audit/matched_articles_with_text.parquet",
        ],
        "news-token retention sensitivity (SI S11)",
    ),
    (
        "lexical_overlap",
        ["analysis/14_lexical_overlap/energy_lexical_overlap.py"],
        [
            "data/07_weekly_series/gt_weekly_36tokens_2018_2022.csv",
            "data/01_search_terms/geoengineering_weight.csv",
        ],
        "energy result under indices stripped of energy-named terms",
    ),
    (
        "unambiguous",
        [
            "analysis/15_unambiguous_index/unambiguous_index.py",
            "analysis/15_unambiguous_index/index_construction_robustness.py",
            "analysis/15_unambiguous_index/index_verdict_table.py",
            "analysis/15_unambiguous_index/build_verdict_table_tex.py",
        ],
        [
            "data/07_weekly_series/gt_weekly_36tokens_2018_2022.csv",
            "analysis/08_rebuild/weekly_series_master.csv",
        ],
        "unambiguous-term index, seven-construction robustness, verdict table",
    ),
    (
        "tables",
        ["analysis/06_token_audit/build_primary_audit_table.py"],
        ["analysis/06_token_audit/audit_precision_table.csv"],
        "SI token-audit table",
    ),
    (
        "band_outputs",
        [
            "analysis/09_stats/pooled_model/build_horizon_band_outputs.py",
            "analysis/10_volume_vs_tone/build_claim_neutral_figure.py",
        ],
        [
            "analysis/09_stats/pooled_model/horizon_band_19_26.json",
            "analysis/14_lexical_overlap/energy_lexical_overlap.json",
            "analysis/15_unambiguous_index/index_construction_robustness_h19.json",
            "analysis/10_volume_vs_tone/dr_curve.npz",
        ],
        "Figure 3 (horizon band), Table 4, SI band tables, Figure 4 with the by-horizon ladder",
    ),
]

# scripts that read paths relative to the repository root
CWD_REPO = {
    "analysis/09_stats/pooled_model/run_corrected_pooled_model.py",
    "analysis/09_stats/pooled_model/run_corrected_tone_interaction.py",
    "analysis/09_stats/pooled_model/run_horizon_band.py",
    "analysis/09_stats/pooled_model/build_horizon_band_outputs.py",
    "analysis/15_unambiguous_index/index_construction_robustness.py",
}


def check_inputs(inputs: list[str]) -> list[str]:
    return [p for p in inputs if not (REPO / p).exists()]


def run_stage(name: str, scripts: list[str]) -> bool:
    for rel in scripts:
        script = REPO / rel
        cwd = REPO if rel in CWD_REPO else script.parent
        print(f"  -> {rel}")
        t0 = time.time()
        proc = subprocess.run([sys.executable, str(script)], cwd=cwd)
        dt = time.time() - t0
        if proc.returncode != 0:
            print(f"  FAILED ({dt:.0f}s): {rel}")
            return False
        print(f"     done ({dt:.0f}s)")
    return True


def verify() -> int:
    expected = json.loads((REPO / "expected_values.json").read_text())
    failures = 0

    def close(a, b, tol):
        return abs(float(a) - float(b)) <= tol

    checks: list[tuple[str, bool]] = []
    try:
        spec = json.loads(
            (REPO / "analysis/09_stats/pooled_model/corrected_final_spec.json").read_text()
        )
        h25 = spec["idx_cleaned_weighted|h25"]
        e = expected["pooled_h25"]
        checks.append(("energy coefficient", close(h25["energy_b"], e["energy_b"], 0.02)))
        checks.append(("energy HAC se", close(h25["energy_se"], e["energy_se"], 0.02)))
        checks.append(("adjusted R2", close(h25["adjR2"], e["adjR2"], 0.01)))
        boot = spec["bootstrap_h25_cleaned"]
        checks.append(("bootstrap CI low", close(boot["energy_CI"][0], e["ci"][0], 0.15)))
        checks.append(("bootstrap CI high", close(boot["energy_CI"][1], e["ci"][1], 0.15)))
        checks.append(("rank-1 share", close(boot["rank1_share"], e["rank1_share"], 0.05)))
    except FileNotFoundError:
        checks.append(("pooled outputs present", False))

    try:
        import pandas as pd

        prof = pd.read_csv(REPO / "analysis/09_stats/lag_profile/focal_profile_primary.csv")
        e = expected["lag_profile"]
        peak = prof.loc[prof.L == 26, "r"].iloc[0]
        band = prof["band_global"].iloc[0]
        checks.append(("peak r(26)", close(peak, e["peak_r26"], 0.005)))
        checks.append(("global null band", close(band, e["band_global"], 0.01)))
    except FileNotFoundError:
        checks.append(("lag-profile outputs present", False))

    try:
        band = json.loads((REPO / "analysis/09_stats/pooled_model/horizon_band_19_26.json").read_text())["summary"]
        e = expected["horizon_band"]
        for key in ("energy_positive_at", "energy_first_at", "energy_sig05_hac_at", "tone_within_benchmark_at"):
            checks.append((f"band {key}", band[key] == e[key]))
        checks.append(("band energy mean", close(band["energy_b_mean"], e["energy_b_mean"], 0.05)))
    except FileNotFoundError:
        checks.append(("horizon-band outputs present", False))

    try:
        verd = json.loads(
            (REPO / "analysis/15_unambiguous_index/index_verdict_table.json").read_text()
        )
        checks.append(("constructions clearing band", verd["n_clearing"] == expected["verdicts"]["n_clearing"]))
        rob = json.loads(
            (REPO / "analysis/15_unambiguous_index/index_construction_robustness.json").read_text()
        )
        checks.append(("energy first-ranked count", rob["energy_first_ranked_in"] == expected["verdicts"]["energy_first"]))
    except FileNotFoundError:
        checks.append(("index-robustness outputs present", False))

    print("\nVERIFICATION AGAINST THE PAPER'S REPORTED VALUES")
    for name, ok in checks:
        print(f"  {'OK  ' if ok else 'FAIL'} {name}")
        failures += (not ok)
    print(f"\n{len(checks) - failures}/{len(checks)} checks passed")
    return 1 if failures else 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stages", help="comma-separated subset of stages to run")
    ap.add_argument("--list", action="store_true", help="list stages and input status")
    ap.add_argument("--verify", action="store_true", help="check key outputs against the paper")
    args = ap.parse_args()

    if args.verify:
        sys.exit(verify())

    selected = None
    if args.stages:
        selected = {s.strip() for s in args.stages.split(",")}
        unknown = selected - {name for name, *_ in STAGES}
        if unknown:
            sys.exit(f"unknown stage(s): {sorted(unknown)}")

    if args.list:
        for name, scripts, inputs, note in STAGES:
            missing = check_inputs(inputs)
            status = "ready" if not missing else f"missing {len(missing)} input(s)"
            print(f"{name:22s} [{status}]  {note}")
            for m in missing:
                print(f"    missing: {m}")
        return

    for name, scripts, inputs, note in STAGES:
        if selected and name not in selected:
            continue
        print(f"\n=== stage: {name} — {note}")
        missing = check_inputs(inputs)
        if missing:
            print("  SKIPPED, missing inputs (see README.md 'Data' for how to obtain):")
            for m in missing:
                print(f"    {m}")
            continue
        if not run_stage(name, scripts):
            sys.exit(f"stage {name} failed")
    print("\nAll requested stages completed. Run `python run_all.py --verify` to check key numbers.")


if __name__ == "__main__":
    main()
