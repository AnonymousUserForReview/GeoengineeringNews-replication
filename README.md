# GeoengineeringNews — replication package

End-to-end code for **"Climate news coverage and online search interest in
geoengineering-related technologies: an exploratory computational analysis."**
A single orchestrator regenerates every statistic, table, and figure reported in
the paper and its Supplementary Information from the input data.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python run_all.py --list      # shows each stage and which inputs it still needs
python run_all.py             # runs everything that has its inputs (in order)
python run_all.py --verify    # checks key outputs against the values in the paper
```

Stages with missing inputs are skipped with an explicit list of what is missing;
nothing fails silently. Individual stages: `python run_all.py --stages pooled,tone`.

## Pipeline

| Stage | What it produces | Paper artefacts |
|---|---|---|
| `rebuild` | weekly media-coverage variants and all index constructions (`analysis/08_rebuild/weekly_series_master.csv`) | Methods measures |
| `lag_profile` | correlation between coverage and search interest at every lead, the level chance alone produces (max-statistic null), influence diagnostics at the peak lead, leave-one-token-out family, 16-combination variant table | RQ1; SI S4, S11 |
| `lag_figures` | Figure 1 (correlation at each lead, leads above the chance level circled); Figure 2 (change in the correlation when each quarter or year is left out, at every lead 19 to 26, `run_influence_by_horizon.py`); SI two-panel comparisons | Figures 1 and 2; SI figures |
| `adl` | dynamic (ADL) models, cumulative and long-run multipliers, event terms, Granger diagnostics | RQ1 closing paragraph; SI S4 |
| `pooled` | pooled topic model at L in {12,16,20,25,26} and at every lead 19 to 26 (`run_horizon_band.py`: HAC and OLS errors, block bootstrap, the across-lead mean with jointly resampled weeks, tone interactions, prediction ladder and shifted-tone benchmark at every lead); per-topic FDR table; SI lead figure | RQ2, RQ3; Table 2; SI S8, S10, S11 |
| `ml` | exploratory predictive models (Linear, XGBoost) against train-mean, persistence and seasonal-naive baselines, with SHAP attributions | SI S9 Table (Linear and XGBoost columns). The Random Forest and CatBoost values quoted in the S9 text come from a separate leakage-check script that is not part of this package. No main-text claim rests on any of these models. |
| `tone` | tone-split correlation curves, permutation tests, archived tone scores reused | Figure 4A; SI S10 |
| `cross_retrieval` | stability of every search-term series across separate downloads | SI S3 |
| `coverage_sensitivity` | news-token retention-rule sensitivity | SI S11 |
| `lexical_overlap` | energy result with energy-related search terms removed from the index, at every lead 19 to 26 | RQ2 footnote; SI lexical table |
| `unambiguous` | unambiguous-term index, seven-construction ranking at every lead 19 to 26, per-construction verdict table | RQ1 footnote; RQ2; SI table |
| `tables` | SI token-audit LaTeX table | SI S2 |
| `band_outputs` | Figure 3 (energy at each lead; all twelve topics over the leads), Table 2 (`generated/pooled_band_table.tex`), the SI band tables, and Figure 4 with panel B (change in prediction error when tone is added, real tone against tone shifted in time) | Figures 3 and 4, Table 2; SI S10 tables |

Every script sets explicit random seeds; `--verify` compares the regenerated
headline numbers (energy coefficient and bootstrap CI at h = 25, the counts
across the leads 19 to 26, peak correlation and chance level, verdict
counts) against `expected_values.json` with tolerances that absorb
platform-level floating-point differences.

## Data

Place the following files at the stated paths. Nothing else is required.

| File | Used by | Provenance |
|---|---|---|
| `data/01_search_terms/geoengineering_weight.csv` | rebuild, pooled, lexical_overlap, unambiguous | Google search-result counts per token, recorded at collection time (2 columns: token, count). Deposited on Zenodo at acceptance. |
| `data/07_weekly_series/gt_weekly_36tokens_2018_2022.csv` | most stages | Weekly worldwide Google Trends series (0–100) for the Table 1/2 tokens, 2018–2022, one retrieval per term. Trends data cannot be redistributed in bulk by us; the exact term list, window, and geography are in the paper (Table 1), and our retrieval will be archived on Zenodo at acceptance. |
| `data/03_google_trends/gt_monthly_worldwide__*.csv` | cross_retrieval | Independent monthly Google Trends exports per term (UI export format), retrieved 2025-09-18. Same access note as above. |
| `data/07_weekly_series/weekly_topic_salience_by_category.csv` and `weekly_topic_sentiment_by_category.csv` | adl, pooled, tone, lexical_overlap, unambiguous | Weekly topic-volume and topic-tone series from the BERTopic consolidation and the fine-tuned tone model (paper Sections 2.2.5–2.2.6). Deposited on Zenodo at acceptance. |
| `analysis/06_token_audit/matched_articles_with_text.parquet` | rebuild, coverage_sensitivity | The token-matched news corpus with article text. **Cannot be redistributed (publisher copyright).** Article identifiers, dates, and the retrieval interfaces are described in the paper (Section 2.1.2); the corpus is reconstructable from the publishers' interfaces with the token lists in Tables 1–2. |
| `analysis/06_token_audit/audit_article_codes.csv` | rebuild | Article-level model codes from the relevance audit (ids + binary codes; no text). Deposited at acceptance. |
| `analysis/06_token_audit/audit_precision_table.csv` | tables | Per-token audit precision table (aggregate). Deposited at acceptance. |
| `analysis/10_volume_vs_tone/climate_clean_tones.csv` | tone | Per-article tone scores for the climate-clean corpus (ids + scores; no text). Deposited at acceptance. Re-scoring from raw text instead requires the fine-tuned checkpoint `models/news_sentiment_bert_retrained_seed42/` (deposited at acceptance). |

## Runtime notes

- `rebuild` downloads two sentence-transformer models on first run and embeds
  ~19k articles; on CPU allow ~30–60 minutes (minutes on Apple Silicon/CUDA).
- The permutation/bootstrap stages (`lag_profile`, `coverage_sensitivity`,
  `unambiguous`) run 2,000–4,000 circular shifts each; allow a few minutes per
  stage.
- Total cold run: roughly 1–2 hours on a laptop.

## Relationship to the paper

Scripts are the research originals with one mechanical change (a
repository-self-locating root replaces an absolute path; see the prelude at the
top of each script). Figures write to `manuscript/src/graphs-Round3/`, LaTeX
tables to `manuscript/src/generated/`, statistics to their stage directories.
The four main-text figures are `fig_lag_profile_main.pdf` (Figure 1),
`fig_influence_main.pdf` (Figure 2), `fig_topic_coefficients_main.pdf`
(Figure 3) and `fig_tone_split.pdf` (Figure 4).

## License

MIT (see `LICENSE`). The license covers the code in this repository; the input
data remain subject to their providers' terms as described under *Data*.
