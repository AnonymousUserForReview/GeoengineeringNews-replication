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
| `rebuild` | weekly media-coverage variants and all index constructions (`analysis/08_rebuild/weekly_series_master.csv`) | Section 2 measures |
| `lag_profile` | scan-corrected lead–lag profiles, max-statistic null bands, influence diagnostics, leave-one-token-out family, 16-combination variant table | RQ1; SI S4, S11 |
| `lag_figures` | | Figures 2–3; SI two-panel figures |
| `adl` | ADL models, cumulative/long-run multipliers, event terms, Granger diagnostics | RQ1 dynamics; SI S4 |
| `pooled` | corrected pooled topic model at h ∈ {12,16,20,25,26} and across the 19–26-week band (`run_horizon_band.py`: HAC and OLS errors, bootstrap, tone interactions, prediction ladder and benchmark at every horizon), per-topic FDR table | Table 4, Figure 3; SI S8, S10–S11 |
| `tone` | nested block decomposition, tone-split permutation tests, shifted-tone benchmark | RQ3, Figure 5; SI S10 |
| `cross_retrieval` | cross-retrieval stability of every search-term series | SI S3 |
| `coverage_sensitivity` | news-token retention-rule sensitivity | SI S11 |
| `lexical_overlap` | energy result under indices stripped of energy-named terms | RQ2 robustness |
| `unambiguous` | unambiguous-term index, seven-construction ranking robustness, per-construction verdict table | RQ2/robustness; SI Table |
| `tables` | SI token-audit LaTeX table | SI S2 |
| `band_outputs` | Figure 3 (energy across the horizon band; all topics), Table 4, the SI band tables, and Figure 4 with the by-horizon prediction ladder | Figures 3–4, Table 4; SI S10 |

Every script sets explicit random seeds; `--verify` compares the regenerated
headline numbers (energy coefficient and bootstrap CI, the horizon-band counts, peak lead–lag correlation
and null band, verdict counts) against `expected_values.json` with tolerances
that absorb platform-level floating-point differences.

## Data

Place the following files at the stated paths. Nothing else is required.

| File | Used by | Provenance |
|---|---|---|
| `data/01_search_terms/geoengineering_weight.csv` | rebuild, pooled, lexical_overlap, unambiguous | Google search-result counts per token, recorded at collection time (2 columns: token, count). Deposited on Zenodo at acceptance. |
| `data/07_weekly_series/gt_weekly_36tokens_2018_2022.csv` | most stages | Weekly worldwide Google Trends series (0–100) for the Table 1/2 tokens, 2018–2022, one retrieval per term. Trends data cannot be redistributed in bulk by us; the exact term list, window, and geography are in the paper (Table 1), and our retrieval will be archived on Zenodo at acceptance. |
| `data/03_google_trends/gt_monthly_worldwide__*.csv` | cross_retrieval | Independent monthly Google Trends exports per term (UI export format), retrieved 2025-09-18. Same access note as above. |
| `data/07_weekly_series/weekly_topic_salience_by_category.csv` and `weekly_topic_sentiment_by_category.csv` | adl, pooled, tone, lexical_overlap, unambiguous | Weekly topic-salience and topic-tone series from the BERTopic consolidation and the fine-tuned tone model (paper Sections 2.2.5–2.2.6). Deposited on Zenodo at acceptance. |
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

## License

MIT (see `LICENSE`). The license covers the code in this repository; the input
data remain subject to their providers' terms as described under *Data*.
