"""Rebuild cleaned weekly series for the revision.

Outputs analysis/08_rebuild/weekly_series_master.csv with:
- M_t variants: original 46-token counts; climate-clean (climate tokens minus
  'natural disaster', excluding geoengineering-specific articles; per-outlet and pooled);
  classifier-filtered climate coverage.
- GeoIndex variants: original-34 weighted (paper) & equal; cleaned weighted/equal/PCA;
  strict umbrella-only; CDR and SRM sub-indices (+ pooling gate statistic); LOTO family.
Also writes token-set metadata, classifier validation, and index correlations.
"""
# --- replication-package prelude (added by build_package.py) ------------------
# The original research scripts pointed at an absolute project root. In this
# package the repository locates itself, so it runs from any checkout location.
from pathlib import Path as _RepoPath
_REPO = str(_RepoPath(__file__).resolve().parents[2])
# ------------------------------------------------------------------------------
import json, hashlib
import numpy as np
import pandas as pd

ROOT = _REPO
OUT = f"{ROOT}/analysis/08_rebuild"

# ---------------- token sets (audit-driven; see 06_token_audit) ----------------
DROP_GEO = ["greenhouse", "ecosystem", "carbon dioxide", "solar radiation",
            "enhanced weather", "carbon management", "carborn management", "gas removal"]
UMBRELLA = ["geoengineering", "geoengineer", "climate engineering", "climate engineer"]
SRM = ["aerosol injection", "stratospheric aerosol injection", "albedo modification",
       "high albedo", "high-albedo", "marine cloud brightening", "ocean mirror",
       "space shade", "space sunshade"]
CDR = ["air capture", "carbon capture", "carbon storage", "carbon sequestration",
       "biochar", "enhanced weathering", "ocean fertilization", "ocean fertilisation",
       "ocean iron fertilization", "ocean iron fertilisation", "methane removal",
       "afforestation", "reforestation", "bioenergy", "bio-energy"]
CLEANED = UMBRELLA + SRM + CDR
CLIMATE_KEEP = ["biodiversity loss", "deforestation", "sea level rise", "plastic pollution",
                "ocean acidification", "air pollution", "global warming", "overfishing",
                "climate change"]  # 'natural disaster' dropped (audit: climate precision 0.17)

# ---------------- weekly GT indices ----------------
gt = pd.read_csv(f"{ROOT}/data/07_weekly_series/gt_weekly_36tokens_2018_2022.csv")
gt["Week"] = pd.to_datetime(gt["Week"])
tok_cols = [c for c in gt.columns if c not in ("Week",) and not c.lower().startswith("geoengineering index")]
tok_cols = [c for c in tok_cols if gt[c].dtype != object]
print("GT token columns:", len(tok_cols))

w_raw = pd.read_csv(f"{ROOT}/data/01_search_terms/geoengineering_weight.csv",
                    header=None, names=["token", "w"], encoding="utf-8-sig")
w_raw["w"] = w_raw["w"].str.replace(",", "").astype(float)
W = dict(zip(w_raw["token"].str.strip(), w_raw["w"]))

def weighted_index(cols):
    cols = [c for c in cols if c in gt.columns and c in W]
    ws = np.array([W[c] for c in cols])
    return (gt[cols].values * ws).sum(1) / ws.sum(), cols

def equal_index(cols):
    cols = [c for c in cols if c in gt.columns]
    return gt[cols].mean(1).values, cols

def pca_index(cols):
    cols = [c for c in cols if c in gt.columns]
    X = gt[cols].values.astype(float)
    Z = (X - X.mean(0)) / X.std(0).clip(1e-9)
    u, s, vt = np.linalg.svd(Z - Z.mean(0), full_matrices=False)
    pc1 = u[:, 0] * s[0]
    if np.corrcoef(pc1, Z.mean(1))[0, 1] < 0:
        pc1 = -pc1
    return pc1, cols

series = pd.DataFrame({"Week": gt["Week"]})
series["idx_orig34_weighted"], used_all = weighted_index(tok_cols)
series["idx_orig34_equal"], _ = equal_index(tok_cols)
series["idx_cleaned_weighted"], used_cl = weighted_index(CLEANED)
series["idx_cleaned_equal"], used_cl_e = equal_index(CLEANED)
series["idx_cleaned_pca"], _ = pca_index(CLEANED)
series["idx_strict_umbrella_equal"], used_str = equal_index(UMBRELLA)
series["idx_srm_equal"], used_srm = equal_index(SRM)
series["idx_cdr_equal"], used_cdr = equal_index(CDR)
print(f"tokens used: all={len(used_all)} cleaned={len(used_cl_e)} strict={len(used_str)} srm={len(used_srm)} cdr={len(used_cdr)}")

# leave-one-token-out family on cleaned equal-weight index (saved separately)
loto = pd.DataFrame({"Week": gt["Week"]})
for c in used_cl_e:
    loto[f"loto_drop__{c}"] = gt[[x for x in used_cl_e if x != c]].mean(1).values
loto.to_csv(f"{OUT}/loto_cleaned_equal.csv", index=False)

# CDR/SRM pooling gate: correlation of 4-week-mean first differences
g4 = series.set_index("Week")[["idx_srm_equal", "idx_cdr_equal"]].resample("4W").mean().diff().dropna()
gate_r = float(np.corrcoef(g4["idx_srm_equal"], g4["idx_cdr_equal"])[0, 1])
dw = series[["idx_srm_equal", "idx_cdr_equal"]].diff().dropna()
gate_r_weekly_diff = float(np.corrcoef(dw["idx_srm_equal"], dw["idx_cdr_equal"])[0, 1])
srm_zero_share = float((gt[[c for c in SRM if c in gt.columns]].sum(1) == 0).mean())

# ---------------- corpus / M_t variants ----------------
prov = pd.read_parquet(f"{ROOT}/analysis/06_token_audit/matched_articles_with_text.parquet")
prov["date"] = pd.to_datetime(prov["date"].astype(str).str.slice(0, 10), format="%Y-%m-%d", errors="coerce")
prov = prov.dropna(subset=["date"])
prov = prov.drop_duplicates(subset=["outlet", "text"], keep="first")  # dedup (was 1,762 involved)
print("after dedup:", prov.groupby("outlet").size().to_dict())

def has_any(cell, toks):
    s = set(cell.split("|")) if cell else set()
    return bool(s & set(toks))

prov["is_geo_specific"] = prov["geo_tokens"].apply(lambda c: has_any(c, CLEANED))
prov["is_climate_kept"] = prov["climate_tokens"].apply(lambda c: has_any(c, CLIMATE_KEEP))
prov["is_srm"] = prov["geo_tokens"].apply(lambda c: has_any(c, SRM + UMBRELLA))
prov["is_cdr"] = prov["geo_tokens"].apply(lambda c: has_any(c, CDR))
print("geo-specific articles (cleaned tokens):", int(prov.is_geo_specific.sum()),
      "| SRM(+umbrella):", int(prov.is_srm.sum()), "| CDR:", int(prov.is_cdr.sum()))

# classifier-filtered climate relevance (MiniLM embeddings of provenance texts + audit labels)
from sentence_transformers import SentenceTransformer
emb_model = SentenceTransformer("all-MiniLM-L6-v2", device=("mps" if __import__("torch").backends.mps.is_available() else "cpu"))
prov_emb = emb_model.encode(prov["text"].fillna("").tolist(), batch_size=128, show_progress_bar=False)
np.save(f"{OUT}/prov_embeddings.npy", prov_emb)
id2pos = {a: i for i, a in enumerate(prov["article_id"])}

codes = pd.read_csv(f"{ROOT}/analysis/06_token_audit/audit_article_codes.csv")
codes["pos"] = codes["article_id"].map(id2pos)
lab = codes.dropna(subset=["pos"]).drop_duplicates(subset=["article_id"])
print("audit labels mapped:", len(lab), "of", len(codes))

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
X = prov_emb[lab["pos"].astype(int).values]
y = lab["climate"].astype(int).values
clf = LogisticRegression(max_iter=2000, C=1.0)
auc = cross_val_score(clf, X, y, cv=5, scoring="roc_auc")
acc = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
clf.fit(X, y)
prov["p_climate"] = clf.predict_proba(prov_emb)[:, 1]
print(f"climate-relevance classifier: 5-fold AUC {auc.mean():.3f}±{auc.std():.3f}, acc {acc.mean():.3f}; base rate {y.mean():.3f}")


def weekly(mask, name):
    d = prov[mask].copy()
    d["wkstart"] = d["date"].dt.to_period("W-SAT").dt.start_time  # Sunday start = GT anchor
    wk = d.groupby(["wkstart", "outlet"]).size().unstack(fill_value=0)
    wk = wk.reindex(series["Week"].values, fill_value=0)
    out = pd.DataFrame(index=wk.index)
    for o in ["bbc", "nytimes"]:
        col = wk[o] if o in wk.columns else pd.Series(0, index=wk.index)
        out[f"{name}_{o}"] = col.values
        out[f"{name}_{o}_z"] = (col - col.mean()) / col.std() if col.std() > 0 else 0.0
    out[f"{name}_pooled_z"] = out[[f"{name}_bbc_z", f"{name}_nytimes_z"]].mean(1)
    return out

m_all = weekly(prov.index == prov.index, "m_all46")
m_cli = weekly(prov["is_climate_kept"] & ~prov["is_geo_specific"], "m_climate_clean")
m_filt = weekly((prov["p_climate"] >= 0.5) & prov["is_climate_kept"] & ~prov["is_geo_specific"], "m_climate_filtered")
m_geo = weekly(prov["is_geo_specific"], "m_geo_specific")

master = pd.concat([series.set_index("Week"), m_all, m_cli, m_filt, m_geo], axis=1).reset_index().rename(columns={"index": "Week"})
assert master[[c for c in master.columns if c.startswith("idx_")]].notna().all().all(), "index columns contain NaN"
master.to_csv(f"{OUT}/weekly_series_master.csv", index=False)

idx_cols = [c for c in master.columns if c.startswith("idx_")]
corr = master[idx_cols].corr().round(3)
corr.to_csv(f"{OUT}/index_variant_correlations.csv")
d4 = master.set_index("Week")[idx_cols].resample("4W").mean().diff().dropna()
corr_d4 = d4.corr().round(3)
corr_d4.to_csv(f"{OUT}/index_variant_correlations_4wdiff.csv")

meta = {
    "dropped_geo_tokens": DROP_GEO, "umbrella": UMBRELLA, "srm": [c for c in SRM if c in gt.columns],
    "cdr": [c for c in CDR if c in gt.columns], "climate_kept": CLIMATE_KEEP,
    "dropped_climate_tokens": ["natural disaster"],
    "corpus_after_dedup": prov.groupby("outlet").size().to_dict(),
    "n_geo_specific_articles": int(prov.is_geo_specific.sum()),
    "n_srm_articles": int(prov.is_srm.sum()), "n_cdr_articles": int(prov.is_cdr.sum()),
    "n_climate_clean_articles": int((prov["is_climate_kept"] & ~prov["is_geo_specific"]).sum()),
    "cdr_srm_gate_r_4wdiff": gate_r, "cdr_srm_gate_r_weekly_diff": gate_r_weekly_diff,
    "srm_index_zero_week_share": srm_zero_share,
    "classifier_auc_5fold": [round(x, 4) for x in auc], "classifier_acc_5fold": [round(x, 4) for x in acc],
    "classifier_base_rate": float(y.mean()), "classifier_n_labels": int(len(lab)),
}
json.dump(meta, open(f"{OUT}/rebuild_metadata.json", "w"), indent=2)
print(json.dumps(meta, indent=1)[:1500])
print("key index correlations (levels):")
print(corr.loc["idx_orig34_weighted", ["idx_cleaned_weighted", "idx_cleaned_equal", "idx_cleaned_pca", "idx_strict_umbrella_equal"]].to_string())
print("CDR-SRM gate r (4w diff):", round(gate_r, 3), "| weekly diff:", round(gate_r_weekly_diff, 3), "| SRM zero-week share:", round(srm_zero_share, 3))
print("DONE")
