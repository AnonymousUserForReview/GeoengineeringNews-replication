"""
Reduced pooled topic-moderation model + horizon robustness (replaces paper's Table 5).
Outputs into <repo>/analysis/09_stats/pooled_model/
"""
# --- replication-package prelude (added by build_package.py) ------------------
# The original research scripts pointed at an absolute project root. In this
# package the repository locates itself, so it runs from any checkout location.
from pathlib import Path as _RepoPath
_REPO = str(_RepoPath(__file__).resolve().parents[3])
# ------------------------------------------------------------------------------
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import matplotlib as mpl

RNG = np.random.default_rng(20260711)
OUT = (_REPO + "/analysis/09_stats/pooled_model/")
TOPICS = ["art", "disaster", "economy", "education", "energy", "medical",
          "nature", "politics", "pollution", "religion", "society", "technology"]
HORIZONS = [12, 16, 20, 25]
OUTCOMES = ["idx_orig34_weighted", "idx_cleaned_weighted"]
NW_BW = 19
BLOCK = 13
NBOOT = 1000

mpl.rcParams.update({
    "figure.dpi": 110, "font.size": 10, "axes.spines.top": False,
    "axes.spines.right": False, "axes.titlesize": 11, "axes.labelsize": 10,
})

# ---------------------------------------------------------------- load & merge
master = pd.read_csv((_REPO + "/analysis/08_rebuild/weekly_series_master.csv"),
                     parse_dates=["Week"])
sal = pd.read_csv((_REPO + "/data/07_weekly_series/weekly_topic_salience_by_category.csv"))
sen = pd.read_csv((_REPO + "/data/07_weekly_series/weekly_topic_sentiment_by_category.csv"))
for d in (sal, sen):
    d.rename(columns={"0": "Week"}, inplace=True)
    d["Week"] = pd.to_datetime(d["Week"])
sal = sal[["Week"] + TOPICS].rename(columns={t: f"sal_{t}" for t in TOPICS})
sen = sen[["Week"] + TOPICS].rename(columns={t: f"sen_{t}" for t in TOPICS})

df = master.merge(sal, on="Week", how="inner").merge(sen, on="Week", how="inner")
df = df.sort_values("Week").reset_index(drop=True)
N_MERGED = len(df)

# ------------------------------------------- empty-week rule in sentiment pivot
zero_tab = []
for t in TOPICS:
    z_sal = (df[f"sal_{t}"] == 0).mean()
    z_sen = (df[f"sen_{t}"] == 0).mean()
    both = ((df[f"sal_{t}"] == 0) & (df[f"sen_{t}"] == 0)).mean()
    zero_tab.append({"topic": t, "share_zero_salience": round(z_sal, 4),
                     "share_zero_sentiment": round(z_sen, 4),
                     "share_zero_both": round(both, 4)})
zero_tab = pd.DataFrame(zero_tab)
zero_tab.to_csv(OUT + "table_zero_week_shares.csv", index=False)

# ---------------------------------------------------------------- standardize
def zscore(x):
    return (x - x.mean()) / x.std(ddof=0)

for t in TOPICS:
    df[f"z_sal_{t}"] = zscore(df[f"sal_{t}"])
    df[f"z_sen_{t}"] = zscore(df[f"sen_{t}"])
    df[f"ix_{t}"] = df[f"z_sal_{t}"] * df[f"z_sen_{t}"]

# standardized interaction terms
df["energy_ix"] = zscore(df["ix_energy"])
others = [t for t in TOPICS if t != "energy"]
df["pooled_other_ix"] = zscore(df[[f"ix_{t}" for t in others]].mean(axis=1))

# controls
df["D_covid"] = ((df["Week"] >= "2020-03-08") & (df["Week"] <= "2021-12-26")).astype(float)
wk = np.arange(len(df), dtype=float)
P = 365.25 / 7.0  # weeks per year
df["sin1"] = np.sin(2 * np.pi * wk / P); df["cos1"] = np.cos(2 * np.pi * wk / P)
df["sin2"] = np.sin(4 * np.pi * wk / P); df["cos2"] = np.cos(4 * np.pi * wk / P)

SAL_COLS = [f"z_sal_{t}" for t in TOPICS]
LEVEL_TERMS = ["G_t", "G_tm1"] + SAL_COLS  # for VIF (plus interactions noted)

# ------------------------------------------------------------ model machinery
def build_xy(outcome, h):
    d = df.copy()
    d["G_t"] = d[outcome]
    d["G_tm1"] = d[outcome].shift(1)
    d["y"] = d[outcome].shift(-h)
    cols = (["G_t", "G_tm1"] + SAL_COLS + ["energy_ix", "pooled_other_ix",
            "D_covid", "sin1", "cos1", "sin2", "cos2"])
    d = d.dropna(subset=cols + ["y"]).reset_index(drop=True)
    X = sm.add_constant(d[cols])
    return d["y"], X, d

def fit_hac(y, X):
    return sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": NW_BW})

def mbb_indices(n, block, rng):
    nblocks = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=nblocks)
    idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
    return idx

def mbb_boot(y, X, coef_names, nboot=NBOOT, block=BLOCK, rng=RNG,
             contrast=("energy_ix", "pooled_other_ix"), track_rank=False):
    n = len(y)
    Xv, yv = X.values, y.values
    cols = list(X.columns)
    pos = {c: cols.index(c) for c in coef_names}
    out = {c: [] for c in coef_names}
    out["contrast"] = []
    ranks = []
    p1, p2 = cols.index(contrast[0]), cols.index(contrast[1])
    sal_pos = [cols.index(c) for c in SAL_COLS]
    for _ in range(nboot):
        idx = mbb_indices(n, block, rng)
        try:
            b = np.linalg.lstsq(Xv[idx], yv[idx], rcond=None)[0]
        except np.linalg.LinAlgError:
            continue
        for c in coef_names:
            out[c].append(b[pos[c]])
        out["contrast"].append(b[p1] - b[p2])
        if track_rank:
            mains = np.abs(b[sal_pos])
            ranks.append(int(np.argsort(-mains).tolist().index(SAL_COLS.index("z_sal_energy"))) + 1)
    res = {k: np.array(v) for k, v in out.items()}
    if track_rank:
        res["energy_rank"] = np.array(ranks)
    return res

def pct_ci(a, lo=2.5, hi=97.5):
    return np.percentile(a, [lo, hi])

# --------------------------------------------------------- 1. pooled estimation
rows = []
fits_h25 = {}
boot_h25 = {}
horizon_fig_data = {o: [] for o in OUTCOMES}

for outcome in OUTCOMES:
    for h in HORIZONS:
        y, X, d = build_xy(outcome, h)
        fit = fit_hac(y, X)
        boot = mbb_boot(y, X, ["energy_ix", "pooled_other_ix"],
                        track_rank=(h == 25))
        c1, c2 = fit.params["energy_ix"], fit.params["pooled_other_ix"]
        # HAC Wald test of c1 - c2 = 0
        wald = fit.t_test("energy_ix - pooled_other_ix = 0")
        ci1 = pct_ci(boot["energy_ix"]); ci2 = pct_ci(boot["pooled_other_ix"])
        cic = pct_ci(boot["contrast"])
        rows.append({
            "outcome": outcome, "h": h, "nobs": int(fit.nobs),
            "n_params": int(X.shape[1]), "adj_R2": round(fit.rsquared_adj, 4),
            "c1_energy_ix": round(c1, 4), "c1_HAC_se": round(fit.bse["energy_ix"], 4),
            "c1_HAC_p": round(fit.pvalues["energy_ix"], 4),
            "c1_boot_lo": round(ci1[0], 4), "c1_boot_hi": round(ci1[1], 4),
            "c2_pooled_ix": round(c2, 4), "c2_HAC_se": round(fit.bse["pooled_other_ix"], 4),
            "c2_HAC_p": round(fit.pvalues["pooled_other_ix"], 4),
            "c2_boot_lo": round(ci2[0], 4), "c2_boot_hi": round(ci2[1], 4),
            "contrast_c1_minus_c2": round(c1 - c2, 4),
            "contrast_HAC_p": round(float(wald.pvalue), 4),
            "contrast_boot_lo": round(cic[0], 4), "contrast_boot_hi": round(cic[1], 4),
        })
        horizon_fig_data[outcome].append((h, c1, ci1[0], ci1[1]))
        if h == 25:
            fits_h25[outcome] = (fit, X, y)
            boot_h25[outcome] = boot

pooled_tab = pd.DataFrame(rows)
pooled_tab.to_csv(OUT + "table_pooled_model.csv", index=False)

# full coefficient table at h=25 for both outcomes
coef_rows = []
for outcome in OUTCOMES:
    fit, X, y = fits_h25[outcome]
    for name in X.columns:
        coef_rows.append({"outcome": outcome, "term": name,
                          "coef": round(fit.params[name], 4),
                          "HAC_se": round(fit.bse[name], 4),
                          "HAC_p": round(fit.pvalues[name], 4)})
pd.DataFrame(coef_rows).to_csv(OUT + "table_pooled_coefs_h25.csv", index=False)

# --------------------------------------------------------------------- 2. VIF
_, Xv, _ = build_xy("idx_orig34_weighted", 25)
vif_cols = ["G_t", "G_tm1"] + SAL_COLS + ["energy_ix", "pooled_other_ix"]
Xvif = Xv[["const"] + vif_cols]
vif_rows = []
for i, c in enumerate(Xvif.columns):
    if c == "const":
        continue
    vif_rows.append({"term": c, "VIF": round(variance_inflation_factor(Xvif.values, i), 3)})
vif_tab = pd.DataFrame(vif_rows)
vif_tab.to_csv(OUT + "table_vif.csv", index=False)

# ------------------------------------- 3. energy dominance: rank + Wald summary
dom_rows = []
for outcome in OUTCOMES:
    fit, X, y = fits_h25[outcome]
    mains = fit.params[SAL_COLS].abs().sort_values(ascending=False)
    rank_energy = int(list(mains.index).index("z_sal_energy")) + 1
    br = boot_h25[outcome].get("energy_rank", np.array([]))
    dom_rows.append({
        "outcome": outcome, "h": 25,
        "b_energy": round(fit.params["z_sal_energy"], 4),
        "b_energy_HAC_p": round(fit.pvalues["z_sal_energy"], 4),
        "abs_rank_of_b_energy": rank_energy,
        "largest_main_effect": mains.index[0].replace("z_sal_", ""),
        "boot_share_rank1": round(float((br == 1).mean()), 3) if br.size else np.nan,
        "boot_share_rank_top3": round(float((br <= 3).mean()), 3) if br.size else np.nan,
        "boot_median_rank": float(np.median(br)) if br.size else np.nan,
    })
pd.DataFrame(dom_rows).to_csv(OUT + "table_energy_dominance.csv", index=False)

# ------------------------------------------- 4. SI per-topic models h=25 + FDR
si_rows = []
h = 25
outcome = "idx_orig34_weighted"
for t in TOPICS:
    d = df.copy()
    d["G_t"] = d[outcome]
    d["y"] = d[outcome].shift(-h)
    d["ix_z"] = zscore(d[f"ix_{t}"])
    cols = ["G_t", f"z_sal_{t}", f"z_sen_{t}", "ix_z"]
    d = d.dropna(subset=cols + ["y"]).reset_index(drop=True)
    X = sm.add_constant(d[cols])
    fit = sm.OLS(d["y"], X).fit(cov_type="HAC", cov_kwds={"maxlags": NW_BW})
    for lbl, c in [("b1_topic", f"z_sal_{t}"), ("b2_sent", f"z_sen_{t}"), ("b3_interaction", "ix_z")]:
        si_rows.append({"topic": t, "coef_name": lbl, "coef": fit.params[c],
                        "HAC_se": fit.bse[c], "HAC_p": fit.pvalues[c],
                        "adj_R2": fit.rsquared_adj, "nobs": int(fit.nobs)})
si = pd.DataFrame(si_rows)
rej, p_adj, _, _ = multipletests(si["HAC_p"], alpha=0.10, method="fdr_bh")
si["p_BH"] = p_adj
si["sig_FDR_q10"] = rej
si["sig_raw_p05"] = si["HAC_p"] < 0.05
for c in ["coef", "HAC_se", "HAC_p", "p_BH", "adj_R2"]:
    si[c] = si[c].round(4)
si.to_csv(OUT + "table_si_pertopic_h25_fdr.csv", index=False)

# ------------------------------------------------------------------- 5. figures
# Fig 1: energy interaction across horizons, both outcomes
fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
labels = {"idx_orig34_weighted": "Original 34-term index (paper)",
          "idx_cleaned_weighted": "Cleaned index"}
for ax, outcome in zip(axes, OUTCOMES):
    dta = horizon_fig_data[outcome]
    hs = [x[0] for x in dta]; cs = [x[1] for x in dta]
    lo = [x[1] - x[2] for x in dta]; hi = [x[3] - x[1] for x in dta]
    ax.axhline(0, color="0.6", lw=0.8)
    ax.errorbar(hs, cs, yerr=[lo, hi], fmt="o", color="#1f5fa8", capsize=3, lw=1.4)
    ax.set_title(labels[outcome])
    ax.set_xlabel("Horizon h (weeks)")
    ax.set_xticks(HORIZONS)
axes[0].set_ylabel("Energy × sentiment coefficient (c1, std.)")
fig.suptitle("Energy-interaction coefficient with 95% moving-block bootstrap CIs", y=1.02, fontsize=11)
fig.tight_layout()
fig.savefig(OUT + "fig_energy_horizon.png", dpi=300, bbox_inches="tight")
fig.savefig(OUT + "fig_energy_horizon.pdf", bbox_inches="tight")
plt.close(fig)

# Fig 2: dot-and-CI of topic main effects + interactions at h=25
fig, axes = plt.subplots(1, 2, figsize=(9.5, 5.2), sharey=True)
for ax, outcome in zip(axes, OUTCOMES):
    fit, X, y = fits_h25[outcome]
    terms = SAL_COLS + ["energy_ix", "pooled_other_ix"]
    names = [c.replace("z_sal_", "") for c in SAL_COLS] + ["energy × sent", "pooled other × sent"]
    coefs = fit.params[terms].values
    ci = fit.conf_int().loc[terms].values  # HAC 95% CI
    order = np.argsort(coefs)
    ypos = np.arange(len(terms))
    colors = ["#c23b22" if terms[i] in ("energy_ix", "pooled_other_ix")
              else ("#1f5fa8" if terms[i] == "z_sal_energy" else "0.35") for i in order]
    ax.axvline(0, color="0.6", lw=0.8)
    for j, i in enumerate(order):
        ax.plot([ci[i, 0], ci[i, 1]], [j, j], color=colors[j], lw=1.4)
        ax.plot(coefs[i], j, "o", color=colors[j], ms=5)
    ax.set_yticks(ypos)
    ax.set_yticklabels([names[i] for i in order])
    ax.set_title(labels[outcome])
    ax.set_xlabel("Coefficient (std. predictors), h = 25")
fig.suptitle("Topic main effects and interaction terms at h = 25 (HAC 95% CIs)", y=1.0, fontsize=11)
fig.tight_layout()
fig.savefig(OUT + "fig_topic_coefficients.png", dpi=300, bbox_inches="tight")
fig.savefig(OUT + "fig_topic_coefficients.pdf", bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------------------------ console dump
print("MERGED_N", N_MERGED)
print("\n== zero-week shares ==\n", zero_tab.to_string(index=False))
print("\n== pooled model ==\n", pooled_tab.to_string(index=False))
print("\n== VIF ==\n", vif_tab.to_string(index=False))
print("\n== dominance ==\n", pd.DataFrame(dom_rows).to_string(index=False))
print("\n== SI per-topic (FDR) ==\n", si.to_string(index=False))
