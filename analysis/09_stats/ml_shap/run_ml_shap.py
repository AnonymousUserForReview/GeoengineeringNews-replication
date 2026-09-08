"""H2 predictive pipeline with proper out-of-sample evaluation + SHAP."""
# --- replication-package prelude (added by build_package.py) ------------------
# The original research scripts pointed at an absolute project root. In this
# package the repository locates itself, so it runs from any checkout location.
from pathlib import Path as _RepoPath
_REPO = str(_RepoPath(__file__).resolve().parents[3])
# ------------------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import shap

OUT = (_REPO + "/analysis/09_stats/ml_shap")
np.random.seed(0)

master = pd.read_csv((_REPO + "/analysis/08_rebuild/weekly_series_master.csv"), parse_dates=["Week"]).set_index("Week")
sal = pd.read_csv((_REPO + "/data/07_weekly_series/weekly_topic_salience_by_category.csv"), parse_dates=["0"]).set_index("0")
sen = pd.read_csv((_REPO + "/data/07_weekly_series/weekly_topic_sentiment_by_category.csv"), parse_dates=["0"]).set_index("0")
cats = ["art","disaster","economy","education","energy","medical","nature","politics","pollution","religion","society","technology"]
sal = sal[cats].add_prefix("sal_")
sen = sen[cats].add_prefix("sen_")

feat = sal.join(sen, how="inner")
# z-score features (full sample z-scoring only for scale; note re leakage is minor but do train-based? Paper: z-scored. We z-score using full sample for features as in paper.)
feat = (feat - feat.mean()) / feat.std(ddof=0)

combos = [(idx, h) for idx in ["idx_orig34_weighted", "idx_cleaned_weighted"] for h in [16, 25]]

perf_rows = []
shap_results = {}  # (idx,h,model) -> Series mean|SHAP|

for idx_name, h in combos:
    G = master[idx_name]
    df = feat.copy()
    df["G_t"] = G.reindex(df.index)
    df["y"] = G.reindex(df.index).shift(-h).values
    # seasonal naive: G_{t+h-52} relative to target date t+h => G at t+h-52 = shift(-(h-52)) = shift(52-h)
    df["seasonal"] = G.reindex(df.index).shift(52 - h).values
    df = df.dropna(subset=["y", "G_t"])
    X_cols = [c for c in df.columns if c not in ("y", "seasonal")]
    n = len(df)
    # 5 contiguous test folds covering second half
    half = n // 2
    fold_edges = np.linspace(half, n, 6).astype(int)
    key = f"{idx_name}_h{h}"
    shap_abs = {"XGB": [], "Linear": []}
    for k in range(5):
        s, e = fold_edges[k], fold_edges[k + 1]
        tr = df.iloc[:s]
        te = df.iloc[s:e]
        Xtr, ytr = tr[X_cols].values, tr["y"].values
        Xte, yte = te[X_cols].values, te["y"].values
        preds = {}
        lin = LinearRegression().fit(Xtr, ytr)
        preds["Linear"] = lin.predict(Xte)
        xgb = XGBRegressor(n_estimators=100, learning_rate=0.3, max_depth=6,
                           random_state=0, n_jobs=4)
        xgb.fit(Xtr, ytr)
        preds["XGB"] = xgb.predict(Xte)
        preds["TrainMean"] = np.full(len(te), ytr.mean())
        preds["Persistence"] = te["G_t"].values
        seas = te["seasonal"].values
        # fallback: if seasonal missing, use persistence
        seas = np.where(np.isnan(seas), te["G_t"].values, seas)
        preds["SeasonalNaive"] = seas
        for m, p in preds.items():
            perf_rows.append(dict(index=idx_name, horizon=h, model=m, fold=k + 1,
                                  n_test=len(te),
                                  MAE=mean_absolute_error(yte, p),
                                  RMSE=np.sqrt(mean_squared_error(yte, p)),
                                  R2=r2_score(yte, p)))
        # SHAP on held-out folds
        ex = shap.TreeExplainer(xgb)
        sv = ex.shap_values(Xte)
        shap_abs["XGB"].append(np.abs(sv))
        lex = shap.LinearExplainer(lin, Xtr)
        lsv = lex.shap_values(Xte)
        shap_abs["Linear"].append(np.abs(lsv))
    for m in ["XGB", "Linear"]:
        arr = np.vstack(shap_abs[m]).mean(axis=0)
        shap_results[(idx_name, h, m)] = pd.Series(arr, index=X_cols).sort_values(ascending=False)

perf = pd.DataFrame(perf_rows)
mean_perf = perf.groupby(["index", "horizon", "model"])[["MAE", "RMSE", "R2"]].mean().reset_index()
mean_perf["fold"] = "mean"
out_perf = pd.concat([perf, mean_perf], ignore_index=True)
out_perf.to_csv(f"{OUT}/model_performance.csv", index=False)

# SHAP rankings CSV
rank_rows = []
for (idx_name, h, m), s in shap_results.items():
    for rank, (fname, val) in enumerate(s.items(), 1):
        rank_rows.append(dict(index=idx_name, horizon=h, model=m, rank=rank,
                              feature=fname, mean_abs_shap=val))
pd.DataFrame(rank_rows).to_csv(f"{OUT}/shap_rankings.csv", index=False)

# Figure: 2x2 grid, XGB SHAP top-10 bars, G_t distinguished
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
for ax, (idx_name, h) in zip(axes.ravel(), combos):
    s = shap_results[(idx_name, h, "XGB")].head(10)[::-1]
    colors = ["#c0392b" if f == "G_t" else "#4878a8" for f in s.index]
    ax.barh(range(len(s)), s.values, color=colors)
    ax.set_yticks(range(len(s)))
    ax.set_yticklabels(s.index, fontsize=8)
    ax.set_title(f"{idx_name}, h={h}", fontsize=10)
    ax.set_xlabel("mean |SHAP| (held-out folds)", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
from matplotlib.patches import Patch
fig.legend(handles=[Patch(color="#c0392b", label="G_t (autoregressive)"),
                    Patch(color="#4878a8", label="topic features")],
           loc="lower center", ncol=2, frameon=False)
fig.suptitle("XGBoost mean |SHAP| on held-out folds (expanding-window CV)", fontsize=12)
fig.tight_layout(rect=[0, 0.04, 1, 1])
fig.savefig(f"{OUT}/fig_shap_summary.png", dpi=300)
fig.savefig(f"{OUT}/fig_shap_summary.pdf")

# Console summary for report writing
pd.set_option("display.width", 200)
print("=== mean OOS performance ===")
print(mean_perf.pivot_table(index=["index", "horizon"], columns="model", values=["RMSE", "R2"]).round(3).to_string())
print("\n=== per-fold R2 XGB vs persistence ===")
print(perf[perf.model.isin(["XGB", "Persistence", "Linear"])].pivot_table(index=["index", "horizon", "fold"], columns="model", values="R2").round(3).to_string())
print("\n=== top-10 SHAP per combo (XGB) ===")
for (idx_name, h, m), s in shap_results.items():
    if m == "XGB":
        print(f"\n{idx_name} h={h}:")
        print(s.head(10).round(3).to_string())
        nonar = [f for f in s.index if f != "G_t"]
        print("top non-AR:", nonar[0], "; G_t rank:", list(s.index).index("G_t") + 1)
print("\n=== Linear top-5 ===")
for (idx_name, h, m), s in shap_results.items():
    if m == "Linear":
        print(f"{idx_name} h={h}: " + ", ".join(f"{f}={v:.2f}" for f, v in s.head(5).items()))
print("\nn per combo:", {f"{i}_h{h}": None for i, h in combos})
