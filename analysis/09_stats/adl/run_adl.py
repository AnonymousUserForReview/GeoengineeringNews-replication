#!/usr/bin/env python3
"""Dynamic-specification ladder (ADL) + COVID + events.

Two variants:
  original: G = idx_orig34_weighted,  M = m_all46_pooled_z
  primary : G = idx_cleaned_weighted, M = m_climate_clean_pooled_z
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
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import acorr_breusch_godfrey
from statsmodels.tsa.api import VAR
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
np.random.seed(20260711)

OUT = (_REPO + "/analysis/09_stats/adl")
MASTER = (_REPO + "/analysis/08_rebuild/weekly_series_master.csv")

df = pd.read_csv(MASTER, parse_dates=["Week"]).set_index("Week").sort_index()

VARIANTS = {
    "original": dict(G="idx_orig34_weighted", M="m_all46_pooled_z"),
    "primary":  dict(G="idx_cleaned_weighted", M="m_climate_clean_pooled_z"),
}

# --- key dates ---
COVID_START, COVID_END = pd.Timestamp("2020-03-08"), pd.Timestamp("2021-12-26")
AR6 = (pd.Timestamp("2021-08-08"), pd.Timestamp("2021-08-29"))
COP26 = (pd.Timestamp("2021-10-24"), pd.Timestamp("2021-11-14"))
UKR = pd.Timestamp("2022-02-20")

idx = df.index
df["D_covid"] = ((idx >= COVID_START) & (idx <= COVID_END)).astype(float)
df["D_ar6"] = ((idx >= AR6[0]) & (idx <= AR6[1])).astype(float)
df["D_cop26"] = ((idx >= COP26[0]) & (idx <= COP26[1])).astype(float)
df["D_ukr"] = (idx >= UKR).astype(float)

# harmonics: 2 pairs, week-of-year
doy = idx.dayofyear.values.astype(float)
frac = doy / 365.25
for k in (1, 2):
    df[f"sin{k}"] = np.sin(2 * np.pi * k * frac)
    df[f"cos{k}"] = np.cos(2 * np.pi * k * frac)
HARM = ["sin1", "cos1", "sin2", "cos2"]

NBINS = 7  # bins t-1..t-4, ..., t-25..t-28
NW_LAGS = 19


def make_bins(s, nbins=NBINS, width=4):
    out = {}
    for b in range(1, nbins + 1):
        lags = range(width * (b - 1) + 1, width * b + 1)
        out[f"Mbar{b}"] = pd.concat([s.shift(l) for l in lags], axis=1).mean(axis=1)
    return pd.DataFrame(out)


def build_frame(g, m, diff=False):
    """Regression frame with G lags placeholder (p added later), bins, harmonics, dummies."""
    if diff:
        g = g.diff()
        m = m.diff()
    bins = make_bins(m)
    fr = pd.concat([g.rename("y"), bins, df[HARM + ["D_covid", "D_ar6", "D_cop26", "D_ukr"]]], axis=1)
    for j in range(1, 7):
        fr[f"ylag{j}"] = g.shift(j)
    return fr


BIN_COLS = [f"Mbar{b}" for b in range(1, NBINS + 1)]


def fit_adl(fr, p, extra=None, interact=False):
    cols = [f"ylag{j}" for j in range(1, p + 1)] + BIN_COLS + HARM + ["D_covid"]
    if extra:
        cols += extra
    d = fr[["y"] + cols].dropna()
    X = d[cols].copy()
    if interact:
        for b in BIN_COLS:
            X[f"cov_x_{b}"] = d["D_covid"] * d[b]
    X = sm.add_constant(X)
    res_ols = sm.OLS(d["y"], X).fit()
    res_nw = sm.OLS(d["y"], X).fit(cov_type="HAC", cov_kwds={"maxlags": NW_LAGS})
    return d, res_ols, res_nw


def choose_p(fr):
    # common sample across p=1..4 for fair BIC
    bics = {}
    common = fr[["y"] + [f"ylag{j}" for j in range(1, 5)] + BIN_COLS + HARM + ["D_covid"]].dropna()
    for p in range(1, 5):
        cols = [f"ylag{j}" for j in range(1, p + 1)] + BIN_COLS + HARM + ["D_covid"]
        X = sm.add_constant(common[cols])
        bics[p] = sm.OLS(common["y"], X).fit().bic
    p_bic = min(bics, key=bics.get)
    # whiten residuals: BG test (4 lags), extend p up to 6
    p = p_bic
    while p <= 6:
        _, res_ols, _ = fit_adl(fr, p)
        bg_p = acorr_breusch_godfrey(res_ols, nlags=4)[1]
        if bg_p > 0.05 or p == 6:
            break
        p += 1
    return p_bic, p, bics, bg_p


def lrm_and_cum(params, p, sd_m, horizons=range(4, 29, 4), tmax=300):
    phis = np.array([params.get(f"ylag{j}", 0.0) for j in range(1, p + 1)])
    betas = np.array([params[b] for b in BIN_COLS])
    denom = 1.0 - phis.sum()
    lrm = betas.sum() / denom if abs(denom) > 1e-8 else np.nan
    # sustained 1-SD shock path
    x = np.zeros(tmax + 1)
    for t in range(1, tmax + 1):
        fracs = np.clip(t - 4 * np.arange(1, NBINS + 1) + 3, 0, 4) / 4.0
        x[t] = sum(phis[j] * x[t - 1 - j] for j in range(p) if t - 1 - j >= 0) \
               + sd_m * float(betas @ fracs)
    cums = {h: x[h] for h in horizons}
    return lrm * sd_m, cums, phis.sum(), betas.sum()


def block_bootstrap_rows(n, block=13, rng=None):
    rng = rng or np.random.default_rng()
    starts = rng.integers(0, n - block + 1, size=int(np.ceil(n / block)))
    rows = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
    return rows


def moving_block_ci(d, p, sd_m, nrep=1000, block=13):
    cols = [f"ylag{j}" for j in range(1, p + 1)] + BIN_COLS + HARM + ["D_covid"]
    ymat = d["y"].values
    Xmat = sm.add_constant(d[cols]).values
    names = ["const"] + cols
    rng = np.random.default_rng(42)
    n = len(d)
    lrms, cum_paths = [], []
    horizons = list(range(4, 29, 4))
    for _ in range(nrep):
        rows = block_bootstrap_rows(n, block, rng)
        yb, Xb = ymat[rows], Xmat[rows]
        try:
            beta = np.linalg.lstsq(Xb, yb, rcond=None)[0]
        except np.linalg.LinAlgError:
            continue
        pars = pd.Series(beta, index=names)
        lrm, cums, phisum, _ = lrm_and_cum(pars, p, sd_m, horizons)
        if not np.isfinite(lrm) or phisum >= 0.999:
            lrm = np.nan
        lrms.append(lrm)
        cum_paths.append([cums[h] for h in horizons])
    lrms = np.array(lrms, float)
    cum_paths = np.array(cum_paths, float)
    lrm_ci = np.nanpercentile(lrms, [2.5, 97.5])
    cum_ci = np.nanpercentile(cum_paths, [2.5, 97.5], axis=0)
    return lrm_ci, dict(zip(horizons, cum_ci.T)), np.nanmean(np.isnan(lrms))


def wald_bins(res_nw, prefix_cols):
    R = " = 0, ".join(prefix_cols) + " = 0"
    w = res_nw.wald_test(R, use_f=True, scalar=True)
    return float(w.statistic), float(w.pvalue), int(w.df_num)


def xcorr(m, g, L):
    a = pd.concat([m.rename("m"), g.shift(-L).rename("g")], axis=1).dropna()
    return a["m"].corr(a["g"]), len(a)


def peak_lag(m, g, lmax=35):
    rs = {L: xcorr(m, g, L)[0] for L in range(0, lmax + 1)}
    Ls = max(rs, key=lambda L: rs[L])
    return Ls, rs[Ls], rs


def midfreq_corr(m, g, L, k, nrep=2000, seed=7):
    """Non-overlapping k-week means; correlate M-block i with G-block i+offset."""
    n = len(m)
    nb = n // k
    mb = np.array([m.values[i * k:(i + 1) * k].mean() for i in range(nb)])
    gb = np.array([g.values[i * k:(i + 1) * k].mean() for i in range(nb)])
    off = int(round(L / k))
    if off > 0:
        x, y = mb[:-off], gb[off:]
    else:
        x, y = mb, gb
    r = np.corrcoef(x, y)[0, 1]
    # circular block bootstrap on the aligned pairs
    rng = np.random.default_rng(seed)
    npairs = len(x)
    bl = max(2, min(4, npairs // 4))
    rs = []
    for _ in range(nrep):
        starts = rng.integers(0, npairs, size=int(np.ceil(npairs / bl)))
        rows = np.concatenate([(np.arange(s, s + bl) % npairs) for s in starts])[:npairs]
        xb, yb = x[rows], y[rows]
        if np.std(xb) > 0 and np.std(yb) > 0:
            rs.append(np.corrcoef(xb, yb)[0, 1])
    lo, hi = np.percentile(rs, [2.5, 97.5])
    return r, lo, hi, off, npairs, bl


# ============================== main loop ==============================
stationarity_rows = []
coef_frames = []
lrm_rows = []
event_rows = []
speccov_rows = []
diff_rows = []
midfreq_rows = []
granger_rows = []
report_bits = {}
fig_data = {}

for vname, vv in VARIANTS.items():
    G = df[vv["G"]].astype(float)
    M = df[vv["M"]].astype(float)

    # ---- 1. stationarity ----
    for label, s in [("M", M), ("G", G), ("dM", M.diff().dropna()), ("dG", G.diff().dropna())]:
        adf_stat, adf_p, adf_lags, *_ = adfuller(s.dropna(), regression="ct", autolag="AIC")
        kp_reg = "ct" if label in ("M", "G") else "c"
        kp_stat, kp_p, kp_lags, _ = kpss(s.dropna(), regression=kp_reg, nlags="auto")
        stationarity_rows.append(dict(variant=vname, series=label,
                                      adf_stat=adf_stat, adf_p=adf_p, adf_lags=adf_lags,
                                      kpss_stat=kp_stat, kpss_p=kp_p, kpss_reg=kp_reg,
                                      verdict=("stationary" if (adf_p < 0.05 and kp_p > 0.05)
                                               else "nonstationary" if (adf_p >= 0.05 and kp_p <= 0.05)
                                               else "ambiguous")))

    # ---- 2. ADL levels ----
    fr = build_frame(G, M)
    p_bic, p_used, bics, bg_p = choose_p(fr)
    d, res_ols, res_nw = fit_adl(fr, p_used)
    sd_m = M.loc[d.index].std()

    co = pd.DataFrame({"coef": res_nw.params, "nw_se": res_nw.bse,
                       "t": res_nw.tvalues, "p": res_nw.pvalues})
    co.insert(0, "variant", vname)
    co.insert(1, "model", "ADL_levels")
    coef_frames.append(co.reset_index().rename(columns={"index": "term"}))

    Fb, pb, dfb = wald_bins(res_nw, BIN_COLS)

    # ---- 3. LRM + cumulative multipliers ----
    lrm_sd, cums, phisum, betasum = lrm_and_cum(res_nw.params, p_used, sd_m)
    lrm_ci, cum_ci, badfrac = moving_block_ci(d, p_used, sd_m, nrep=1000, block=13)
    lrm_rows.append(dict(variant=vname, p_bic=p_bic, p_used=p_used, bg_p_final=bg_p,
                         n=len(d), sd_M=sd_m, sum_phi=phisum, sum_beta=betasum,
                         LRM_per_unit=betasum / (1 - phisum),
                         LRM_1sd=lrm_sd, LRM_1sd_lo=lrm_ci[0], LRM_1sd_hi=lrm_ci[1],
                         bins_F=Fb, bins_F_p=pb, boot_nan_frac=badfrac,
                         **{f"cum_h{h}": cums[h] for h in cums},
                         **{f"cum_h{h}_lo": cum_ci[h][0] for h in cum_ci},
                         **{f"cum_h{h}_hi": cum_ci[h][1] for h in cum_ci}))
    fig_data[vname] = dict(h=list(cums.keys()), c=[cums[h] for h in cums],
                           lo=[cum_ci[h][0] for h in cum_ci], hi=[cum_ci[h][1] for h in cum_ci],
                           lrm=lrm_sd, lrm_ci=lrm_ci)

    # ---- 4. events + covid interaction ----
    d4, res4_ols, res4_nw = fit_adl(fr, p_used, extra=["D_ar6", "D_cop26", "D_ukr"])
    Fb4, pb4, _ = wald_bins(res4_nw, BIN_COLS)
    co4 = pd.DataFrame({"coef": res4_nw.params, "nw_se": res4_nw.bse,
                        "t": res4_nw.tvalues, "p": res4_nw.pvalues})
    co4.insert(0, "variant", vname)
    co4.insert(1, "model", "ADL_levels_events")
    coef_frames.append(co4.reset_index().rename(columns={"index": "term"}))

    dI, resI_ols, resI_nw = fit_adl(fr, p_used, interact=True)
    int_cols = [f"cov_x_{b}" for b in BIN_COLS]
    Fi, pi, dfi = wald_bins(resI_nw, int_cols)
    event_rows.append(dict(variant=vname,
                           bins_F_events=Fb4, bins_p_events=pb4,
                           ar6_coef=res4_nw.params["D_ar6"], ar6_p=res4_nw.pvalues["D_ar6"],
                           cop26_coef=res4_nw.params["D_cop26"], cop26_p=res4_nw.pvalues["D_cop26"],
                           ukr_coef=res4_nw.params["D_ukr"], ukr_p=res4_nw.pvalues["D_ukr"],
                           covid_x_M_F=Fi, covid_x_M_p=pi, covid_x_M_df=dfi))

    # ---- 5. COVID Spec C ----
    Lstar, rstar, _ = peak_lag(M, G)
    pairs = pd.concat([M.rename("m"), G.shift(-Lstar).rename("g")], axis=1).dropna()
    g_week = pairs.index + pd.Timedelta(weeks=Lstar)
    keep = (pairs.index.year != 2020) & (g_week.year != 2020)
    r_ex = pairs.loc[keep, "m"].corr(pairs.loc[keep, "g"])
    speccov_rows.append(dict(variant=vname, peak_lag=Lstar, r_full=rstar, n_full=len(pairs),
                             r_excl2020=r_ex, n_excl2020=int(keep.sum())))

    # ---- 6. first differences ----
    frd = build_frame(G, M, diff=True)
    pd_bic, pd_used, _, bgd_p = choose_p(frd)
    dd, resd_ols, resd_nw = fit_adl(frd, pd_used)
    Fbd, pbd, _ = wald_bins(resd_nw, BIN_COLS)
    cod = pd.DataFrame({"coef": resd_nw.params, "nw_se": resd_nw.bse,
                        "t": resd_nw.tvalues, "p": resd_nw.pvalues})
    cod.insert(0, "variant", vname)
    cod.insert(1, "model", "ADL_diff")
    coef_frames.append(cod.reset_index().rename(columns={"index": "term"}))
    diff_rows.append(dict(variant=vname, p_bic=pd_bic, p_used=pd_used, bg_p_final=bgd_p,
                          n=len(dd), bins_F=Fbd, bins_F_p=pbd,
                          adjR2=resd_ols.rsquared_adj))

    for k in (8, 13):
        r, lo, hi, off, npairs, bl = midfreq_corr(M, G, Lstar, k)
        midfreq_rows.append(dict(variant=vname, block_weeks=k, offset_blocks=off,
                                 peak_lag_weeks=Lstar, r=r, ci_lo=lo, ci_hi=hi,
                                 n_pairs=npairs, boot_block=bl))

    # ---- 7. Granger in differences ----
    dat = pd.concat([G.diff().rename("dG"), M.diff().rename("dM")], axis=1).dropna()
    var = VAR(dat)
    sel = var.select_order(maxlags=12)
    kbic = max(1, sel.bic)
    vres = var.fit(kbic)
    for cause, effect in [("dM", "dG"), ("dG", "dM")]:
        tc = vres.test_causality(effect, [cause], kind="f")
        granger_rows.append(dict(variant=vname, direction=f"{cause} -> {effect}",
                                 bic_lags=kbic, F=float(tc.test_statistic),
                                 p=float(tc.pvalue)))

    report_bits[vname] = dict(p_bic=p_bic, p_used=p_used, bg_p=bg_p, n=len(d),
                              bins_F=Fb, bins_p=pb, lrm=lrm_sd, lrm_ci=lrm_ci,
                              phisum=phisum, sd_m=sd_m,
                              Lstar=Lstar, rstar=rstar, r_ex=r_ex,
                              Fbd=Fbd, pbd=pbd, pd_used=pd_used,
                              Fi=Fi, pi=pi, Fb4=Fb4, pb4=pb4)

# ============================== save tables ==============================
pd.DataFrame(stationarity_rows).to_csv(f"{OUT}/stationarity.csv", index=False)
pd.concat(coef_frames, ignore_index=True).to_csv(f"{OUT}/adl_coefficients.csv", index=False)
pd.DataFrame(lrm_rows).to_csv(f"{OUT}/lrm_multipliers.csv", index=False)
pd.DataFrame(event_rows).to_csv(f"{OUT}/event_terms.csv", index=False)
pd.DataFrame(speccov_rows).to_csv(f"{OUT}/covid_specC_correlation.csv", index=False)
pd.DataFrame(diff_rows).to_csv(f"{OUT}/first_difference_adl.csv", index=False)
pd.DataFrame(midfreq_rows).to_csv(f"{OUT}/midfrequency_correlations.csv", index=False)
pd.DataFrame(granger_rows).to_csv(f"{OUT}/granger_differences.csv", index=False)

# ============================== figure ==============================
COL = {"original": "#2a78d6", "primary": "#1baf7a"}
LBL = {"original": "Original (idx_orig34 × m_all46)",
       "primary": "Primary (idx_cleaned × m_climate_clean)"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#888", "axes.labelcolor": "#333",
                     "xtick.color": "#555", "ytick.color": "#555"})
fig, ax = plt.subplots(figsize=(7.2, 4.4))
for vname, fd in fig_data.items():
    h = [0] + fd["h"]
    c = [0.0] + fd["c"]
    lo = [0.0] + fd["lo"]
    hi = [0.0] + fd["hi"]
    ax.step(h, c, where="post", color=COL[vname], lw=2, label=LBL[vname])
    ax.fill_between(h, lo, hi, step="post", color=COL[vname], alpha=0.15, lw=0)
    ytxt = {"original": 2.2, "primary": 7.2}[vname]
    ax.annotate(f"LRM = {fd['lrm']:.2f}\n95% CI [{fd['lrm_ci'][0]:.2f}, {fd['lrm_ci'][1]:.2f}]",
                xy=(28, fd["c"][-1]), xytext=(28.5, ytxt),
                color=COL[vname], fontsize=8.5, va="center")
ax.axhline(0, color="#bbb", lw=0.8)
ax.set_xlabel("Horizon (weeks after onset of sustained 1-SD media shock)")
ax.set_ylabel("Cumulative effect on index (index points)")
ax.set_title("Cumulative multipliers from ADL in levels, moving-block bootstrap 95% CI",
             fontsize=10.5, color="#333")
ax.set_xticks(range(0, 29, 4))
ax.set_xlim(0, 34)
ax.legend(frameon=False, loc="upper left", fontsize=9)
fig.tight_layout()
fig.savefig(f"{OUT}/fig_cumulative_multiplier.png", dpi=300)
fig.savefig(f"{OUT}/fig_cumulative_multiplier.pdf")
plt.close(fig)

# stash report bits for the report writer
import json
with open(f"{OUT}/_report_bits.json", "w") as f:
    json.dump({k: {kk: (float(x) if isinstance(x, (int, float, np.floating, np.integer))
                        else [float(z) for z in x] if isinstance(x, (tuple, list, np.ndarray))
                        else x) for kk, x in v.items()} for k, v in report_bits.items()},
              f, indent=2, default=float)
print("DONE")
for v, b in report_bits.items():
    print(v, {k: (round(x, 4) if isinstance(x, float) else x) for k, x in b.items()
              if not isinstance(x, (tuple, list, np.ndarray))})
