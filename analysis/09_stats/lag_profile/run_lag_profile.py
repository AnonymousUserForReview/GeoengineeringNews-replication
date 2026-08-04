#!/usr/bin/env python3
"""Corrected lag-profile analysis + influence diagnostics.

rho(L) = corr(M_t, G_{t+L}).  Positive L: media leads search interest.
"""
# --- replication-package prelude (added by build_package.py) ------------------
# The original research scripts pointed at an absolute project root. In this
# package the repository locates itself, so it runs from any checkout location.
from pathlib import Path as _RepoPath
_REPO = str(_RepoPath(__file__).resolve().parents[3])
# ------------------------------------------------------------------------------
import numpy as np
import pandas as pd

RNG = np.random.default_rng(20260711)
OUT = (_REPO + "/analysis/09_stats/lag_profile")
MASTER = (_REPO + "/analysis/08_rebuild/weekly_series_master.csv")

LAGS = np.arange(-26, 53)
M_VARS = ["m_all46_pooled_z", "m_climate_clean_pooled_z",
          "m_climate_filtered_pooled_z", "m_geo_specific_pooled_z"]
G_VARS = ["idx_orig34_weighted", "idx_cleaned_weighted",
          "idx_cleaned_equal", "idx_strict_umbrella_equal"]

df = pd.read_csv(MASTER, parse_dates=["Week"]).sort_values("Week").reset_index(drop=True)
weeks = df["Week"].to_numpy()
N = len(df)

# ---------------------------------------------------------------- helpers
def pairs_at_lag(m, g, L):
    """Return (x, y, idx_t) where y_t = g_{t+L}, valid t indices."""
    if L >= 0:
        t = np.arange(0, N - L)
    else:
        t = np.arange(-L, N)
    return m[t], g[t + L], t

def corr(x, y):
    if len(x) < 10:
        return np.nan
    sx, sy = x.std(), y.std()
    if sx == 0 or sy == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])

def profile(m, g, lags=LAGS):
    rs, ns = [], []
    for L in lags:
        x, y, _ = pairs_at_lag(m, g, L)
        rs.append(corr(x, y))
        ns.append(len(x))
    return np.array(rs), np.array(ns)

def block_bootstrap_ci(x, y, reps=2000, block=13, alpha=0.05):
    """Moving-block bootstrap of corr on PAIRED tuples (x_t, y_t)."""
    n = len(x)
    nblocks = int(np.ceil(n / block))
    starts = RNG.integers(0, n - block + 1, size=(reps, nblocks))
    idx = (starts[:, :, None] + np.arange(block)).reshape(reps, -1)[:, :n]
    bx, by = x[idx], y[idx]
    bxc = bx - bx.mean(axis=1, keepdims=True)
    byc = by - by.mean(axis=1, keepdims=True)
    denom = np.sqrt((bxc ** 2).sum(axis=1) * (byc ** 2).sum(axis=1))
    rboot = (bxc * byc).sum(axis=1) / denom
    return np.nanpercentile(rboot, 100 * alpha / 2), np.nanpercentile(rboot, 100 * (1 - alpha / 2))

def maxstat_band(m, g, lags, reps=2000):
    """Circular-shift null: 95th pct of max_L |rho(L)| under random shifts of G."""
    maxima = np.empty(reps)
    shifts = RNG.integers(53, N - 53, size=reps)  # keep shifts away from trivial values
    for i, s in enumerate(shifts):
        gs = np.roll(g, s)
        rs, _ = profile(m, gs, lags)
        maxima[i] = np.nanmax(np.abs(rs))
    return float(np.percentile(maxima, 95)), maxima

# ---------------------------------------------------------------- 1. full grid
rows = []
for mv in M_VARS:
    for gv in G_VARS:
        m, g = df[mv].to_numpy(float), df[gv].to_numpy(float)
        rs, ns = profile(m, g)
        for L, r, n in zip(LAGS, rs, ns):
            rows.append({"pair": f"{mv}__x__{gv}", "M": mv, "G": gv, "L": int(L),
                         "r": r, "n": int(n)})
grid = pd.DataFrame(rows)
grid.to_csv(f"{OUT}/lag_profile_grid.csv", index=False)
print("grid saved:", grid.shape)

# ---------------------------------------------------------------- 2. focal pairs
FOCAL = {
    "original": ("m_all46_pooled_z", "idx_orig34_weighted"),
    "primary": ("m_climate_clean_pooled_z", "idx_cleaned_weighted"),
}
SEC_LAGS = np.arange(12, 21)
focal_out = {}
for name, (mv, gv) in FOCAL.items():
    m, g = df[mv].to_numpy(float), df[gv].to_numpy(float)
    rs, ns = profile(m, g)
    lo = np.empty_like(rs); hi = np.empty_like(rs)
    for j, L in enumerate(LAGS):
        x, y, _ = pairs_at_lag(m, g, L)
        lo[j], hi[j] = block_bootstrap_ci(x, y)
    band_glob, _ = maxstat_band(m, g, LAGS)
    band_sec, _ = maxstat_band(m, g, SEC_LAGS)
    prof = pd.DataFrame({"L": LAGS, "r": rs, "n": ns, "ci_lo": lo, "ci_hi": hi})
    prof["band_global"] = band_glob
    prof["band_secondary_12_20"] = band_sec
    prof.to_csv(f"{OUT}/focal_profile_{name}.csv", index=False)
    focal_out[name] = dict(m=m, g=g, prof=prof, band_glob=band_glob, band_sec=band_sec,
                           mv=mv, gv=gv)
    ann = {int(L): float(rs[list(LAGS).index(L)]) for L in (12, 16, 20, 25)}
    pos = LAGS >= 0
    Lpeak = int(LAGS[pos][np.nanargmax(rs[pos])])
    focal_out[name]["ann"] = ann
    focal_out[name]["Lpeak"] = Lpeak
    focal_out[name]["rpeak"] = float(rs[list(LAGS).index(Lpeak)])
    print(f"{name}: peak L={Lpeak} r={focal_out[name]['rpeak']:.3f} "
          f"band_glob={band_glob:.3f} band_sec={band_sec:.3f} ann={ann}")

# ---------------------------------------------------------------- 3. influence
years = pd.DatetimeIndex(weeks).year
yr = np.clip(years, 2018, 2022)                      # fold edge weeks into 2018/2022
qtr_raw = pd.PeriodIndex(pd.DatetimeIndex(weeks), freq="Q")
lo_ord, hi_ord = pd.Period("2018Q1", "Q").ordinal, pd.Period("2022Q4", "Q").ordinal
qtr = np.array([str(pd.Period(ordinal=min(max(p.ordinal, lo_ord), hi_ord), freq="Q"))
                for p in qtr_raw])
h2_2021 = (weeks >= np.datetime64("2021-07-01")) & (weeks <= np.datetime64("2021-12-31"))

inf_rows = []
for name, fo in FOCAL.items():
    d = focal_out[name]
    m, g, L = d["m"], d["g"], d["Lpeak"]
    x, y, t = pairs_at_lag(m, g, L)
    te = t + L                                       # endpoint indices
    full_r = corr(x, y)
    inf_rows.append({"pair": name, "M": d["mv"], "G": d["gv"], "L_peak": L,
                     "exclusion": "none (full sample)", "r": full_r, "n": len(x)})
    def excl(mask_weeks, label):
        drop = mask_weeks[t] | mask_weeks[te]
        keep = ~drop
        inf_rows.append({"pair": name, "M": d["mv"], "G": d["gv"], "L_peak": L,
                         "exclusion": label, "r": corr(x[keep], y[keep]),
                         "n": int(keep.sum())})
    for y_ in range(2018, 2023):
        excl(yr == y_, f"drop {y_}")
    for q in sorted(set(qtr)):
        excl(np.array(qtr == q), f"drop {q}")
    excl(h2_2021, "drop 2021H2")

influence = pd.DataFrame(inf_rows)
influence.to_csv(f"{OUT}/influence.csv", index=False)
print("influence saved:", influence.shape)

# ---------------------------------------------------------------- 4. deseasonalization
woy = pd.DatetimeIndex(weeks).dayofyear.to_numpy() / 365.25
X = np.column_stack([np.ones(N)] + [f(2 * np.pi * k * woy)
                                    for k in (1, 2) for f in (np.sin, np.cos)])
def deseason(v):
    v = np.asarray(v, dtype=np.float64)
    beta, *_ = np.linalg.lstsq(X, v, rcond=None)
    resid = v - X @ beta
    assert np.all(np.isfinite(resid)), "non-finite deseason residuals"
    return resid

des_rows = []
for name, (mv, gv) in FOCAL.items():
    m_d = deseason(df[mv].to_numpy(float))
    g_d = deseason(df[gv].to_numpy(float))
    rs_d, ns_d = profile(m_d, g_d)
    for L, r, n in zip(LAGS, rs_d, ns_d):
        des_rows.append({"pair": name, "L": int(L), "r_deseason": r, "n": int(n)})
deseason_df = pd.DataFrame(des_rows)
# merge raw r for comparison
for name in FOCAL:
    raw = focal_out[name]["prof"][["L", "r"]]
    sel = deseason_df["pair"] == name
    deseason_df.loc[sel, "r_raw"] = deseason_df.loc[sel, "L"].map(
        dict(zip(raw["L"], raw["r"])))
deseason_df.to_csv(f"{OUT}/deseason_profile.csv", index=False)
for name in FOCAL:
    sub = deseason_df[deseason_df["pair"] == name]
    hi = sub[(sub.L >= 35) & (sub.L <= 45)]
    print(f"{name}: raw max r(35..45)={hi.r_raw.max():.3f} deseason max={hi.r_deseason.max():.3f}")

# peak-r across the four G variants at the primary media series & original media series
print("\n--- r at peak across G variants ---")
for mv in M_VARS:
    for gv in G_VARS:
        sub = grid[(grid.M == mv) & (grid.G == gv) & (grid.L >= 0)]
        i = sub.r.idxmax()
        print(f"{mv:32s} x {gv:28s} peak L={int(sub.loc[i,'L']):3d} r={sub.loc[i,'r']:.3f}")

np.save(f"{OUT}/_focal_cache.npy", np.array([0]))  # marker
print("\nDONE")
