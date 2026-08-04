"""Two strengthening analyses for the tone claims:
A. Difference curve dr(L) = r_pos(L) - r_neg(L) with joint block-bootstrap CI at every lead,
   plus a global label-permutation test of max_L |dr(L)| (tone labels permuted within outlet).
B. Noise benchmark for the OOS ladder: circularly shift the 12-tone block (preserves
   autocorrelation, destroys alignment); distribution of OOS RMSE across shifts vs real tones.
"""
# --- replication-package prelude (added by build_package.py) ------------------
# The original research scripts pointed at an absolute project root. In this
# package the repository locates itself, so it runs from any checkout location.
from pathlib import Path as _RepoPath
_REPO = str(_RepoPath(__file__).resolve().parents[2])
# ------------------------------------------------------------------------------
import pandas as pd, numpy as np
import statsmodels.api as sm

ROOT=_REPO
master = pd.read_csv(f"{ROOT}/analysis/08_rebuild/weekly_series_master.csv", parse_dates=["Week"])
tones = pd.read_csv(f"{ROOT}/analysis/10_volume_vs_tone/climate_clean_tones.csv", parse_dates=["date"])
tones["wk"] = tones["date"].dt.to_period("W-SAT").dt.start_time
G = master["idx_cleaned_weighted"].values
weeks = master["Week"].values

def split_series(df, med):
    out={}
    for name, sub in [("pos", df[df.tone>=med]), ("neg", df[df.tone<med])]:
        wk = sub.groupby(["wk","outlet"]).size().unstack(fill_value=0).reindex(weeks, fill_value=0)
        z = wk.apply(lambda c: (c-c.mean())/c.std() if c.std()>0 else c*0)
        out[name]=z.mean(1).values
    return out["pos"], out["neg"]

med = tones["tone"].median()
mp, mn = split_series(tones, med)

def profile(x):
    return np.array([np.corrcoef(x[:-L],G[L:])[0,1] if L>0 else np.corrcoef(x,G)[0,1] for L in range(53)])
rp, rn = profile(mp), profile(mn)
dr = rp - rn

# --- A1: joint block-bootstrap CI for dr(L) at every lead ---
rng=np.random.default_rng(42); n=len(G); bl=13; B=1000
boot=np.empty((B,53))
for b in range(B):
    idx=np.concatenate([np.arange(s,s+bl)%n for s in rng.integers(0,n,n//bl+1)])[:n]
    xp,xn,g = mp[idx],mn[idx],G[idx]
    for L in range(53):
        if L==0: boot[b,L]=np.corrcoef(xp,g)[0,1]-np.corrcoef(xn,g)[0,1]
        else:    boot[b,L]=np.corrcoef(xp[:-L],g[L:])[0,1]-np.corrcoef(xn[:-L],g[L:])[0,1]
lo,hi = np.percentile(boot,2.5,axis=0), np.percentile(boot,97.5,axis=0)
covers_zero = ((lo<=0)&(hi>=0)).mean()
print(f"A1: dr CI covers zero at {covers_zero:.0%} of leads; max|dr| observed {np.abs(dr).max():.3f} at L={np.abs(dr).argmax()}", flush=True)

# --- A2: global label-permutation test (permute tone within outlet) ---
# The restricted window is the set of leads at which the primary lead--lag profile
# actually exceeds its own scan-corrected band, read from the archived profile rather
# than assumed to be a contiguous span.
prof_primary = pd.read_csv(f"{ROOT}/analysis/09_stats/lag_profile/focal_profile_primary.csv")
band_primary = float(prof_primary["band_global"].iloc[0])
CLEARING = sorted(int(L) for L in prof_primary.loc[
    (prof_primary["L"] > 0) & (prof_primary["r"] > band_primary), "L"])
print(f"A2: band-clearing leads (band {band_primary:.4f}): {CLEARING}", flush=True)

P=500; rng2=np.random.default_rng(7); maxs=np.empty(P); maxs_win=np.empty(P)
tt=tones.copy()
for p in range(P):
    tt["tone_p"]=tt.groupby("outlet")["tone"].transform(lambda s: s.sample(frac=1, random_state=int(rng2.integers(1e9))).values)
    d2=tt.rename(columns={"tone":"orig","tone_p":"tone"})
    mpp,mnp = split_series(d2, d2["tone"].median())
    drp = profile(mpp)-profile(mnp)
    maxs[p]=np.abs(drp).max()
    maxs_win[p]=np.abs(drp[CLEARING]).max()
obs_max=np.abs(dr).max()
obs_win=np.abs(dr[CLEARING]).max()
pval=(1+(maxs>=obs_max).sum())/(P+1)
pval_win=(1+(maxs_win>=obs_win).sum())/(P+1)
print(f"A2: permutation global test: observed max|dr|={obs_max:.3f}, null 95th pct={np.percentile(maxs,95):.3f}, p={pval:.3f}", flush=True)
print(f"A2: restricted to band-clearing leads: observed {obs_win:.3f}, null 95th pct={np.percentile(maxs_win,95):.3f}, p={pval_win:.3f}", flush=True)

np.savez(f"{ROOT}/analysis/10_volume_vs_tone/dr_curve.npz", dr=dr, lo=lo, hi=hi, rp=rp, rn=rn,
         perm_null_95=np.percentile(maxs,95), perm_p=pval, obs_max=obs_max,
         clearing_leads=np.array(CLEARING), perm_null_95_window=np.percentile(maxs_win,95),
         perm_p_window=pval_win, obs_max_window=obs_win, band_primary=band_primary)

# --- B: noise benchmark for OOS ladder ---
ps = pd.read_csv(f"{ROOT}/data/07_weekly_series/weekly_topic_salience_by_category.csv"); ps["Week"]=pd.to_datetime(ps["0"])
pt = pd.read_csv(f"{ROOT}/data/07_weekly_series/weekly_topic_sentiment_by_category.csv"); pt["Week"]=pd.to_datetime(pt["0"])
CATS=["art","disaster","economy","education","energy","medical","nature","politics","pollution","religion","society","technology"]
df = master.merge(ps[["Week"]+CATS].rename(columns={c:f"sal_{c}" for c in CATS}), on="Week")
df = df.merge(pt[["Week"]+CATS].rename(columns={c:f"ton_{c}" for c in CATS}), on="Week")
for c in [c for c in df.columns if c.startswith(("sal_","ton_"))]:
    df[c]=(df[c]-df[c].mean())/df[c].std()
h=25
df["y"]=df["idx_cleaned_weighted"].shift(-h); df["G1"]=df["idx_cleaned_weighted"]; df["G2"]=df["idx_cleaned_weighted"].shift(1)
woy=df["Week"].dt.isocalendar().week.astype(float)
df["s1"],df["c1s"],df["s2"],df["c2s"]=np.sin(2*np.pi*woy/52),np.cos(2*np.pi*woy/52),np.sin(4*np.pi*woy/52),np.cos(4*np.pi*woy/52)
df["covid"]=((df.Week>="2020-03-08")&(df.Week<="2021-12-26")).astype(float)
df["ukr"]=(df.Week>="2022-02-20").astype(float)
d=df.dropna(subset=["y","G2"]).reset_index(drop=True)
base=["G1","G2","s1","c1s","s2","c2s","covid","ukr"]; sal=[f"sal_{c}" for c in CATS]; ton=[f"ton_{c}" for c in CATS]

def oos(dd, cols):
    n=len(dd); errs=[]
    for f in [0.5,0.6,0.7,0.8,0.9]:
        a,b=int(n*f),int(n*(f+0.1))
        tr,te=dd.iloc[:a],dd.iloc[a:b]
        X=sm.add_constant(tr[cols]); mfit=sm.OLS(tr["y"],X).fit()
        Xt=sm.add_constant(te[cols],has_constant="add")
        errs.extend((te["y"]-mfit.predict(Xt))**2)
    return float(np.sqrt(np.mean(errs)))

r_m1=oos(d, base+sal); r_m2=oos(d, base+sal+ton)
print(f"B: OOS RMSE +volumes {r_m1:.2f}, +real tones {r_m2:.2f}", flush=True)
S=200; rng3=np.random.default_rng(11); noise_rmse=np.empty(S)
tonmat=d[ton].values; nn=len(d)
for si in range(S):
    k=int(rng3.integers(20,nn-20))
    dd=d.copy(); dd[ton]=np.roll(tonmat, k, axis=0)
    noise_rmse[si]=oos(dd, base+sal+ton)
print(f"B: shifted-tone RMSE mean {noise_rmse.mean():.2f}, 2.5-97.5 pct [{np.percentile(noise_rmse,2.5):.2f},{np.percentile(noise_rmse,97.5):.2f}]; real within? {np.percentile(noise_rmse,2.5)<=r_m2<=np.percentile(noise_rmse,97.5)}", flush=True)
np.savez(f"{ROOT}/analysis/10_volume_vs_tone/noise_benchmark.npz", r_m1=r_m1, r_m2=r_m2, noise=noise_rmse, r_m0=oos(d,base))
print("DONE")
