"""Energy tone-interaction coefficient across horizons, corrected specification.

Same design as run_corrected_pooled_model.py. Reports the HAC estimate, the
moving-block bootstrap interval (block length = max(h, 25)), and the Wald test
of equality against the pooled other-topic interaction, for both index
constructions. Cleaned-index keys are bare "h<N>" for backward compatibility.

Run from the repository root. Writes corrected_tone_interaction.json.
"""
# --- replication-package prelude (added by build_package.py) ------------------
# The original research scripts pointed at an absolute project root. In this
# package the repository locates itself, so it runs from any checkout location.
from pathlib import Path as _RepoPath
_REPO = str(_RepoPath(__file__).resolve().parents[3])
# ------------------------------------------------------------------------------
import pandas as pd, numpy as np, statsmodels.api as sm, json
m=pd.read_csv("analysis/08_rebuild/weekly_series_master.csv", parse_dates=["Week"])
ps=pd.read_csv("data/07_weekly_series/weekly_topic_salience_by_category.csv"); ps["Week"]=pd.to_datetime(ps["0"])
pt=pd.read_csv("data/07_weekly_series/weekly_topic_sentiment_by_category.csv"); pt["Week"]=pd.to_datetime(pt["0"])
CATS=["art","disaster","economy","education","energy","medical","nature","politics","pollution","religion","society","technology"]
df=m.merge(ps[["Week"]+CATS].rename(columns={c:f"sal_{c}" for c in CATS}),on="Week")
df=df.merge(pt[["Week"]+CATS].rename(columns={c:f"ton_{c}" for c in CATS}),on="Week")
for c in [c for c in df.columns if c.startswith(("sal_","ton_"))]:
    df[c]=(df[c]-df[c].mean())/df[c].std()
COLS=["G1","G2"]+[f"sal_{c}" for c in CATS]+["en_ix","oth_ix","s1","c1","s2","c2","covid","ukr"]
def build(G,h):
    d=df.copy(); d["y"]=d[G].shift(-h); d["G1"]=d[G]; d["G2"]=d[G].shift(1)
    woy=d["Week"].dt.isocalendar().week.astype(float)
    d["s1"],d["c1"]=np.sin(2*np.pi*woy/52),np.cos(2*np.pi*woy/52)
    d["s2"],d["c2"]=np.sin(4*np.pi*woy/52),np.cos(4*np.pi*woy/52)
    d["covid"]=((d.Week>="2020-03-08")&(d.Week<="2021-12-26")).astype(float)
    d["ukr"]=(d.Week>="2022-02-20").astype(float)
    d["en_ix"]=d["sal_energy"]*d["ton_energy"]
    d["oth_ix"]=np.mean([d[f"sal_{c}"]*d[f"ton_{c}"] for c in CATS if c!="energy"],axis=0)
    return d.dropna(subset=["y","G2"]).reset_index(drop=True)
res={}
for G in ["idx_cleaned_weighted","idx_orig34_weighted"]:
  for h in [12,16,20,25,26]:
    d=build(G,h); n=len(d)
    r=sm.OLS(d["y"],sm.add_constant(d[COLS])).fit(cov_type="HAC",cov_kwds={"maxlags":max(h,25)})
    rng=np.random.default_rng(42); bl=max(h,25); e=[];o=[];diff=[]
    for _ in range(1000):
        idx=np.concatenate([np.arange(s,s+bl)%n for s in rng.integers(0,n,n//bl+1)])[:n]
        dd=d.iloc[idx]
        try:
            rr=sm.OLS(dd["y"],sm.add_constant(dd[COLS])).fit()
            e.append(rr.params["en_ix"]); o.append(rr.params["oth_ix"]); diff.append(rr.params["en_ix"]-rr.params["oth_ix"])
        except Exception: pass
    # HAC Wald for en_ix == oth_ix
    R=np.zeros(len(r.params)); R[list(r.params.index).index("en_ix")]=1; R[list(r.params.index).index("oth_ix")]=-1
    w=r.f_test(R.reshape(1,-1))
    res[f"{G}|h{h}" if G!="idx_cleaned_weighted" else f"h{h}"]={"en_ix_b":round(r.params["en_ix"],3),"en_ix_hacp":round(r.pvalues["en_ix"],4),
        "en_ix_bootCI":[round(float(np.percentile(e,2.5)),3),round(float(np.percentile(e,97.5)),3)],
        "oth_ix_b":round(r.params["oth_ix"],3),
        "diff_bootCI":[round(float(np.percentile(diff,2.5)),3),round(float(np.percentile(diff,97.5)),3)],
        "wald_p_en_eq_oth":round(float(w.pvalue),4)}
print(json.dumps(res,indent=1))
json.dump(res,open("analysis/09_stats/pooled_model/corrected_tone_interaction.json","w"),indent=1)
