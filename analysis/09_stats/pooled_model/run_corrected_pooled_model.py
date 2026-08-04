"""Corrected pooled dynamic topic model (Eq. 2) at all reported horizons.

This is the specification behind Table 4 and the RQ2 results in the main text:
12 standardized topic saliences, the energy and pooled-other tone interactions,
two week-of-year harmonic pairs, the pandemic indicator, the post-2022-02-20
regime indicator, and the lagged index, with Newey--West standard errors whose
truncation lag is at least the forecast horizon.

Run from the repository root. Writes corrected_final_spec.json.
"""
# --- replication-package prelude (added by build_package.py) ------------------
# The original research scripts pointed at an absolute project root. In this
# package the repository locates itself, so it runs from any checkout location.
from pathlib import Path as _RepoPath
_REPO = str(_RepoPath(__file__).resolve().parents[3])
# ------------------------------------------------------------------------------
import pandas as pd, numpy as np, statsmodels.api as sm, json
gt=pd.read_csv("data/07_weekly_series/gt_weekly_36tokens_2018_2022.csv"); gt["Week"]=pd.to_datetime(gt["Week"])
m=pd.read_csv("analysis/08_rebuild/weekly_series_master.csv", parse_dates=["Week"])
ps=pd.read_csv("data/07_weekly_series/weekly_topic_salience_by_category.csv"); ps["Week"]=pd.to_datetime(ps["0"])
pt=pd.read_csv("data/07_weekly_series/weekly_topic_sentiment_by_category.csv"); pt["Week"]=pd.to_datetime(pt["0"])
CATS=["art","disaster","economy","education","energy","medical","nature","politics","pollution","religion","society","technology"]
df=m.merge(ps[["Week"]+CATS].rename(columns={c:f"sal_{c}" for c in CATS}),on="Week")
df=df.merge(pt[["Week"]+CATS].rename(columns={c:f"ton_{c}" for c in CATS}),on="Week")
for c in [c for c in df.columns if c.startswith(("sal_","ton_"))]:
    df[c]=(df[c]-df[c].mean())/df[c].std()

def build(G,h):
    d=df.copy()
    d["y"]=d[G].shift(-h); d["G1"]=d[G]; d["G2"]=d[G].shift(1)
    woy=d["Week"].dt.isocalendar().week.astype(float)
    d["s1"],d["c1"]=np.sin(2*np.pi*woy/52),np.cos(2*np.pi*woy/52)
    d["s2"],d["c2"]=np.sin(4*np.pi*woy/52),np.cos(4*np.pi*woy/52)
    d["covid"]=((d.Week>="2020-03-08")&(d.Week<="2021-12-26")).astype(float)
    d["ukr"]=(d.Week>="2022-02-20").astype(float)
    d["en_ix"]=d["sal_energy"]*d["ton_energy"]
    d["oth_ix"]=np.mean([d[f"sal_{c}"]*d[f"ton_{c}"] for c in CATS if c!="energy"],axis=0)
    return d.dropna(subset=["y","G2"]).reset_index(drop=True)

COLS=["G1","G2"]+[f"sal_{c}" for c in CATS]+["en_ix","oth_ix","s1","c1","s2","c2","covid","ukr"]
out={}
for G in ["idx_cleaned_weighted","idx_orig34_weighted"]:
    for h in [12,16,20,25,26]:
        d=build(G,h)
        r=sm.OLS(d["y"],sm.add_constant(d[COLS])).fit(cov_type="HAC",cov_kwds={"maxlags":max(h,25)})
        key=f"{G}|h{h}"
        sal={c: r.params[f"sal_{c}"] for c in CATS}
        rank1=max(sal,key=lambda k: abs(sal[k]))
        out[key]={"n":int(r.nobs),"adjR2":round(r.rsquared_adj,4),
                  "energy_b":round(r.params["sal_energy"],3),"energy_p":round(r.pvalues["sal_energy"],4),
                  "energy_se":round(r.bse["sal_energy"],3),"rank1":rank1,
                  "ukr_b":round(r.params["ukr"],2),"ukr_p":round(r.pvalues["ukr"],5)}
        if h==25:
            out[key]["all_topics"]={c:[round(r.params[f"sal_{c}"],3),round(r.bse[f"sal_{c}"],3),round(r.pvalues[f"sal_{c}"],4)] for c in CATS}
            out[key]["en_ix"]=[round(r.params["en_ix"],3),round(r.bse["en_ix"],3),round(r.pvalues["en_ix"],4)]
            out[key]["oth_ix"]=[round(r.params["oth_ix"],3),round(r.bse["oth_ix"],3),round(r.pvalues["oth_ix"],4)]
            out[key]["G1"]=[round(r.params["G1"],3),round(r.bse["G1"],3),round(r.pvalues["G1"],4)]
            out[key]["G2"]=[round(r.params["G2"],3),round(r.bse["G2"],3),round(r.pvalues["G2"],4)]
            out[key]["covid"]=[round(r.params["covid"],3),round(r.bse["covid"],3),round(r.pvalues["covid"],4)]

# bootstrap rank stability + energy CI at h=25, both indices under the corrected spec
def bootstrap_h25(G):
    d=build(G,25); n=len(d); rng=np.random.default_rng(42); bl=25
    bs=[]; rank1s=0
    for _ in range(1000):
        idx=np.concatenate([np.arange(s,s+bl)%n for s in rng.integers(0,n,n//bl+1)])[:n]
        dd=d.iloc[idx]
        try:
            rr=sm.OLS(dd["y"],sm.add_constant(dd[COLS])).fit()
            bs.append(rr.params["sal_energy"])
            sal={c:abs(rr.params[f"sal_{c}"]) for c in CATS}
            if max(sal,key=sal.get)=="energy": rank1s+=1
        except Exception: pass
    return {"energy_CI":[round(float(np.percentile(bs,2.5)),3),round(float(np.percentile(bs,97.5)),3)],
            "rank1_share":round(rank1s/len(bs),3),"n_boot":len(bs)}

out["bootstrap_h25_cleaned"]=bootstrap_h25("idx_cleaned_weighted")
out["bootstrap_h25_orig34"]=bootstrap_h25("idx_orig34_weighted")
print(json.dumps(out,indent=1))
json.dump(out,open("analysis/09_stats/pooled_model/corrected_final_spec.json","w"),indent=1)
