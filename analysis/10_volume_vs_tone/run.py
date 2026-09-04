"""Affirmative 'volume, not tone' analysis for RQ3.
A. Nested block decomposition (in-sample adj R2 + HAC block Wald + expanding-window OOS).
B. Tone-split coverage series: positive- vs negative-toned climate coverage, equality tests.
C. Grouped SHAP share (salience block vs tone block) for XGB.
"""
# --- replication-package prelude (added by build_package.py) ------------------
# The original research scripts pointed at an absolute project root. In this
# package the repository locates itself, so it runs from any checkout location.
from pathlib import Path as _RepoPath
_REPO = str(_RepoPath(__file__).resolve().parents[2])
# ------------------------------------------------------------------------------
import numpy as np, pandas as pd, torch, time, json
import os
import statsmodels.api as sm

ROOT = _REPO
OUT = f"{ROOT}/analysis/10_volume_vs_tone"

# ---------- B1: score climate-clean articles ----------
prov = pd.read_parquet(f"{ROOT}/analysis/06_token_audit/matched_articles_with_text.parquet")
prov["date"] = pd.to_datetime(prov["date"].astype(str).str.slice(0,10), format="%Y-%m-%d", errors="coerce")
prov = prov.dropna(subset=["date"]).drop_duplicates(subset=["outlet","text"])
UMBRELLA=["geoengineering","geoengineer","climate engineering","climate engineer"]
SRM=["aerosol injection","stratospheric aerosol injection","albedo modification","high albedo","high-albedo","marine cloud brightening","ocean mirror","space shade","space sunshade"]
CDR=["air capture","carbon capture","carbon storage","carbon sequestration","biochar","enhanced weathering","ocean fertilization","ocean fertilisation","ocean iron fertilization","ocean iron fertilisation","methane removal","afforestation","reforestation","bioenergy","bio-energy"]
CLEANED=set(UMBRELLA+SRM+CDR)
KEEP=set(["biodiversity loss","deforestation","sea level rise","plastic pollution","ocean acidification","air pollution","global warming","overfishing","climate change"])
gs = prov.geo_tokens.apply(lambda c: bool(set(c.split("|")) & CLEANED) if c else False)
ck = prov.climate_tokens.apply(lambda c: bool(set(c.split("|")) & KEEP) if c else False)
cc = prov[ck & ~gs].copy()
print("climate-clean articles:", len(cc), flush=True)

TONES_CSV = f"{OUT}/climate_clean_tones.csv"
if os.path.exists(TONES_CSV):
    # Reuse the archived per-article tone scores (deterministic output of the
    # fine-tuned model); re-scoring requires the model checkpoint below.
    cached = pd.read_csv(TONES_CSV, parse_dates=["date"])
    cc = cc.merge(cached[["article_id", "tone"]], on="article_id", how="inner")
    print(f"reusing archived tone scores for {len(cc)} articles", flush=True)
else:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    DEV = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
    m = AutoModelForSequenceClassification.from_pretrained(f"{ROOT}/models/news_sentiment_bert_retrained_seed42").eval().to(DEV)
    scores, t0 = [], time.time()
    texts = cc["text"].fillna("").tolist()
    with torch.no_grad():
        for i in range(0, len(texts), 32):
            batch = tok(texts[i:i+32], padding=True, truncation=True, max_length=512, return_tensors="pt").to(DEV)
            logits = m(**batch).logits.float().cpu().numpy()
            p = np.exp(logits)/np.exp(logits).sum(1, keepdims=True)
            scores.extend((p*np.arange(1,6)).sum(1))
    cc["tone"] = scores
    print(f"scored in {time.time()-t0:.0f}s; tone mean {np.mean(scores):.2f}", flush=True)
    cc[["article_id","outlet","date","tone"]].to_csv(TONES_CSV, index=False)

# ---------- B2: tone-split weekly series ----------
master = pd.read_csv(f"{ROOT}/analysis/08_rebuild/weekly_series_master.csv", parse_dates=["Week"])
med = cc["tone"].median()
cc["wk"] = cc["date"].dt.to_period("W-SAT").dt.start_time
def pooled_z(sub):
    wk = sub.groupby(["wk","outlet"]).size().unstack(fill_value=0).reindex(master["Week"].values, fill_value=0)
    z = wk.apply(lambda col: (col-col.mean())/col.std() if col.std()>0 else col*0)
    return z.mean(1).values
master["m_pos"] = pooled_z(cc[cc.tone >= med])
master["m_neg"] = pooled_z(cc[cc.tone < med])
r_split = float(np.corrcoef(master.m_pos, master.m_neg)[0,1])
print(f"pos/neg series correlate r={r_split:.3f}", flush=True)

def block_boot_r(x, y, L, B=2000, bl=13, seed=42):
    x, y = np.asarray(x), np.asarray(y)
    xs, ys = x[:-L] if L>0 else x, y[L:] if L>0 else y
    n = len(xs); rng = np.random.default_rng(seed)
    robs = np.corrcoef(xs, ys)[0,1]
    rs = []
    for _ in range(B):
        idx = np.concatenate([np.arange(s, s+bl) % n for s in rng.integers(0, n, n//bl + 1)])[:n]
        rs.append(np.corrcoef(xs[idx], ys[idx])[0,1])
    return robs, np.percentile(rs, [2.5, 97.5]), np.array(rs)

res = {}
for G in ["idx_orig34_weighted", "idx_cleaned_weighted"]:
    rp, cip, dp = block_boot_r(master.m_pos, master[G], 26)
    rn, cin, dn = block_boot_r(master.m_neg, master[G], 26)
    # difference test: bootstrap the difference jointly
    x1, x2, y = master.m_pos.values[:-26], master.m_neg.values[:-26], master[G].values[26:]
    n = len(y); rng = np.random.default_rng(7); diffs=[]
    for _ in range(2000):
        idx = np.concatenate([np.arange(s, s+13) % n for s in rng.integers(0, n, n//13+1)])[:n]
        diffs.append(np.corrcoef(x1[idx], y[idx])[0,1] - np.corrcoef(x2[idx], y[idx])[0,1])
    dob = rp - rn
    res[G] = {"r_pos": round(rp,3), "ci_pos": [round(v,3) for v in cip],
              "r_neg": round(rn,3), "ci_neg": [round(v,3) for v in cin],
              "r_diff": round(dob,3), "diff_ci": [round(v,3) for v in np.percentile(diffs,[2.5,97.5])]}
print(json.dumps(res, indent=1), flush=True)

# ---------- A: nested block decomposition ----------
piv_s = pd.read_csv(f"{ROOT}/data/07_weekly_series/weekly_topic_salience_by_category.csv"); piv_s["Week"]=pd.to_datetime(piv_s["0"])
piv_t = pd.read_csv(f"{ROOT}/data/07_weekly_series/weekly_topic_sentiment_by_category.csv"); piv_t["Week"]=pd.to_datetime(piv_t["0"])
CATS=["art","disaster","economy","education","energy","medical","nature","politics","pollution","religion","society","technology"]
df = master.merge(piv_s[["Week"]+CATS].rename(columns={c:f"sal_{c}" for c in CATS}), on="Week")
df = df.merge(piv_t[["Week"]+CATS].rename(columns={c:f"ton_{c}" for c in CATS}), on="Week")
for c in df.columns:
    if c.startswith(("sal_","ton_")):
        df[c] = (df[c]-df[c].mean())/df[c].std()

def nested(G, h):
    d = df.copy()
    d["y"] = d[G].shift(-h); d["G1"]=d[G]; d["G2"]=d[G].shift(1)
    woy = d["Week"].dt.isocalendar().week.astype(float)
    d["s1"],d["c1"],d["s2"],d["c2"] = np.sin(2*np.pi*woy/52),np.cos(2*np.pi*woy/52),np.sin(4*np.pi*woy/52),np.cos(4*np.pi*woy/52)
    d["covid"]=((d.Week>="2020-03-08")&(d.Week<="2021-12-26")).astype(float)
    d["ukr"]=(d.Week>="2022-02-20").astype(float)
    d["en_ix"]=d["sal_energy"]*d["ton_energy"]
    d["oth_ix"]=np.mean([d[f"sal_{c}"]*d[f"ton_{c}"] for c in CATS if c!="energy"],axis=0)
    d = d.dropna(subset=["y","G2"])
    base=["G1","G2","s1","c1","s2","c2","covid","ukr"]
    sal=[f"sal_{c}" for c in CATS]; ton=[f"ton_{c}" for c in CATS]; ix=["en_ix","oth_ix"]
    out=[]
    prev_cols=None
    for name, cols in [("M0 baseline",base),("M1 +salience",base+sal),("M2 +tone mains",base+sal+ton),("M3 +interactions",base+sal+ton+ix)]:
        X=sm.add_constant(d[cols]); mfit=sm.OLS(d["y"],X).fit(cov_type="HAC",cov_kwds={"maxlags":max(h,25)})
        row={"model":name,"k":len(cols),"adjR2":round(mfit.rsquared_adj,4)}
        if prev_cols is not None:
            add=[c for c in cols if c not in prev_cols]
            R=np.zeros((len(add),len(mfit.params)))
            for i,c in enumerate(add): R[i,list(mfit.params.index).index(c)]=1
            row["block_F_p"]=round(float(mfit.f_test(R).pvalue),5)
        out.append(row); prev_cols=cols
    # OOS expanding-window ladder
    oos=[]
    n=len(d); folds=[(int(n*f), int(n*(f+0.1))) for f in [0.5,0.6,0.7,0.8,0.9]]
    for name, cols in [("M0 baseline",base),("M1 +salience",base+sal),("M2 +tone mains",base+sal+ton),("M3 +interactions",base+sal+ton+ix)]:
        errs=[]
        for a,b in folds:
            tr, te = d.iloc[:a], d.iloc[a:b]
            if len(te)==0: continue
            # The regime indicator is constant (zero) in the early expanding windows and is
            # therefore not identified there; drop any column with no training variation so
            # the fold is estimated on a full-rank design rather than a pseudo-inverse.
            fit_cols=[c for c in cols if tr[c].std() > 0]
            X=sm.add_constant(tr[fit_cols]); f=sm.OLS(tr["y"],X).fit()
            Xt=sm.add_constant(te[fit_cols], has_constant="add")
            errs.extend((te["y"]-f.predict(Xt))**2)
        oos.append({"model":name,"OOS_RMSE":round(float(np.sqrt(np.mean(errs))),3)})
    return pd.DataFrame(out), pd.DataFrame(oos), len(d)

for G in ["idx_orig34_weighted","idx_cleaned_weighted"]:
    for _h in (25, 26):
        tab, oos, n = nested(G, _h)
        tab.to_csv(f"{OUT}/nested_{G}_h{_h}.csv", index=False); oos.to_csv(f"{OUT}/nested_oos_{G}_h{_h}.csv", index=False)
        print(f"--- {G} (h={_h}, n={n})"); print(tab.to_string(index=False)); print(oos.to_string(index=False), flush=True)

json.dump(res, open(f"{OUT}/tone_split_results.json","w"), indent=1)
master[["Week","m_pos","m_neg"]].to_csv(f"{OUT}/tone_split_series.csv", index=False)
print("DONE")
