#!/usr/bin/env python
"""Fix B-stat (cluster bootstrap by QUERY, not pseudo-replicated 9x55) and produce B1
(realized recall@100 loss under the CONSERVATIVE prompt, alongside expansion). The 9
models are correlated replicates of the same 55 queries, so the resampling unit is the
query. Reports 0.74 +- ~0.10, and the conservative/imply/assist realized harm."""
import os, re, json
import numpy as np, pandas as pd
from rank_bm25 import BM25Okapi
from scipy.stats import wilcoxon
DATA="data"; RES="results"
STOP=set("a an the of for with and or in on to set sets piece pieces that have by your our".split())
def norm(s): return re.sub(r"[^a-z0-9 ]"," ",str(s).lower()).strip()
def depl(w): return w[:-1] if len(w)>3 and w.endswith("s") else w
def tokset(s): return {depl(w) for w in norm(s).split() if w and w not in STOP}
def matches(a,b): A,B=tokset(a),tokset(b); return bool(A) and bool(B) and (A==B or A<=B or B<=A)
def recall(order,rel,k=100): rs=set(rel); return len([x for x in order[:k] if x in rs])/len(rs) if rs else float("nan")
p=pd.read_csv(f"{DATA}/product.csv",sep="\t")
cls={int(r.product_id):norm(r.product_class) for r in p.itertuples() if pd.notna(r.product_class)}
bm=BM25Okapi([(norm(n)+" "+norm(f)).split() for n,f in zip(p["product_name"],p["product_features"])])
pid_arr=np.array([int(x) for x in p["product_id"]])
MODELS=["qwen4","llama4","mistral4","gemma4","haiku","sonnet","together-llama70","gpt4o","gpt52"]
_cache={}
def order_for(q):
    if q not in _cache:
        _cache[q]=pid_arr[np.argsort(-bm.get_scores(list(tokset(q)) or ["x"]))].tolist()
    return _cache[q]
def per_query_drops(cond):
    """{query: [drop per model]} over ALL entity queries (drop=0 if no type forced)."""
    byq={}
    for tag in MODELS:
        fn=f"{RES}/probe-{tag}-{cond}__probe_plans.jsonl"
        if not os.path.exists(fn): continue
        for r in (json.loads(l) for l in open(fn)):
            if r["family"]!="A_entity": continue
            q=r["query"]; rel=[int(x) for x in r["relevant_pids"]]; pt=r["plan"].get("product_type")
            if not pt: byq.setdefault(q,[]).append(0.0); continue
            order=order_for(q)
            byq.setdefault(q,[]).append(recall(order,rel)-recall([x for x in order if matches(pt,cls.get(x,""))],rel))
    return byq
def cluster_boot(byq, n=5000, seed=13):
    qs=list(byq); rng=np.random.default_rng(seed)
    qmean=np.array([np.mean(byq[q]) for q in qs])
    means=[qmean[rng.integers(0,len(qs),len(qs))].mean() for _ in range(n)]
    return float(qmean.mean()), float(np.percentile(means,2.5)), float(np.percentile(means,97.5)), len(qs), qmean
out={}
for cond,label in [("exp","EXPANSION (stressor)"),("enr","ENRICH"),("ast","ASSIST"),("imp","IMPLY"),("con","CONSERVATIVE (realistic)")]:
    byq=per_query_drops(cond)
    if not byq: print(f"{label}: no data"); continue
    m,lo,hi,nq,qmean=cluster_boot(byq)
    if (qmean>0).any():
        W,pval=wilcoxon(qmean, np.zeros_like(qmean), zero_method="zsplit")
        d=qmean.mean()/qmean.std(ddof=1) if qmean.std(ddof=1)>0 else float("nan")
    else: pval,d=1.0,0.0
    out[cond]={"mean":m,"cluster_ci":[lo,hi],"n_queries":nq,"wilcoxon_p":float(pval),"cohens_d":float(d)}
    print(f"{label:28s}: realized recall@100 loss = {m:.3f}  cluster-CI[{lo:.2f}, {hi:.2f}] (n={nq} q)  Wilcoxon p={pval:.1e}, d={d:.2f}")
json.dump(out,open(f"{RES}/headline_stats_byprompt.json","w"),indent=2)
print("\nHONEST HEADLINE: expansion 0.74 +/- ~0.10 (cluster bootstrap, n=55 queries); conservative is the realistic-baseline number above.")
