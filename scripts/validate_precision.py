#!/usr/bin/env python
"""Head-of-list harm: precision@10 and nDCG@10 on the ENTITY segment, no-filter vs
hard-filter, for all nine main models. Answers reviewer C2 (does the hard category
filter trade recall for head-of-list precision? It does not, on entity queries).
Persists results/entity_precision.json with per-model values and the band."""
import os, json, re, math
import numpy as np, pandas as pd
from rank_bm25 import BM25Okapi
DATA=os.path.join(os.path.dirname(__file__),"..","data"); RES=os.path.join(os.path.dirname(__file__),"..","results")
STOP=set("a an the of for with and or in on to set sets piece pieces that have by your our".split())
def norm(s): return re.sub(r"[^a-z0-9 ]"," ",str(s).lower()).strip()
def depl(w): return w[:-1] if len(w)>3 and w.endswith("s") else w
def tokset(s): return {depl(w) for w in norm(s).split() if w and w not in STOP}
def matches(a,b): A,B=tokset(a),tokset(b); return bool(A) and bool(B) and (A==B or A<=B or B<=A)
def prec_at(order,rel,k=10): rs=set(rel); return sum(1 for x in order[:k] if x in rs)/k
def ndcg_at(order,rel,k=10):
    rs=set(rel); dcg=sum(1/math.log2(i+2) for i,x in enumerate(order[:k]) if x in rs)
    idcg=sum(1/math.log2(i+2) for i in range(min(k,len(rs)))); return dcg/idcg if idcg else 0.0
def boot(x,n=2000,seed=13):
    x=np.array(x,float); rng=np.random.default_rng(seed); s=x[rng.integers(0,len(x),(n,len(x)))].mean(1)
    return float(x.mean()),float(np.percentile(s,2.5)),float(np.percentile(s,97.5))
MAIN=[("Qwen2.5-7B","qwen4"),("Llama-3.1-8B","llama4"),("Mistral-7B","mistral4"),("Gemma-2-9B","gemma4"),
      ("Haiku 4.5","haiku"),("Sonnet 4.6","sonnet"),("Llama-3.3-70B","together-llama70"),("gpt-4o","gpt4o"),("gpt-5.2","gpt52")]
def main():
    p=pd.read_csv(f"{DATA}/product.csv",sep="\t")
    cls={int(r.product_id):norm(r.product_class) for r in p.itertuples() if pd.notna(r.product_class)}
    bm=BM25Okapi([(norm(n)+" "+norm(f)).split() for n,f in zip(p["product_name"],p["product_features"])])
    pid_arr=np.array([int(x) for x in p["product_id"]])
    per_model={}; dP=[]; dN=[]
    for name,t in MAIN:
        rows=[r for r in (json.loads(l) for l in open(f"{RES}/probe-{t}-exp__probe_plans.jsonl")) if r["family"]=="A_entity" and "product_type" in r["plan"]]
        P_no,P_f,N_no,N_f=[],[],[],[]
        for r in rows:
            pt=r["plan"]["product_type"]; rel=[int(x) for x in r["relevant_pids"]]
            order=pid_arr[np.argsort(-bm.get_scores(list(tokset(r["query"])) or ["x"]))].tolist()
            keep=[x for x in order if matches(pt,cls.get(x,""))] or order
            P_no.append(prec_at(order,rel)); P_f.append(prec_at(keep,rel))
            N_no.append(ndcg_at(order,rel)); N_f.append(ndcg_at(keep,rel))
        pn,pf,nn,nf=boot(P_no)[0],boot(P_f)[0],boot(N_no)[0],boot(N_f)[0]
        per_model[name]=dict(n=len(rows),p10_nofilter=pn,p10_filter=pf,p10_delta=pf-pn,
                             ndcg10_nofilter=nn,ndcg10_filter=nf,ndcg10_delta=nf-nn)
        dP.append(pf-pn); dN.append(nf-nn)
        print(f"  {name:18s} P@10 {pn:.2f}->{pf:.2f} (d{pf-pn:+.2f})  nDCG@10 {nn:.2f}->{nf:.2f} (d{nf-nn:+.2f})")
    rep=dict(n_models=len(MAIN),per_model=per_model,
             p10_delta=[min(dP),max(dP)],ndcg10_delta=[min(dN),max(dN)])
    json.dump(rep,open(f"{RES}/entity_precision.json","w"),indent=2)
    print(f"\nP@10 delta band: {min(dP):+.2f} to {max(dP):+.2f};  nDCG@10 delta band: {min(dN):+.2f} to {max(dN):+.2f}  (all 9 models)")
if __name__=="__main__": main()
