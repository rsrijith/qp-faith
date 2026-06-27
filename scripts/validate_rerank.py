#!/usr/bin/env python
"""Two-stage retrieve-then-rerank validation (ESWA R3 ask): show that a learned
cross-encoder reranker does NOT recover items a hard category pre-filter removed,
because the filter is upstream of ranking. For each entity-probe query:
  (1) no-filter: BM25 top-N -> cross-encoder rerank -> recall@100
  (2) hard-filter: BM25 top-N -> keep only emitted-category products -> rerank -> recall@100
The (1)->(2) recall drop is the pre-filter harm; reranking cannot undo it.
Usage: python scripts/validate_rerank.py --tag probe-qwen4-exp"""
import os, re, json, argparse
import numpy as np, pandas as pd
from rank_bm25 import BM25Okapi
RES=os.path.join(os.path.dirname(__file__),"..","results"); DATA=os.path.join(os.path.dirname(__file__),"..","data")
def norm(s): return re.sub(r"[^a-z0-9 ]"," ",str(s).lower()).strip()
def depl(w): return w[:-1] if len(w)>3 and w.endswith("s") else w
def tokset(s): return {depl(w) for w in norm(s).split() if w}
def matches(a,b): A,B=tokset(a),tokset(b); return bool(A) and bool(B) and (A==B or A<=B or B<=A)
def recall(order,rel,k=100): rs=set(rel); return len([x for x in order[:k] if x in rs])/len(rs) if rs else float('nan')

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--tag",required=True); ap.add_argument("--N",type=int,default=1000)
    a=ap.parse_args()
    p=pd.read_csv(f"{DATA}/product.csv",sep="\t")
    pid=np.array([int(x) for x in p["product_id"]])
    cls={int(r.product_id):norm(r.product_class) for r in p.itertuples() if pd.notna(r.product_class)}
    texts=[(norm(n)+" "+norm(f)) for n,f in zip(p["product_name"],p["product_features"])]
    bm=BM25Okapi([t.split() for t in texts])
    from sentence_transformers import CrossEncoder
    import torch
    dev="mps" if torch.backends.mps.is_available() else "cpu"
    try: ce=CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=dev)
    except Exception: ce=CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    rows=[r for r in (json.loads(l) for l in open(f"{RES}/{a.tag}__probe_plans.jsonl"))
          if r["family"]=="A_entity" and "product_type" in r["plan"]]
    base,hard=[],[]
    for r in rows:
        q=r["query"]; ptv=r["plan"]["product_type"]; rel=[int(x) for x in r["relevant_pids"]]
        sc=np.array(bm.get_scores(list(tokset(q)) or ["x"]))
        top=pid[np.argsort(-sc)[:a.N]].tolist()
        # cross-encoder rerank of the BM25 top-N
        idx={int(pid[i]):i for i in range(len(pid))}
        pairs=[(q,texts[idx[t]]) for t in top]
        ce_sc=ce.predict(pairs,batch_size=128,show_progress_bar=False)
        order=[t for _,t in sorted(zip(ce_sc,top),key=lambda x:-x[0])]
        base.append(recall(order,rel))
        kept=[t for t in order if matches(ptv,cls.get(t,""))]
        hard.append(recall(kept if kept else order,rel))
    res=dict(tag=a.tag,n=len(rows),
             rerank_norfilter_recall=float(np.mean(base)),
             rerank_hardfilter_recall=float(np.mean(hard)),
             rerank_harm=float(np.mean([b-h for b,h in zip(base,hard)])))
    json.dump(res,open(f"{RES}/{a.tag}__rerank.json","w"),indent=1)
    print(json.dumps(res,indent=1))

if __name__=="__main__": main()
