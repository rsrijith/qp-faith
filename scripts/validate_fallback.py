#!/usr/bin/env python
"""Null/low-result fallback test, persisted (replaces unsourced manuscript numbers).
On the entity probe, apply the emitted product_type as a hard category filter, BUT if
fewer than T catalog products carry that category, fall back to no-filter (the standard
low-result guardrail). Same BM25 retrieval as Table 2 (validate_retrieval). Report the
hard-filter recall@100 loss, the loss WITH fallback at T in {10,50}, and the relative
recovery. Persists results/<tag>__fallback.json. CPU only."""
import os, re, json, argparse, collections
import numpy as np, pandas as pd
from rank_bm25 import BM25Okapi
DATA = os.path.join(os.path.dirname(__file__), "..", "data")
RES = os.path.join(os.path.dirname(__file__), "..", "results")
STOP = set("a an the of for with and or in on to set sets piece pieces that have by your our".split())
def norm(s): return re.sub(r"[^a-z0-9 ]", " ", str(s).lower()).strip()
def depl(w): return w[:-1] if len(w) > 3 and w.endswith("s") else w
def tokset(s): return {depl(w) for w in norm(s).split() if w and w not in STOP}
def matches(a, b):
    A, B = tokset(a), tokset(b); return bool(A) and bool(B) and (A == B or A <= B or B <= A)
def recall(order, rel, k=100): rs=set(rel); return len([p for p in order[:k] if p in rs])/len(rs) if rs else float("nan")
def boot(x,n=2000,seed=13):
    x=np.array([v for v in x if v==v],float); rng=np.random.default_rng(seed)
    return float(x.mean()), *(float(np.percentile(x[rng.integers(0,len(x),(n,len(x)))].mean(1),p)) for p in (2.5,97.5))

_C={}
def index():
    if _C: return _C["bm"],_C["pid"],_C["cls"],_C["clscount"]
    p=pd.read_csv(f"{DATA}/product.csv",sep="\t")
    cls={int(r.product_id):norm(r.product_class) for r in p.itertuples() if pd.notna(r.product_class)}
    clscount=collections.Counter(cls.values())
    bm=BM25Okapi([(norm(n)+" "+norm(f)).split() for n,f in zip(p["product_name"],p["product_features"])])
    _C.update(bm=bm,pid=np.array([int(x) for x in p["product_id"]]),cls=cls,clscount=clscount)
    return _C["bm"],_C["pid"],_C["cls"],_C["clscount"]

def cat_count(pt, clscount):
    # how many catalog products carry a category matching the emitted type
    return sum(c for k,c in clscount.items() if matches(pt,k))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--tags",nargs="+",required=True); ap.add_argument("--Ts",nargs="+",type=int,default=[10,50]); a=ap.parse_args()
    bm,pid_arr,cls,clscount=index()
    for tag in a.tags:
        rows=[r for r in (json.loads(l) for l in open(f"{RES}/{tag}__probe_plans.jsonl")) if r["family"]=="A_entity" and "product_type" in r["plan"]]
        hard=[]; fb={T:[] for T in a.Ts}
        for r in rows:
            pt=r["plan"]["product_type"]; rel=[int(x) for x in r["relevant_pids"]]
            order=pid_arr[np.argsort(-bm.get_scores(list(tokset(r["query"])) or ["x"]))].tolist()
            base=recall(order,rel)
            keep=[p for p in order if matches(pt,cls.get(p,""))]
            hard.append(base-recall(keep,rel))
            ccount=cat_count(pt,clscount)
            for T in a.Ts:
                fb[T].append(base-recall(order if ccount<T else keep, rel))  # fallback fires if category too small
        rep=dict(tag=tag,n=len(rows),hard_filter_loss=boot(hard),
                 fallback_loss={str(T):boot(fb[T]) for T in a.Ts},
                 recovery_frac={str(T): (boot(hard)[0]-boot(fb[T])[0])/boot(hard)[0] if boot(hard)[0] else 0.0 for T in a.Ts})
        json.dump(rep,open(f"{RES}/{tag}__fallback.json","w"),indent=2)
        hl=rep["hard_filter_loss"][0]
        print(f"{tag:18s} hard {hl:.2f}  " + "  ".join(f"T={T} {rep['fallback_loss'][str(T)][0]:.2f} (recover {rep['recovery_frac'][str(T)]*100:.0f}%)" for T in a.Ts))

if __name__=="__main__": main()
