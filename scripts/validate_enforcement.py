#!/usr/bin/env python
"""Enforcement-strength sweep, persisted (fixes the prior non-artifact-backed table).
On the entity probe, vary ONLY the enforcement weight lambda on the injected
product_type, holding queries and retriever fixed. CONSISTENT semantics across
retrievers: conflicting products' score x= lambda (lambda=0 => ranked last == hard
filter); NO silent fallback. Runs BM25 and dense (bge-small). Persists per-(tag,retriever)
curve to results/<tag>__enforce_<retr>.json. This is harm CALIBRATION, not a behavioral
finding: the entity-probe relevant set is non-matching by construction, so the curve
shows the cost of ENFORCING a (by-design wrong) category constraint at strength lambda."""
import os, re, json, argparse
import numpy as np, pandas as pd

RES = os.path.join(os.path.dirname(__file__), "..", "results")
DATA = os.path.join(os.path.dirname(__file__), "..", "data")
EMB = f"{RES}/dense_catalog_bge.npy"
LAMS = [1.0, 0.9, 0.7, 0.5, 0.3, 0.0]
def norm(s): return re.sub(r"[^a-z0-9 ]", " ", str(s).lower()).strip()
def depl(w): return w[:-1] if len(w) > 3 and w.endswith("s") else w
def tokset(s): return {depl(w) for w in norm(s).split() if w}
def matches(val, v): A=tokset(val); B=tokset(v); return bool(A) and bool(B) and (A==B or A<=B or B<=A)
def recall(order, rel, k=100): rs=set(rel); return len([x for x in order[:k] if x in rs])/len(rs) if rs else float("nan")
def boot(x):
    x=np.array(x,float); rng=np.random.default_rng(13); s=x[rng.integers(0,len(x),(2000,len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(s,2.5)), float(np.percentile(s,97.5))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--tags",nargs="+",required=True)
    ap.add_argument("--retrievers",nargs="+",default=["bm25","dense"]); a=ap.parse_args()
    p=pd.read_csv(f"{DATA}/product.csv",sep="\t")
    pid_arr=np.array([int(x) for x in p["product_id"]])
    cls={int(r.product_id):norm(r.product_class) for r in p.itertuples() if pd.notna(r.product_class)}
    bm=emb=st=None
    if "bm25" in a.retrievers:
        from rank_bm25 import BM25Okapi
        bm=BM25Okapi([(norm(n)+" "+norm(f)).split() for n,f in zip(p["product_name"],p["product_features"])])
    if "dense" in a.retrievers:
        from sentence_transformers import SentenceTransformer
        emb=np.load(EMB); st=SentenceTransformer("BAAI/bge-small-en-v1.5")
    for tag in a.tags:
        rows=[r for r in (json.loads(l) for l in open(f"{RES}/{tag}__probe_plans.jsonl"))
              if r["family"]=="A_entity" and "product_type" in r["plan"]]
        for retr in a.retrievers:
            curve={l:[] for l in LAMS}
            for r in rows:
                pt=r["plan"]["product_type"]; rel=[int(x) for x in r["relevant_pids"]]
                sc=np.array(bm.get_scores(list(tokset(r["query"])) or ["x"])) if retr=="bm25" else emb@st.encode([r["query"]],normalize_embeddings=True)[0]
                conf=np.array([0.0 if matches(pt,cls.get(int(pid_arr[i]),"")) else 1.0 for i in range(len(pid_arr))])
                base=recall(pid_arr[np.argsort(-sc)].tolist(),rel)
                for l in LAMS:
                    pen=np.where(conf>0,l,1.0)
                    curve[l].append(base-recall(pid_arr[np.argsort(-(sc*pen))].tolist(),rel))
            rep={"tag":tag,"retriever":retr,"n":len(rows),
                 "lambda_drop":{str(l):boot(curve[l]) for l in LAMS}}
            json.dump(rep,open(f"{RES}/{tag}__enforce_{retr}.json","w"),indent=2)
            print(f"{tag:18s} {retr:5s} n={len(rows):2d} " + " ".join(f"{boot(curve[l])[0]:.2f}" for l in LAMS))
    print("cols lambda:", LAMS, "(1.0 no penalty -> 0.0 hard filter)")

if __name__=="__main__":
    main()
