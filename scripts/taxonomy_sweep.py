#!/usr/bin/env python
"""INT-3: regenerable taxonomy-filter tau-sweep (was previously hand-noted).
Pooled recall@100 loss when keeping products whose category is within cosine tau of the
emitted type. exact=0.74, no-filter=0. Writes results/taxonomy_tau_sweep.json."""
import os, re, json
import numpy as np, pandas as pd
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
DATA="data"; RES="results"
STOP=set("a an the of for with and or in on to set sets piece pieces that have by your our".split())
def norm(s): return re.sub(r"[^a-z0-9 ]"," ",str(s).lower()).strip()
def depl(w): return w[:-1] if len(w)>3 and w.endswith("s") else w
def tokset(s): return {depl(w) for w in norm(s).split() if w and w not in STOP}
def recall(o,rel,k=100): rs=set(rel); return len([x for x in o[:k] if x in rs])/len(rs) if rs else float("nan")
p=pd.read_csv(f"{DATA}/product.csv",sep="\t")
cls={int(r.product_id):norm(r.product_class) for r in p.itertuples() if pd.notna(r.product_class)}
bm=BM25Okapi([(norm(n)+" "+norm(f)).split() for n,f in zip(p["product_name"],p["product_features"])])
pid=np.array([int(x) for x in p["product_id"]])
classes=sorted(set(cls.values())); st=SentenceTransformer("BAAI/bge-small-en-v1.5")
cemb=st.encode(classes,normalize_embeddings=True); cidx={c:i for i,c in enumerate(classes)}
rows=[]; oc={}; tc={}
for tag in ["qwen4","llama4","mistral4","gemma4","haiku","sonnet","together-llama70","gpt4o","gpt52"]:
    for r in (json.loads(l) for l in open(f"{RES}/probe-{tag}-exp__probe_plans.jsonl")):
        if r["family"]=="A_entity" and "product_type" in r["plan"]:
            rows.append((r["query"],r["plan"]["product_type"],[int(x) for x in r["relevant_pids"]]))
def od(q):
    if q not in oc: oc[q]=pid[np.argsort(-bm.get_scores(list(tokset(q)) or ["x"]))].tolist()
    return oc[q]
out={}
for tau in [0.5,0.6,0.7]:
    ds=[]
    for q,pt,rel in rows:
        if pt not in tc: tc[pt]=st.encode([pt],normalize_embeddings=True)[0]
        near=set(np.where(cemb@tc[pt]>=tau)[0]); o=od(q)
        ds.append(recall(o,rel)-recall([x for x in o if cidx.get(cls.get(x,''),-2) in near],rel))
    out[f"tau_{tau}"]=round(float(np.mean(ds)),2); print(f"  tau={tau}: {np.mean(ds):.2f}")
out["note"]="taxonomy filter pooled recall loss vs threshold; exact=0.74, no-filter=0"
json.dump(out,open(f"{RES}/taxonomy_tau_sweep.json","w"),indent=2)
