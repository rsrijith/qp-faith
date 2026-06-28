#!/usr/bin/env python
"""C2 / construct circularity: the probe's relevant set and the BM25 retriever both use
the lexical name signal, so a recall drop is partly built in. Decouple the relevance
signal: define each motif's relevant set by DENSE semantic similarity (bge embeddings),
independent of lexical name co-occurrence, then measure how much of THAT set the emitted
category filter excludes. If the dense-relevant set also spans the wrong categories, the
harm is not a lexical-construction artifact; if a semantic model concentrates relevance
on the guessed category, the 'correct refinement' reading is supported."""
import os, re, json
import numpy as np, pandas as pd
from sentence_transformers import SentenceTransformer
DATA="data"; RES="results"
def norm(s): return re.sub(r"[^a-z0-9 ]"," ",str(s).lower()).strip()
def depl(w): return w[:-1] if len(w)>3 and w.endswith("s") else w
def tokset(s): return {depl(w) for w in norm(s).split() if w}
def matches(a,b): A,B=tokset(a),tokset(b); return bool(A) and bool(B) and (A==B or A<=B or B<=A)
p=pd.read_csv(f"{DATA}/product.csv",sep="\t")
cls={int(r.product_id):norm(r.product_class) for r in p.itertuples() if pd.notna(r.product_class)}
pid=np.array([int(x) for x in p["product_id"]])
emb=np.load(f"{RES}/dense_catalog_bge.npy"); st=SentenceTransformer("BAAI/bge-small-en-v1.5")
# probe relevant-set sizes (for comparable N)
probe={json.loads(l)["query"]:json.loads(l) for l in open(f"{DATA}/probe/probe.jsonl") if json.loads(l)["family"]=="A_entity"}
MODELS=["qwen4","llama4","mistral4","gemma4","haiku","sonnet","together-llama70","gpt4o","gpt52"]
# dense-relevant set per motif: top-N products by cosine to the motif word (N = lexical relevant-set size)
denserel={}
for q,r in probe.items():
    N=len(r["relevant_pids"]); sims=emb@st.encode([q],normalize_embeddings=True)[0]
    denserel[q]=set(pid[np.argsort(-sims)[:N]].tolist())
# also: how many distinct classes does the DENSE-relevant set span (vs lexical median 8)?
dspan=[len(set(cls.get(x,"") for x in denserel[q])) for q in denserel]
print(f"dense-relevant set spans median {int(np.median(dspan))} classes (lexical was 8)")
# exclusion of the dense-relevant set by the emitted category filter
byq={}
for tag in MODELS:
    for r in (json.loads(l) for l in open(f"{RES}/probe-{tag}-exp__probe_plans.jsonl")):
        if r["family"]!="A_entity" or "product_type" not in r["plan"]: continue
        q=r["query"]; pt=r["plan"]["product_type"]; dr=denserel[q]
        excl=sum(1 for x in dr if not matches(pt,cls.get(x,"")))/len(dr) if dr else 0.0
        byq.setdefault(q,[]).append(excl)
qmean=np.array([np.mean(byq[q]) for q in byq]); rng=np.random.default_rng(13)
boot=[qmean[rng.integers(0,len(qmean),len(qmean))].mean() for _ in range(5000)]
print(f"DENSE-decoupled exclusion of the semantic relevant set by the emitted category: {qmean.mean():.2f} [cluster CI {np.percentile(boot,2.5):.2f}, {np.percentile(boot,97.5):.2f}]")
print(f"  (compare: lexical by-construction exclusion 0.90-0.97; realized BM25 recall loss 0.74)")
json.dump({"dense_exclusion_mean":float(qmean.mean()),"dense_exclusion_ci":[float(np.percentile(boot,2.5)),float(np.percentile(boot,97.5))],"dense_relset_class_span_median":int(np.median(dspan))},open(f"{RES}/decoupled_relevance.json","w"),indent=2)
