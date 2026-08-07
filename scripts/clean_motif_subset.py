#!/usr/bin/env python
"""Blind-panel-4 item 7 (construct validity of "labels known by construction").

The 55-motif entity probe contains a few entries that are not cleanly type-less
motifs (adjectives, a bed size, a literal product type, ambiguous plant/animal
parts). We (a) document the WordNet animal/plant/food filter's precision on the
55, and (b) recompute the headline realized recall@100 loss on a curated
clean-motif subset to show the magnitude is not driven by the noise entries.

Reuses the realized-loss computation of headline_stats_v2.py (BM25 + stored plans;
no model inference). Run: ./.venv/bin/python scripts/clean_motif_subset.py
"""
import os, re, json
import numpy as np, pandas as pd
from rank_bm25 import BM25Okapi
DATA="data"; RES="results"
STOP=set("a an the of for with and or in on to set sets piece pieces that have by your our".split())
def norm(s): return re.sub(r"[^a-z0-9 ]"," ",str(s).lower()).strip()
def depl(w): return w[:-1] if len(w)>3 and w.endswith("s") else w
def tokset(s): return {depl(w) for w in norm(s).split() if w and w not in STOP}
def matches(a,b): A,B=tokset(a),tokset(b); return bool(A) and bool(B) and (A==B or A<=B or B<=A)
def recall(order,rel,k=100): rs=set(rel); return len([x for x in order[:k] if x in rs])/len(rs) if rs else float("nan")

# noise motifs: not cleanly type-less decorative motifs (panel-named + literal product-type/structure)
NOISE={"queen","simple","giant","blade","wing","root","trunk","stump","climber","ornamental","sticker","arbor"}
# the 13 impurities from the HUMAN adjudication (results/r1_1/adjudication_result.json wands_impurities);
# used to report the clean-subset harm on the human-adjudicated-clean-42 (R1.1 residual, T2f).
ADJ_IMPURE={"queen","simple","bulb","trunk","blade","feeder","sticker","flour","noodle","cereal","condiment","sauce","casserole"}

p=pd.read_csv(f"{DATA}/product.csv",sep="\t")
cls={int(r.product_id):norm(r.product_class) for r in p.itertuples() if pd.notna(r.product_class)}
bm=BM25Okapi([(norm(n)+" "+norm(f)).split() for n,f in zip(p["product_name"],p["product_features"])])
pid_arr=np.array([int(x) for x in p["product_id"]])
MODELS=["qwen4","llama4","mistral4","gemma4","haiku","sonnet","together-llama70","gpt4o","gpt52"]
_cache={}
def order_for(q):
    if q not in _cache: _cache[q]=pid_arr[np.argsort(-bm.get_scores(list(tokset(q)) or ["x"]))].tolist()
    return _cache[q]
def per_query_drops(cond, keep=None):
    byq={}
    for tag in MODELS:
        fn=f"{RES}/probe-{tag}-{cond}__probe_plans.jsonl"
        if not os.path.exists(fn): continue
        for r in (json.loads(l) for l in open(fn)):
            if r["family"]!="A_entity": continue
            q=r["query"]
            if keep is not None and q not in keep: continue
            rel=[int(x) for x in r["relevant_pids"]]; pt=r["plan"].get("product_type")
            if not pt: byq.setdefault(q,[]).append(0.0); continue
            order=order_for(q)
            byq.setdefault(q,[]).append(recall(order,rel)-recall([x for x in order if matches(pt,cls.get(x,""))],rel))
    return byq
def boot(byq,n=5000,seed=13):
    qs=list(byq); rng=np.random.default_rng(seed); qm=np.array([np.mean(byq[q]) for q in qs])
    b=[qm[rng.integers(0,len(qs),len(qs))].mean() for _ in range(n)]
    return round(float(qm.mean()),3),round(float(np.percentile(b,2.5)),3),round(float(np.percentile(b,97.5)),3),len(qs)

# all 55 motifs
allq=set()
for r in (json.loads(l) for l in open(f"{DATA}/probe/probe.jsonl")):
    if r["family"]=="A_entity": allq.add(r["query"])
clean=allq-NOISE

# (a) WordNet filter precision on the 55
import nltk
try: from nltk.corpus import wordnet as wn; wn.synsets("dog")
except Exception:
    nltk.download("wordnet"); nltk.download("omw-1.4"); from nltk.corpus import wordnet as wn
ROOTS={"animal":"animal.n.01","plant":"plant.n.02","food":"food.n.01","food2":"food.n.02","organism":"organism.n.01"}
roots={k:wn.synset(v) for k,v in ROOTS.items()}
def bio_or_food(w):
    w=depl(w)
    for s in wn.synsets(w, pos=wn.NOUN):
        paths=set(h for path in s.hypernym_paths() for h in path)
        if any(roots[k] in paths for k in roots): return True
    return False
wn_pass=[q for q in allq if bio_or_food(q)]
# precision = of WordNet-passed, how many are in the curated clean set (not noise)
wn_clean=[q for q in wn_pass if q not in NOISE]

out={"n_all":len(allq),"n_noise":len(allq&NOISE),"noise_motifs":sorted(allq&NOISE),
     "n_clean":len(clean),
     "wordnet_filter_pass":len(wn_pass),"wordnet_filter_precision":round(len(wn_clean)/len(wn_pass),3),
     "wordnet_kept_noise":sorted(set(wn_pass)&NOISE)}
adj_clean=allq-ADJ_IMPURE   # human-adjudicated-clean 42
out["n_adj_impure"]=len(allq&ADJ_IMPURE); out["n_adj_clean"]=len(adj_clean)
for cond,label in [("exp","expansion"),("con","conservative")]:
    m,lo,hi,nq=boot(per_query_drops(cond))                 # full 55 (sanity)
    cm,clo,chi,cnq=boot(per_query_drops(cond,keep=clean))  # WordNet-heuristic clean subset
    am,alo,ahi,anq=boot(per_query_drops(cond,keep=adj_clean))  # human-adjudicated-clean 42
    out[f"{label}_full55"]=[m,lo,hi,nq]
    out[f"{label}_clean"]=[cm,clo,chi,cnq]
    out[f"{label}_adjclean"]=[am,alo,ahi,anq]
json.dump(out,open(f"{RES}/clean_motif_subset.json","w"),indent=2)
print(f"55 motifs: {len(allq)-len(NOISE&allq)} clean, {len(allq&NOISE)} noise ({sorted(allq&NOISE)})")
print(f"WordNet animal/plant/food filter: keeps {len(wn_pass)}/55, precision {out['wordnet_filter_precision']} (kept noise: {out['wordnet_kept_noise']})")
for label in ["expansion","conservative"]:
    f5=out[f"{label}_full55"]; c=out[f"{label}_clean"]
    print(f"{label}: full-55 {f5[0]} [{f5[1]},{f5[2]}] (n={f5[3]})  ->  clean {c[0]} [{c[1]},{c[2]}] (n={c[3]})")
print("wrote", f"{RES}/clean_motif_subset.json")
