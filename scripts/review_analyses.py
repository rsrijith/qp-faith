#!/usr/bin/env python
"""Review round 4: (C3) pooled bootstrap CI for the headline magnitude; (D1) plant/food
vs decorative motif partition; (D2) taxonomy-aware (semantic) category filter vs the
exact head-noun filter. All on the n=55 WANDS entity probe, BM25 retrieval (Table 2)."""
import os, re, json
import numpy as np, pandas as pd
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
DATA=os.path.join(os.path.dirname(__file__),"..","data"); RES=os.path.join(os.path.dirname(__file__),"..","results")
STOP=set("a an the of for with and or in on to set sets piece pieces that have by your our".split())
def norm(s): return re.sub(r"[^a-z0-9 ]"," ",str(s).lower()).strip()
def depl(w): return w[:-1] if len(w)>3 and w.endswith("s") else w
def tokset(s): return {depl(w) for w in norm(s).split() if w and w not in STOP}
def matches(a,b): A,B=tokset(a),tokset(b); return bool(A) and bool(B) and (A==B or A<=B or B<=A)
def recall(order,rel,k=100): rs=set(rel); return len([x for x in order[:k] if x in rs])/len(rs) if rs else float("nan")
MODELS=["qwen4","llama4","mistral4","gemma4","haiku","sonnet","together-llama70","gpt4o","gpt52"]
CONTEST=set("sunflowers lily roses pineapples daisy lotus vine aloe holly succulent fruit foliage orchids flour noodle cereal condiment casserole sauce breakfast lunch dinner cuisine".split())
p=pd.read_csv(f"{DATA}/product.csv",sep="\t")
cls={int(r.product_id):norm(r.product_class) for r in p.itertuples() if pd.notna(r.product_class)}
bm=BM25Okapi([(norm(n)+" "+norm(f)).split() for n,f in zip(p["product_name"],p["product_features"])])
pid_arr=np.array([int(x) for x in p["product_id"]])
# embed distinct product_class once for the semantic taxonomy filter
classes=sorted(set(cls.values())); st=SentenceTransformer("BAAI/bge-small-en-v1.5")
cemb=st.encode(classes,normalize_embeddings=True); cidx={c:i for i,c in enumerate(classes)}
pid_class_i=np.array([cidx.get(cls.get(int(pid),""),-1) for pid in pid_arr])

exact_drops=[]; tax_drops=[]; contest_d=[]; noncontest_d=[]
typecache={}
for tag in MODELS:
    rows=[r for r in (json.loads(l) for l in open(f"{RES}/probe-{tag}-exp__probe_plans.jsonl")) if r["family"]=="A_entity" and "product_type" in r["plan"]]
    for r in rows:
        pt=r["plan"]["product_type"]; rel=[int(x) for x in r["relevant_pids"]]; q=r["query"].lower()
        order=pid_arr[np.argsort(-bm.get_scores(list(tokset(r["query"])) or ["x"]))].tolist()
        base=recall(order,rel)
        keep_exact=[x for x in order if matches(pt,cls.get(x,""))]  # no null-fallback (Table 2 semantics)
        d_exact=base-recall(keep_exact,rel); exact_drops.append(d_exact)
        (contest_d if q in CONTEST else noncontest_d).append(d_exact)
        # taxonomy-aware: keep products whose class is semantically near the emitted type (tau=0.6, ~parent/sibling)
        if pt not in typecache: typecache[pt]=st.encode([pt],normalize_embeddings=True)[0]
        sims=cemb@typecache[pt]; near=set(np.where(sims>=0.6)[0])
        keep_tax=[pid for pid in order if cidx.get(cls.get(pid,''),-2) in near]
        tax_drops.append(base-recall(keep_tax,rel))
def boot(x,n=5000,seed=13):
    x=np.array(x,float); rng=np.random.default_rng(seed); s=x[rng.integers(0,len(x),(n,len(x)))].mean(1)
    return float(x.mean()),float(np.percentile(s,2.5)),float(np.percentile(s,97.5))
pe=boot(exact_drops); pt_=boot(tax_drops); pc=boot(contest_d); pn=boot(noncontest_d)
rep={"pooled_exact_recall_loss":pe,"pooled_taxonomy06_recall_loss":pt_,
     "contestable_plantfood_n":len(contest_d)//9,"contestable_loss":pc,
     "decorative_n":len(noncontest_d)//9,"decorative_loss":pn,"n_per_model":55,"n_models":9}
json.dump(rep,open(f"{RES}/review_analyses.json","w"),indent=2)
print(f"(C3) POOLED exact recall@100 loss (9 models x 55q, n={len(exact_drops)}): {pe[0]:.3f} [{pe[1]:.3f}, {pe[2]:.3f}]")
print(f"(D2) POOLED taxonomy-aware (semantic tau>=0.6) filter loss:               {pt_[0]:.3f} [{pt_[1]:.3f}, {pt_[2]:.3f}]")
print(f"(D1) decorative/animal motifs (n={len(noncontest_d)//9}): {pn[0]:.3f} [{pn[1]:.3f},{pn[2]:.3f}]  |  plant/food motifs (n={len(contest_d)//9}): {pc[0]:.3f} [{pc[1]:.3f},{pc[2]:.3f}]")
