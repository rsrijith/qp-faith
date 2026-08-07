#!/usr/bin/env python
"""Blind-panel-4 move (c): convert the ESCI selectivity counterfactual into a REAL
routing + harm measurement on human grades.

We run the query planners on the 480 real ESCI queries (cached plans, the same 4
local models the paper's causal/harm results use) and ask: do planners spuriously
route a BRAND or COLOR the shopper never typed, and if that emitted value is
applied as a hard equality filter to the query's HUMAN-Exact set, how much is
excluded?

For facet f in {brand, color}, per query:
  emitted   = plan[f] is non-empty
  in_query  = the emitted value's head token appears in the query text
  spurious  = emitted AND NOT in_query           (routed a value the shopper never typed)
  exclusion = 1 - (fraction of the query's human-Exact products whose f matches
                   the emitted value)             (realized selectivity on human grades)
Report the spurious-routing rate, the exclusion-when-spuriously-emitted, and the
realized recall loss = mean over queries of (spurious ? exclusion : 0), with
query-clustered bootstrap CIs, per model and pooled. Uses only cached plans +
data/esci/pool.parquet; no new inference. Run: ./.venv/bin/python scripts/esci_routing.py
"""
import os, re, json, collections
import numpy as np, pandas as pd
HERE=os.path.dirname(os.path.abspath(__file__)); RES=os.path.join(HERE,"..","results"); DATA=os.path.join(HERE,"..","data")
RNG=np.random.default_rng(13); MODELS=["qwen4","llama4","mistral4","gemma4"]; MIN_EXACT=5
def norm(s): return re.sub(r"[^a-z0-9 ]"," ",str(s).lower()).strip()
def head(s):
    t=norm(s).split(); return t[-1] if t else ""
def toks(s): return set(norm(s).split())

pool=pd.read_parquet(f"{DATA}/esci/pool.parquet")
E=pool[pool.esci_label=="Exact"]
# per query_id: list of (brand,color) for Exact products
exact={}
for qid,g in E.groupby("query_id"):
    exact[qid]=[(norm(b),norm(c)) for b,c in zip(g.product_brand,g.product_color)]
qtext={int(r.query_id):str(r["query"]) for _,r in
       pd.DataFrame([json.loads(l) for l in open(f"{DATA}/esci/queries.jsonl")]).iterrows()}

def facet_match(emitted, val):
    """lenient hard-equality match between an emitted facet value and a product's value."""
    e=norm(emitted); v=norm(val)
    if not e or not v: return False
    return e==v or e in v or v in e or bool(toks(e)&toks(v))

def in_query(emitted, q):
    e=head(emitted)
    return bool(e) and (e in toks(q) or norm(emitted) in norm(q))

# R1 comment 3: a non-committal placeholder (brand="unknown"/"any"/"varies"/...) is the
# planner declining to guess a value, not routing a spurious one. Treat as no emission.
NONCOMMITTAL={"any","none","n a","na","unknown","unspecified","not specified","varies",
 "various","varied","generic","unbranded","brandless","no brand","no color","no colour",
 "multiple","assorted","misc","miscellaneous","not applicable","not available","other",
 "standard","default","null","nil","tbd","undefined"}
def is_committal(v):
    n=norm(v); return bool(n) and n not in NONCOMMITTAL

def cluster_ci(vals,B=5000):
    a=np.array(vals,float); n=len(a)
    if n==0: return (0.0,0.0,0.0)
    b=[a[RNG.integers(0,n,n)].mean() for _ in range(B)]
    return round(float(a.mean()),3),round(float(np.percentile(b,2.5)),3),round(float(np.percentile(b,97.5)),3)

def analyze(mode):
    res={}
    for facet in ["brand","color"]:
        # accumulate per-query (pooled across models) and per-model
        per_model={m:{"spur":[], "loss":[], "excl":[]} for m in MODELS}
        for m in MODELS:
            fn=f"{RES}/esci-{m}-{mode}__plans.jsonl"
            if not os.path.exists(fn): continue
            for l in open(fn):
                r=json.loads(l); qid=int(r["query_id"]); p=r.get("plan") or {}
                if not isinstance(p,dict): continue
                if qid not in exact or len(exact[qid])<MIN_EXACT: continue
                q=qtext.get(qid,""); emitted_raw=str(p.get(facet,"")).strip()
                emitted = emitted_raw if is_committal(emitted_raw) else ""
                spurious = bool(emitted) and not in_query(emitted,q)
                per_model[m]["spur"].append(1.0 if spurious else 0.0)
                if spurious:
                    kept=np.mean([1.0 if facet_match(emitted, ev[0 if facet=="brand" else 1]) else 0.0
                                  for ev in exact[qid]])
                    excl=1.0-kept
                    per_model[m]["excl"].append(excl)
                    per_model[m]["loss"].append(excl)
                else:
                    per_model[m]["loss"].append(0.0)
        # pooled across models (query-clustered = per (model,query) rows; report simple pooled)
        allspur=[x for m in MODELS for x in per_model[m]["spur"]]
        allloss=[x for m in MODELS for x in per_model[m]["loss"]]
        allexcl=[x for m in MODELS for x in per_model[m]["excl"]]
        res[facet]={
          "spurious_routing_rate_pooled":cluster_ci(allspur),
          "exclusion_when_spurious_pooled":cluster_ci(allexcl),
          "realized_loss_pooled":cluster_ci(allloss),
          "by_model":{m:{"spurious_routing_rate":round(float(np.mean(per_model[m]["spur"])),3) if per_model[m]["spur"] else None,
                         "realized_loss":round(float(np.mean(per_model[m]["loss"])),3) if per_model[m]["loss"] else None}
                      for m in MODELS},
          "n_queries":len(per_model[MODELS[0]]["spur"])}
    return res

out={"design":"real routing on 404 ESCI queries (the >=5 human-Exact subset of 480), 4 local planners; spurious=emitted value not in query; exclusion=hard equality on the human-Exact set","min_exact":MIN_EXACT}
for mode,label in [("exp","expansion"),("con","conservative")]:
    out[label]=analyze(mode)
json.dump(out,open(f"{RES}/esci_routing.json","w"),indent=2)
def f(t): m,lo,hi=t; return f"{m} [{lo}, {hi}]"
for label in ["expansion","conservative"]:
    print(f"=== {label} (n={out[label]['brand']['n_queries']} queries, 4 local models) ===")
    for facet in ["brand","color"]:
        d=out[label][facet]
        print(f"  {facet}: spurious-routing {f(d['spurious_routing_rate_pooled'])} ; "
              f"exclusion-when-spurious {f(d['exclusion_when_spurious_pooled'])} ; "
              f"realized loss {f(d['realized_loss_pooled'])}")
        print(f"      by-model spurious-routing: "+", ".join(f"{m} {d['by_model'][m]['spurious_routing_rate']}" for m in MODELS))
print("wrote",f"{RES}/esci_routing.json")
