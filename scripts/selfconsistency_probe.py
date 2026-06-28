#!/usr/bin/env python
"""Review D3: is self-consistency a free abstention signal for the category slot?
Sample k plans per query at temp>0 (4 local models, expansion prompt). For each query
measure (a) emit-rate of product_type and (b) cross-sample agreement on the emitted type.
If type-less ENTITY queries show lower self-agreement than concrete ATTRIBUTED (Family B)
queries, disagreement separates them -> a usable abstention signal; build a risk-coverage
curve. If not (the planner is confidently consistent on a wrong type), report the null."""
import os, sys, json, time, collections, re
sys.path.insert(0, os.path.dirname(__file__))
from run_pilot import PROMPTS, extract_json
DATA=os.path.join(os.path.dirname(__file__),"..","data"); RES=os.path.join(os.path.dirname(__file__),"..","results")
K=5; TEMP=0.8
def norm(s): return re.sub(r"[^a-z0-9 ]"," ",str(s).lower()).strip()
def load_sampling(path, tmpl, temp, seed):
    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler
    model,tok=load(path); sampler=make_sampler(temp=temp)
    def run(q):
        msgs=[{"role":"user","content":tmpl.format(q=q)}]
        prompt=tok.apply_chat_template(msgs,add_generation_prompt=True)
        return generate(model,tok,prompt=prompt,max_tokens=120,sampler=sampler,verbose=False)
    return run
probe=[json.loads(l) for l in open(f"{DATA}/probe/probe.jsonl")]
MODELS=[("qwen4","mlx-community/Qwen2.5-7B-Instruct-4bit"),("llama4","mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"),
        ("mistral4","mlx-community/Mistral-7B-Instruct-v0.3-4bit"),("gemma4","mlx-community/gemma-2-9b-it-4bit")]
def agreement(types):  # modal-fraction among samples that emitted a type; None-handling
    emitted=[t for t in types if t]
    if not emitted: return 0.0, 0
    c=collections.Counter(norm(t).split()[-1] if (norm(t) and norm(t).split()) else "" for t in emitted)  # head noun
    return max(c.values())/len(types), len(emitted)
allrows=[]
for tag,path in MODELS:
    planner=load_sampling(path, PROMPTS["expansion"], TEMP, 0); t0=time.time()
    for r in probe:
        types=[ (extract_json(planner(r["query"])) or {}).get("product_type") for _ in range(K)]
        agr,nem=agreement(types)
        allrows.append({"model":tag,"query":r["query"],"family":r["family"],
                        "emit_rate":nem/K,"type_agreement":agr,"distinct":len(set(norm(t).split()[-1] for t in types if t and norm(t).split()))})
    print(f"{tag}: done ({time.time()-t0:.0f}s)",flush=True)
json.dump(allrows,open(f"{RES}/selfconsistency.json","w"),indent=1)
import numpy as np
ent=[x for x in allrows if x["family"]=="A_entity"]; att=[x for x in allrows if x["family"]=="B_attr"]
ea=np.mean([x["type_agreement"] for x in ent]); aa=np.mean([x["type_agreement"] for x in att])
ed=np.mean([x["distinct"] for x in ent]); ad=np.mean([x["distinct"] for x in att])
# simple AUC: can type_agreement rank concrete>entity?
from itertools import product as iproduct
pairs=list(iproduct([x["type_agreement"] for x in att],[x["type_agreement"] for x in ent]))
auc=np.mean([1.0 if a>e else (0.5 if a==e else 0.0) for a,e in pairs])
print(f"\n=== SELF-CONSISTENCY (k={K}, temp={TEMP}) ===")
print(f"  type-agreement: ENTITY {ea:.2f} vs CONCRETE/attr {aa:.2f}  (higher=more self-consistent)")
print(f"  distinct head-types per query: ENTITY {ed:.2f} vs CONCRETE {ad:.2f}")
print(f"  separation AUC (does low agreement flag type-less?): {auc:.2f}")
json.dump({"k":K,"temp":TEMP,"entity_agreement":float(ea),"concrete_agreement":float(aa),
           "entity_distinct":float(ed),"concrete_distinct":float(ad),"separation_auc":float(auc)},
          open(f"{RES}/selfconsistency_summary.json","w"),indent=2)
print("ALL DONE",flush=True)
