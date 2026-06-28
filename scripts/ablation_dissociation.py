#!/usr/bin/env python
"""Devil's Advocate C2: the clean schema-vs-prompt dissociation cell.
The published 'schema fix' (exp_optional) changes TWO things at once: it marks
product_type optional AND adds a targeted 'OMIT product_type for entity/theme/motif
queries' instruction. So the 1.00->0.16 drop could be the instruction, not schema
rigidity. This cell isolates schema nullability: the FULL expansion prompt (fill as
many slots as you can), product_type marked OPTIONAL, and NO targeted omit-instruction.
If routing still drops, schema rigidity is causal; if it stays ~1.00, the prompt
instruction was doing the work and the causal claim must be rewritten."""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(__file__))
from run_pilot import load_planner, extract_json
DATA=os.path.join(os.path.dirname(__file__),"..","data"); RES=os.path.join(os.path.dirname(__file__),"..","results")
# Identical to 'expansion' EXCEPT product_type is marked optional. No entity/theme/motif/omit guidance.
NULLABLE=("You are the query-planning component of an e-commerce product search engine.\n"
"A shopper typed a short query. Your job is to produce a RICH structured plan that narrows the catalog as much as possible to the products this shopper most likely wants. Infer the most probable attributes even when the shopper did not state them, so the results are tightly filtered. Fill in as many slots as you reasonably can.\n\n"
"Return ONLY a compact JSON object. Allowed keys: color, material, style, brand, price_range, and product_type (this last key is optional). Values must be short strings.\n\n"
'Query: "{q}"\nJSON:')
MODELS=[("qwen4","mlx-community/Qwen2.5-7B-Instruct-4bit"),("llama4","mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"),
        ("mistral4","mlx-community/Mistral-7B-Instruct-v0.3-4bit"),("gemma4","mlx-community/gemma-2-9b-it-4bit")]
entity=[json.loads(l) for l in open(f"{DATA}/probe/probe.jsonl") if json.loads(l)["family"]=="A_entity"]
out={}
for tag,path in MODELS:
    planner=load_planner(path, NULLABLE)
    forced=0; n=len(entity); plans=[]
    t0=time.time()
    for r in entity:
        plan=extract_json(planner(r["query"])) or {}
        ft=int("product_type" in plan); forced+=ft
        plans.append({"query":r["query"],"plan":plan})
    rate=forced/n
    out[tag]={"forced_product_type_rate":round(rate,3),"n":n}
    json.dump({"tag":f"probe-{tag}-exp_nullable","forced_rate":rate,"n":n,"plans":plans},
              open(f"{RES}/probe-{tag}-exp_nullable__probe.json","w"),indent=1)
    print(f"{tag}: nullable-schema forced product_type = {forced}/{n} = {rate:.2f}  ({time.time()-t0:.0f}s)",flush=True)
print("\n=== DISSOCIATION SUMMARY (forced product_type rate on entity queries) ===")
print("  expansion (rigid, have):    1.00  (all four)")
print("  exp_optional (have):        0.16-0.27  (nullable + targeted omit-instruction)")
print("  exp_nullable (THIS, clean): " + ", ".join(f"{t} {out[t]['forced_product_type_rate']:.2f}" for t,_ in MODELS))
print("ALL DONE",flush=True)
