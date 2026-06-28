#!/usr/bin/env python
"""C6 / MJ-quant: is the schema-vs-prompt dissociation a 4-bit quantization artifact?
Re-run the 3-way dissociation (expansion / nullable-no-instruction / omit-instruction)
on a FULL-PRECISION (bf16) model, same family as the 4-bit qwen4. If the pattern holds
(expansion ~1.00, nullable ~1.00, omit-instruction ~0.16), it is not a quantization effect."""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(__file__))
from run_pilot import load_planner, PROMPTS, extract_json
DATA=os.path.join(os.path.dirname(__file__),"..","data"); RES=os.path.join(os.path.dirname(__file__),"..","results")
NULLABLE=("You are the query-planning component of an e-commerce product search engine.\n"
"A shopper typed a short query. Your job is to produce a RICH structured plan that narrows the catalog as much as possible to the products this shopper most likely wants. Infer the most probable attributes even when the shopper did not state them, so the results are tightly filtered. Fill in as many slots as you reasonably can.\n\n"
"Return ONLY a compact JSON object. Allowed keys: color, material, style, brand, price_range, and product_type (this last key is optional). Values must be short strings.\n\n"
'Query: "{q}"\nJSON:')
import sys; PATH=sys.argv[1] if len(sys.argv)>1 else "mlx-community/Qwen2.5-7B-Instruct-bf16"
entity=[json.loads(l) for l in open(f"{DATA}/probe/probe.jsonl") if json.loads(l)["family"]=="A_entity"]
out={}
for label,tmpl in [("expansion",PROMPTS["expansion"]),("nullable",NULLABLE),("omit_instruction",PROMPTS["exp_optional"])]:
    pl=load_planner(PATH,tmpl); t0=time.time()
    forced=sum(1 for r in entity if "product_type" in (extract_json(pl(r["query"])) or {}))
    out[label]=round(forced/len(entity),3)
    print(f"{label}: forced {forced}/{len(entity)} = {forced/len(entity):.2f} ({time.time()-t0:.0f}s)",flush=True)
json.dump({"model":PATH.split("/")[-1],"n":len(entity),"forced_rate":out},open(f"{RES}/dissociation_fullprec_{PATH.split(chr(45))[-1].replace(chr(47),chr(95))}.json","w"),indent=2)
print(f"FULL-PRECISION dissociation: expansion {out['expansion']}, nullable {out['nullable']}, omit-instruction {out['omit_instruction']}  (4-bit was 1.00/1.00/0.16)",flush=True)
print("ALL DONE",flush=True)
