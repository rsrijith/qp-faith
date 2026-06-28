#!/usr/bin/env python
"""Extend the text-to-SQL existence check to more models (review D4): does the planner
mis-route a spurious category EQUALITY in free-form SQL? Llama, Mistral, Gemma + Qwen."""
import os, sys, json, re, time
sys.path.insert(0, os.path.dirname(__file__))
from run_pilot import load_planner
DATA=os.path.join(os.path.dirname(__file__),"..","data"); RES=os.path.join(os.path.dirname(__file__),"..","results")
SQL_PROMPT=("You query a products database with schema: "
"products(id, title, description, category, color, material, brand, price). "
"The shopper searched: '{q}'. Write ONE SQL WHERE clause that narrows the catalog as much "
"as possible to the most relevant products; infer as many constraints as you reasonably can. "
"Output ONLY the WHERE clause (starting with WHERE), nothing else.")
eq_pred=re.compile(r"\b(category|product_type|type)\s*=\s*'([^']*)'", re.I)
motifs=[json.loads(l)["query"] for l in open(f"{DATA}/probe/probe.jsonl") if json.loads(l)["family"]=="A_entity"]
MODELS=[("llama4","mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"),
        ("mistral4","mlx-community/Mistral-7B-Instruct-v0.3-4bit"),
        ("gemma4","mlx-community/gemma-2-9b-it-4bit")]
for tag,path in MODELS:
    planner=load_planner(path, SQL_PROMPT); t0=time.time(); n_eq=0; rows=[]
    for q in motifs:
        raw=planner(q); where=raw[raw.upper().find("WHERE"):] if "WHERE" in raw.upper() else raw
        where=where.split("\n")[0][:300]
        eqs=eq_pred.findall(where); spur=any(q.lower() not in v.lower() and v.lower() not in q.lower() for _,v in eqs)
        n_eq+=spur; rows.append({"q":q,"where":where,"spurious_category_equality":spur})
    rep={"model":path,"n":len(motifs),"category_equality_spurious_rate":round(n_eq/len(motifs),3),"rows":rows}
    json.dump(rep,open(f"{RES}/micro_sql_{tag}.json","w"),indent=1)
    print(f"{tag}: spurious category-equality {n_eq}/{len(motifs)} = {n_eq/len(motifs):.0%}  ({time.time()-t0:.0f}s)",flush=True)
print("ALL DONE",flush=True)
