#!/usr/bin/env python
"""MJ1: is the category-routing behavior decor-specific? Routing-only probe (no catalog,
no relevance) on type-less motifs drawn from NON-decor domains (occasions, activities,
themes spanning apparel/electronics/grocery/toys). Measure forced product_type rate."""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(__file__))
from run_pilot import load_planner, PROMPTS, extract_json
RES=os.path.join(os.path.dirname(__file__),"..","results")
MOTIFS=["wedding","birthday","halloween","christmas","graduation","anniversary","camping","yoga",
        "hiking","gaming","running","fishing","baby shower","retirement","tailgate","space","ocean",
        "superhero","unicorn","tropical","western","nautical","vintage","minimalist","industrial"]
MODELS=[("qwen4","mlx-community/Qwen2.5-7B-Instruct-4bit"),("llama4","mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"),
        ("mistral4","mlx-community/Mistral-7B-Instruct-v0.3-4bit"),("gemma4","mlx-community/gemma-2-9b-it-4bit")]
out={}
for tag,path in MODELS:
    pl=load_planner(path, PROMPTS["expansion"]); t0=time.time()
    forced=sum(1 for m in MOTIFS if "product_type" in (extract_json(pl(m)) or {}))
    out[tag]=forced/len(MOTIFS)
    print(f"{tag}: forced product_type on {len(MOTIFS)} non-decor motifs = {forced}/{len(MOTIFS)} = {forced/len(MOTIFS):.2f} ({time.time()-t0:.0f}s)",flush=True)
json.dump({"n_motifs":len(MOTIFS),"motifs":MOTIFS,"forced_rate":out},open(f"{RES}/nondecor_routing.json","w"),indent=2)
print("band:",f"{min(out.values()):.2f}-{max(out.values()):.2f}",flush=True); print("ALL DONE",flush=True)
