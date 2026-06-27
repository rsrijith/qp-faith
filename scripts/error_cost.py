#!/usr/bin/env python
"""Error-cost / exposure analysis (ESWA R3 ask): translate the per-query recall loss
into a deployment-exposure figure. exposure = (type-less query share) x (mean recall
loss on type-less queries) = fraction of relevant catalog supply silently withheld
across the whole query stream when a planner hard-filters under-specified queries.
Reads the scaled retrieval validations + the type-less base rates."""
import os, re, json
import numpy as np, pandas as pd
RES=os.path.join(os.path.dirname(__file__),"..","results"); DATA=os.path.join(os.path.dirname(__file__),"..","data")
def L(t):
    try: return json.load(open(f"{RES}/{t}__retrieval_validation.json"))
    except: return None

# type-less base rates (recompute, same logic as baselines_detector)
def norm(s): return re.sub(r"[^a-z0-9 ]"," ",str(s).lower()).strip()
STOP=set("a an the of for with and or in on to set sets piece pieces that have by your our".split())
def toks(s): return [w for w in norm(s).split() if w and w not in STOP]
p=pd.read_csv(f"{DATA}/product.csv",sep="\t")
class_tokens=set()
for c in p["product_class"].dropna():
    for w in norm(c).split():
        if len(w)>2: class_tokens.add(w)
def is_concrete(q): return any(w in class_tokens for w in toks(q))
q=pd.read_csv(f"{DATA}/query.csv",sep="\t")
wands_typeless=sum(1 for x in q["query"] if not is_concrete(x))/len(q)

# mean recall loss on entity queries, pooled over the local+API models present
wands_tags=[f"probe-{m}-exp" for m in
            ["qwen4","llama4","mistral4","gemma4","haiku","sonnet","together-llama70","gem3flash","gem31pro","gpt4o","gpt52"]]
drops=[L(t)["mean_recall_drop_real_retrieval"][0] for t in wands_tags if L(t)]
mean_loss=float(np.mean(drops)) if drops else float('nan')
exposure=wands_typeless*mean_loss
out=dict(wands_typeless_share=round(wands_typeless,3),
         mean_recall_loss_entity=round(mean_loss,3),
         exposure_fraction_supply_withheld=round(exposure,3),
         interpretation=(f"On WANDS, {wands_typeless:.1%} of queries are type-less; a planner that hard-filters "
                         f"loses {mean_loss:.0%} of relevant recall on them, so {exposure:.1%} of relevant catalog "
                         f"supply is silently withheld across the whole stream (more on a broad catalog: ESCI "
                         f"type-less share ~40.8%)."))
json.dump(out,open(f"{RES}/error_cost.json","w"),indent=1)
print(json.dumps(out,indent=1))
