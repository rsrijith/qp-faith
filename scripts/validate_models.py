#!/usr/bin/env python
"""Health-check the scaled run: for each CURRENT-run tag, confirm the model produced
valid output. Distinguishes genuine format failure (con/exp empty plans) from the
optional-schema fix correctly omitting (opt empty plans = intended)."""
import json, os
RES=os.path.join(os.path.dirname(__file__),"..","results")
WANDS=[f"probe-{m}-{s}" for m in ["qwen4","llama4","mistral4","gemma4","haiku","sonnet","together-llama70","gem3flash","gem31pro","gpt4o","gpt52"] for s in ["con","exp","opt"]]
ESCI=[f"probe-esci-{m}-{s}" for m in ["qwen4","llama4","mistral4","gemma4"] for s in ["con","exp","opt"]]
NW=len(open(os.path.join(os.path.dirname(__file__),"..","data","probe","probe.jsonl")).readlines())
NE=len(open(os.path.join(os.path.dirname(__file__),"..","data","probe_esci","probe.jsonl")).readlines())
def load(f):
    rows=[]
    for l in open(f):
        l=l.strip()
        if not l: continue
        try: rows.append(json.loads(l))
        except: pass
    return rows
def check(tags,N,dom):
    print(f"\n=== {dom} (expect {N} rows) ===")
    print(f"{'tag':28} {'rows':>5} {'empty%':>7} {'forced(ent)':>11}  status")
    for t in tags:
        f=f"{RES}/{t}__probe_plans.jsonl"
        if not os.path.exists(f): print(f"{t:28} {'--':>5} {'':>7} {'':>11}  pending"); continue
        rows=load(f); n=len(rows); ent=[r for r in rows if r.get("family")=="A_entity"]
        empty=sum(1 for r in rows if not r.get("plan"))/n if n else 1
        ft=sum(1 for r in ent if "product_type" in (r.get("plan") or {}))/len(ent) if ent else float('nan')
        mode=t.split("-")[-1]
        if n==0: st="ERROR empty"
        elif n<N: st=f"running {n}/{N}"
        elif mode in ("con","exp") and empty>0.4: st="WARN format-fail"
        elif mode=="exp" and ft<0.8: st="WARN low-forced"
        elif mode=="opt" and ft<0.4: st="ok (fix omitting)"
        else: st="ok"
        fts=f"{ft:.2f}" if ft==ft else "--"
        print(f"{t:28} {n:>5} {empty:>7.2f} {fts:>11}  {st}")
check(WANDS,NW,"WANDS"); check(ESCI,NE,"ESCI")
