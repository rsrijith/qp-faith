#!/usr/bin/env python
"""Gather every canonical number for the finalized paper (Option A: 9-model symmetric
main on WANDS + Amazon/ESCI-full; Gemini Flash + Flash Lite as WANDS-only supplement).
Run after ESCI scoring completes. Output drives the paper Table/section updates."""
import json, os, collections
RES = os.path.join(os.path.dirname(__file__), "..", "results")
def L(p):
    try: return json.load(open(os.path.join(RES, p)))
    except: return None
MAIN = ["qwen4","llama4","mistral4","gemma4","haiku","sonnet","together-llama70","gpt4o","gpt52"]
NAMES = {"qwen4":"Qwen2.5-7B","llama4":"Llama-3.1-8B","mistral4":"Mistral-7B","gemma4":"Gemma-2-9B",
         "haiku":"Haiku 4.5","sonnet":"Sonnet 4.6","together-llama70":"Llama-3.3-70B","gpt4o":"gpt-4o","gpt52":"gpt-5.2"}

print("="*72); print("TABLE: WANDS (9 main models) — con/exp forced, exp recall drop"); print("="*72)
wexp=[]
for m in MAIN:
    pc=L(f"probe-{m}-con__probe_metrics.json"); pe=L(f"probe-{m}-exp__probe_metrics.json"); rv=L(f"probe-{m}-exp__retrieval_validation.json")
    fc=pc["A_product_type_forced_rate"][0] if pc else None; fe=pe["A_product_type_forced_rate"][0] if pe else None
    rd=rv["mean_recall_drop_real_retrieval"][0] if rv else None
    if rd is not None: wexp.append(rd)
    print(f"  {NAMES[m]:16} con {fc if fc else 0:.2f}  exp {fe if fe else 0:.2f}  recall_drop {rd if rd else 0:.2f}")
if wexp: print(f"  WANDS recall-drop range: {min(wexp):.2f}-{max(wexp):.2f}")

print("\n"+"="*72); print("TABLE: Amazon/ESCI-full (9 main models) — con/exp forced, exp recall drop"); print("="*72)
eexp=[]
for m in MAIN:
    pc=L(f"probe-esci-{m}-con__probe_metrics.json"); pe2=L(f"probe-esci-{m}-exp__probe_metrics.json")
    em=L(f"probe-esci-{m}-exp__esci_probe_metrics.json")
    fc=pc["A_product_type_forced_rate"][0] if pc else None; fe=pe2["A_product_type_forced_rate"][0] if pe2 else None
    rd=em["real_recall_drop"][0] if em else None
    if rd is not None: eexp.append(rd)
    print(f"  {NAMES[m]:16} con {fc if fc is not None else -1:.2f}  exp {fe if fe is not None else -1:.2f}  recall_drop {rd if rd is not None else -1:.2f}")
if eexp: print(f"  ESCI recall-drop range: {min(eexp):.2f}-{max(eexp):.2f}")

print("\n=== GEMINI SUPPLEMENT (WANDS-only): forced con/exp ===")
for m in ["gem3flash","gem25flashlite"]:
    pc=L(f"probe-{m}-con__probe_metrics.json"); pe=L(f"probe-{m}-exp__probe_metrics.json")
    print(f"  {m}: con {pc['A_product_type_forced_rate'][0]:.2f}  exp {pe['A_product_type_forced_rate'][0]:.2f}" if pc and pe else f"  {m}: missing")

print("\n=== SCHEMA-FIX (opt) forced rate, main models w/ opt ===")
for m in MAIN:
    po=L(f"probe-{m}-opt__probe_metrics.json")
    if po: print(f"  {m}: {po['A_product_type_forced_rate'][0]:.2f}")

print("\n=== SLOT-LOCUS (per-slot P(spurious excludes relevant), mean over main models, WANDS exp) ===")
slots=["product_type","color","material","style"]; agg={s:[] for s in slots}
for m in MAIN:
    mm=L(f"{m}-exp__metrics_v2.json")
    if mm and "per_slot" in mm:
        for s in slots:
            d=mm["per_slot"].get(s)
            if d and d["spurious"]>0: agg[s].append(d["harmful"]/d["spurious"])
import statistics
for s in slots:
    if agg[s]: print(f"  {s}: {statistics.mean(agg[s]):.2f}")

print("\n=== TAXONOMY (WANDS exp injections, main 9 models) ===")
vals=collections.Counter(); n=0
for m in MAIN:
    try: rows=[json.loads(l) for l in open(f"{RES}/probe-{m}-exp__probe_plans.jsonl")]
    except: continue
    for r in rows:
        if r["family"]=="A_entity" and (r.get("plan") or {}).get("product_type"):
            vals[r["plan"]["product_type"].strip().lower()]+=1; n+=1
print(f"  total injected={n}; top={vals.most_common(8)}")

print("\n=== probe scale ===")
wa=sum(1 for l in open(os.path.join(os.path.dirname(__file__),'..','data','probe','probe.jsonl')) if json.loads(l)['family']=='A_entity')
ea=sum(1 for _ in open(os.path.join(os.path.dirname(__file__),'..','data','probe_esci','probe_full.jsonl')))
print(f"  WANDS entity motifs={wa}, ESCI-full motifs={ea}, combined queries~{wa+ea}")
