#!/usr/bin/env python
"""Comprehensive data integrity check across ALL runs (WANDS 11 models + ESCI 9 models).
Flags: wrong row count, high empty-plan rate on con/exp (= corruption/credit-failure,
needs re-run), implausible forced rates. opt-mode high-empty is EXPECTED (the fix omits).
Prints a FIX LIST of tags to re-run."""
import json, os, sys
RES = os.path.join(os.path.dirname(__file__), "..", "results")
DATA = os.path.join(os.path.dirname(__file__), "..", "data")
NW = len(open(f"{DATA}/probe/probe.jsonl").readlines())
NE = len(open(f"{DATA}/probe_esci/probe_full.jsonl").readlines())
WMODELS = ["qwen4","llama4","mistral4","gemma4","haiku","sonnet","together-llama70","gem3flash","gem25flashlite","gpt4o","gpt52"]
EMODELS = ["qwen4","llama4","mistral4","gemma4","haiku","sonnet","together-llama70","gpt4o","gpt52"]  # ESCI: 9 (no Gemini API)

def check(tag, N):
    f = f"{RES}/{tag}__probe_plans.jsonl"
    if not os.path.exists(f): return ("MISSING", 0, 0.0, float('nan'))
    rows = [json.loads(l) for l in open(f) if l.strip()]
    n = len(rows); ent = [r for r in rows if r.get("family") == "A_entity"]
    # ERROR = empty raw string (API/credit failure). ABSTAIN = raw present but no/empty plan (model returned {} — valid, faithful).
    err = sum(1 for r in rows if not (r.get("raw") or "").strip()) / n if n else 1.0
    ft = sum(1 for r in ent if "product_type" in (r.get("plan") or {})) / len(ent) if ent else float('nan')
    mode = tag.split("-")[-1]
    if n < N: status = "INCOMPLETE"
    elif err > 0.05: status = "ERROR(raw-empty)"   # genuine API failures
    elif mode == "exp" and ft < 0.7: status = "LOW-FORCED?"
    else: status = "ok"
    return (status, n, err, ft)

fix = []
for dom, models, N, pfx in [("WANDS", WMODELS, NW, "probe-"), ("ESCI", EMODELS, NE, "probe-esci-")]:
    print(f"\n=== {dom} (expect {N}) ===")
    modes = ["con","exp","opt"] if dom == "WANDS" else ["con","exp"]
    for m in models:
        for s in modes:
            tag = f"{pfx}{m}-{s}"
            st, n, empty, ft = check(tag, N)
            flag = "" if st in ("ok","LOW-FORCED?") else "  <<< FIX"
            if st in ("MISSING","INCOMPLETE","ERROR(raw-empty)"): fix.append(tag)
            print(f"  {tag:30} {st:16} rows={n} err={empty:.0%} forced={ft if ft==ft else 0:.2f}{flag}")
print("\n=== FIX LIST (re-run these) ===")
print("\n".join(fix) if fix else "  none — all clean")
