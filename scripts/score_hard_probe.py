#!/usr/bin/env python
"""R1.5: score the category false-application on the HARD type-less probe (misspellings,
compositional, attribute-modifier queries), aggressive (expansion) vs the omit-instruction
fix, per variant and model. Every hard query is type-less by construction, so any emitted
product_type is a false application. Usage: ./.venv/bin/python scripts/score_hard_probe.py"""
import json, os, re
import numpy as np
RES = os.path.join(os.path.dirname(__file__), "..", "results")
DATA = os.path.join(os.path.dirname(__file__), "..", "data")
RNG = np.random.default_rng(13)

variant_of = {json.loads(l)["query"]: json.loads(l)["variant"]
              for l in open(f"{DATA}/probe/probe_hard.jsonl")}
VARIANTS = ["misspelled", "compositional_entity", "modifier"]
MODELS = [("Qwen2.5-7B", "qwen4"), ("Haiku 4.5", "haiku"), ("gpt-4o", "gpt4o")]

def emit(plan):
    if not isinstance(plan, dict): return None       # parse failure
    v = str(plan.get("product_type", "")).strip().lower()
    return 1.0 if (v and v not in {"", "any", "none", "unknown", "n/a"}) else 0.0

def load(cond, tag):
    p = os.path.join(RES, f"cond-{cond}-{tag}__probe_hard_plans.jsonl")
    d = {}
    if not os.path.exists(p): return d
    for l in open(p):
        r = json.loads(l); d[r["query"]] = r.get("plan")
    return d

def ci(vals, B=5000):
    a = np.array([x for x in vals if x is not None], float); n = len(a)
    if n == 0: return (0.0, 0.0, 0.0, 0)
    b = [a[RNG.integers(0, n, n)].mean() for _ in range(B)]
    return (round(float(a.mean()), 3), round(float(np.percentile(b, 2.5)), 3),
            round(float(np.percentile(b, 97.5)), 3), n)

out = {"design": "category false-application on 165 HARD type-less queries (55 misspelled, "
                 "55 compositional entity+entity, 55 attribute-modifier), aggressive vs "
                 "omit-instruction fix. Any product_type emission is a false application.",
       "by_model": {}}
print(f"{'model':11} {'variant':22} {'FA aggressive':>14} {'FA fix':>10}")
for name, tag in MODELS:
    agg = load("expansion", tag); fix = load("omit_instruction", tag)
    if not agg or not fix:
        print(f"{name:11}  (incomplete: agg={len(agg)} fix={len(fix)})"); continue
    out["by_model"][tag] = {"name": name}
    for var in VARIANTS:
        qs = [q for q, v in variant_of.items() if v == var]
        fa_a = ci([emit(agg[q]) for q in qs if q in agg])
        fa_f = ci([emit(fix[q]) for q in qs if q in fix])
        out["by_model"][tag][var] = {"aggressive": fa_a, "fix": fa_f}
        print(f"{name:11} {var:22} {fa_a[0]:.2f} [{fa_a[1]:.2f},{fa_a[2]:.2f}]   {fa_f[0]:.2f} [{fa_f[1]:.2f},{fa_f[2]:.2f}]")
json.dump(out, open(os.path.join(RES, "hard_probe_mitigation.json"), "w"), indent=2)
print("\nwrote results/hard_probe_mitigation.json")
