#!/usr/bin/env python
"""R1.5 extension: does a brand/color-specific omit-instruction reduce the spurious
brand/color routing measured in esci_routing.py? Compares the aggressive baseline vs the
`omit_bc` condition on the same 404 >=5-Exact ESCI queries, placeholder-aware (is_committal).
committal-spurious routing = emitted committal value whose head token is not in the query.
Usage: ./.venv/bin/python scripts/esci_bc_mitigation.py"""
import json, os, re
import numpy as np
import pandas as pd

RES = os.path.join(os.path.dirname(__file__), "..", "results")
DATA = os.path.join(os.path.dirname(__file__), "..", "data")
RNG = np.random.default_rng(13)

def norm(s): return re.sub(r"[^a-z0-9 ]", " ", str(s).lower()).strip()
def toks(s): return set(norm(s).split())
def head(s):
    t = norm(s).split(); return t[-1] if t else ""
def in_query(v, q):
    h = head(v); return bool(h) and (h in toks(q) or norm(v) in norm(q))
NONCOMMITTAL = {"any", "none", "n a", "na", "unknown", "unspecified", "not specified",
 "varies", "various", "varied", "generic", "unbranded", "brandless", "no brand", "no color",
 "no colour", "multiple", "assorted", "misc", "miscellaneous", "not applicable",
 "not available", "other", "standard", "default", "null", "nil", "tbd", "undefined"}
def is_committal(v):
    n = norm(v); return bool(n) and n not in NONCOMMITTAL

_pool = pd.read_parquet(os.path.join(DATA, "esci", "pool.parquet"))
_ge5 = _pool[_pool.esci_label == "Exact"].groupby("query_id").size()
EVAL = set(int(q) for q in _ge5[_ge5 >= 5].index)

MODELS = [("Qwen2.5-7B", "qwen4"), ("Llama-3.1-8B", "llama4"), ("Mistral-7B", "mistral4"),
          ("Gemma-2-9B", "gemma4"), ("gpt-4o", "gpt4o"), ("gpt-5.2", "gpt52")]

def load(*paths):
    for path in paths:
        if os.path.exists(path):
            d = {}
            for l in open(path):
                r = json.loads(l); qid = int(r["query_id"])
                if qid not in EVAL: continue
                p = r.get("plan"); d[qid] = {"q": r.get("query", ""), "plan": p if isinstance(p, dict) else {}}
            if d: return d
    return {}

def committal_spurious(rec, facet):
    v = str(rec["plan"].get(facet, "")).strip()
    return 1.0 if (is_committal(v) and not in_query(v, rec["q"])) else 0.0

def ci(vals, B=5000):
    a = np.array(vals, float); n = len(a)
    if n == 0: return (0.0, 0.0, 0.0)
    b = [a[RNG.integers(0, n, n)].mean() for _ in range(B)]
    return (round(float(a.mean()), 3), round(float(np.percentile(b, 2.5)), 3), round(float(np.percentile(b, 97.5)), 3))

out = {"design": "brand/color omit_bc fix vs aggressive baseline, 404 >=5-Exact ESCI queries, "
                 "committal-spurious routing (placeholder-aware).", "by_model": {}}
print(f"{'model':12} {'brand agg->fix':>16} {'color agg->fix':>16}")
for name, tag in MODELS:
    agg = load(os.path.join(RES, f"esci-{tag}-exp__plans.jsonl"),
               os.path.join(RES, f"cond-expansion-{tag}__esci_plans.jsonl"))
    fix = load(os.path.join(RES, f"cond-omit_bc-{tag}__esci_plans.jsonl"))
    ids = sorted(set(agg) & set(fix))
    if len(ids) < 50:
        print(f"{name:12}  (incomplete: {len(ids)} overlap)"); continue
    ba = [committal_spurious(agg[q], "brand") for q in ids]
    bf = [committal_spurious(fix[q], "brand") for q in ids]
    ca = [committal_spurious(agg[q], "color") for q in ids]
    cf = [committal_spurious(fix[q], "color") for q in ids]
    out["by_model"][tag] = {"name": name, "n": len(ids),
        "brand_spurious_baseline": ci(ba), "brand_spurious_fix": ci(bf),
        "color_spurious_baseline": ci(ca), "color_spurious_fix": ci(cf)}
    print(f"{name:12}  {ci(ba)[0]:.2f} -> {ci(bf)[0]:.2f}      {ci(ca)[0]:.2f} -> {ci(cf)[0]:.2f}   (n={len(ids)})")
json.dump(out, open(os.path.join(RES, "esci_bc_mitigation.json"), "w"), indent=2)
print("\nwrote results/esci_bc_mitigation.json")
