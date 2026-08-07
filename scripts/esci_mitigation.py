#!/usr/bin/env python
"""R1.5 extension to ESCI + the frontier over-omission check (limitation).
Scores the category omit-instruction fix on the 404 human-graded ESCI queries, from
EXISTING plans (no new model runs): aggressive baseline `esci-{m}-exp__plans.jsonl`
vs the fix `cond-omit_instruction-{m}__esci_plans.jsonl`.

typed(q)   = the aggressive planner emitted a product_type whose head token is a surface
             token of the query (the query literally names a category).
false_application = P(fix still emits a product_type | NOT typed)   [want LOW; the harm the fix targets]
false_omission    = P(fix drops the product_type | typed)           [want LOW; over-omission — the frontier worry]
Reports per model incl. the frontier gpt-4o / gpt-5.2.
Usage: ./.venv/bin/python scripts/esci_mitigation.py"""
import json, os, re
import numpy as np

RES = os.path.join(os.path.dirname(__file__), "..", "results")
DATA = os.path.join(os.path.dirname(__file__), "..", "data")
RNG = np.random.default_rng(13)

# 9 MAIN planners: (aggressive tag, omit tag)
MODELS = [("Qwen2.5-7B", "qwen4"), ("Llama-3.1-8B", "llama4"), ("Mistral-7B", "mistral4"),
          ("Gemma-2-9B", "gemma4"), ("Haiku 4.5", "haiku"), ("Sonnet 4.6", "sonnet"),
          ("Llama-3.3-70B", "together-llama70"), ("gpt-4o", "gpt4o"), ("gpt-5.2", "gpt52")]

def norm(s): return re.sub(r"[^a-z0-9 ]", " ", str(s).lower()).strip()
def toks(s): return set(norm(s).split())
def head(s):
    t = norm(s).split(); return t[-1] if t else ""
def in_query(v, q):
    h = head(v); return bool(h) and (h in toks(q) or norm(v) in norm(q))

# Restrict to the SAME 404 >=5-human-Exact ESCI queries used for the harm analysis,
# so every ESCI number in the paper is on one evaluation set (per R1's N-consistency point).
import pandas as pd
_pool = pd.read_parquet(os.path.join(DATA, "esci", "pool.parquet"))
_ge5 = _pool[_pool.esci_label == "Exact"].groupby("query_id").size()
EVAL_QIDS = set(int(q) for q in _ge5[_ge5 >= 5].index)

def load(path):
    d = {}
    if not os.path.exists(path): return d
    for l in open(path):
        r = json.loads(l); qid = int(r["query_id"])
        if qid not in EVAL_QIDS:   # same 404 >=5-Exact subset as the harm analysis
            continue
        p = r.get("plan"); raw = r.get("raw", "")
        parseable = isinstance(p, dict)
        pt = str((p or {}).get("product_type", "")).strip() if parseable else ""
        d[qid] = {"q": r.get("query", ""), "pt": pt, "parseable": parseable,
                  "raw_json": bool(re.search(r"[{}]", raw))}
    return d

def cluster_ci(vals, B=5000):
    a = np.array(vals, float); n = len(a)
    if n == 0: return (0.0, 0.0, 0.0)
    b = [a[RNG.integers(0, n, n)].mean() for _ in range(B)]
    return (round(float(a.mean()), 3), round(float(np.percentile(b, 2.5)), 3),
            round(float(np.percentile(b, 97.5)), 3))

def emit(rec): return bool(rec and rec["pt"])

out = {"design": "category omit-instruction on 404 ESCI queries; typed = aggressive planner "
                 "emitted a product_type whose head token is in the query; false_application on "
                 "not-typed, false_omission on typed. From cached plans, no new runs.",
       "by_model": {}}
pooled_fa, pooled_fo = [], []
def load_first(*paths):
    for p in paths:
        d = load(p)
        if d: return d
    return {}

for name, tag in MODELS:
    # aggressive baseline: local models used esci-{tag}-exp; frontier reruns used cond-expansion-{tag}
    agg = load_first(os.path.join(RES, f"esci-{tag}-exp__plans.jsonl"),
                     os.path.join(RES, f"cond-expansion-{tag}__esci_plans.jsonl"))
    fix = load(os.path.join(RES, f"cond-omit_instruction-{tag}__esci_plans.jsonl"))
    ids = sorted(set(agg) & set(fix))
    if not ids:
        continue
    typed = [q for q in ids if agg[q]["pt"] and in_query(agg[q]["pt"], agg[q]["q"])]
    typeless = [q for q in ids if q not in set(typed)]
    # false omission: on typed queries, the fix drops the (real) product_type
    fo = [0.0 if emit(fix[q]) else 1.0 for q in typed]
    # false application: on not-typed queries, the fix still emits a product_type not in query
    fa = [1.0 if (emit(fix[q]) and not in_query(fix[q]["pt"], fix[q]["q"])) else 0.0 for q in typeless]
    # aggressive-baseline false application (what the fix is reducing from)
    fa_base = [1.0 if (emit(agg[q]) and not in_query(agg[q]["pt"], agg[q]["q"])) else 0.0 for q in typeless]
    parsefail = np.mean([0.0 if fix[q]["parseable"] else 1.0 for q in ids])
    out["by_model"][tag] = {
        "name": name, "n_typed": len(typed), "n_typeless": len(typeless),
        "false_application_fix": cluster_ci(fa), "false_application_baseline": cluster_ci(fa_base),
        "false_omission_fix": cluster_ci(fo), "parse_fail_rate": round(float(parsefail), 3)}
    pooled_fa += fa; pooled_fo += fo
out["pooled"] = {"false_application_fix": cluster_ci(pooled_fa),
                 "false_omission_fix": cluster_ci(pooled_fo)}
json.dump(out, open(os.path.join(RES, "esci_mitigation.json"), "w"), indent=2)

print("model            n_typed n_tless  FA_base -> FA_fix   FO_fix(over-omit)  parsefail")
for name, tag in MODELS:
    if tag not in out["by_model"]: continue
    e = out["by_model"][tag]
    print(f"{name:16} {e['n_typed']:5d} {e['n_typeless']:6d}   "
          f"{e['false_application_baseline'][0]:.2f} -> {e['false_application_fix'][0]:.2f}      "
          f"{e['false_omission_fix'][0]:.2f}            {e['parse_fail_rate']:.2f}")
print("\npooled  FA_fix", out["pooled"]["false_application_fix"], " FO_fix", out["pooled"]["false_omission_fix"])
