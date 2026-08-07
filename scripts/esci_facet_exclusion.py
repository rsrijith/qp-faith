#!/usr/bin/env python
"""Blind-panel-3 LEVER B1: does the coverage-selectivity mechanism generalize
beyond the WANDS category slot, to other high-coverage hard facets, on a second
dataset with HUMAN intent grades?

ESCI pool (data/esci/pool.parquet) carries human esci_label (Exact/Substitute/
Irrelevant/Complement) plus product_brand and product_color. For each query we
take the human-Exact relevant set and ask: if a planner emitted ONE guessed
brand (or color) and it were applied as a hard equality pre-filter, how much of
the human-relevant set would it exclude? Exclusion = 1 - fraction of the relevant
set carrying the single most common (plurality) value, since a hard equality also
drops products whose facet is missing or different. Reported over queries with
>=5 Exact products, with a query-bootstrap 95% CI.

The point: a high-coverage facet applied as a hard equality is exclusionary for
ANY such facet; category is not special except that planners route into it. This
generalizes the mechanism off WANDS and off the by-construction category label,
onto human-graded relevance.

Run: ./.venv/bin/python scripts/esci_facet_exclusion.py
"""
import os, json
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, "..", "results")
RNG = np.random.default_rng(13)
df = pd.read_parquet(os.path.join(HERE, "..", "data", "esci", "pool.parquet"))

import collections
E = df[df.esci_label == "Exact"].copy()
MIN = 5
def norm(v):
    v = str(v).strip().lower()
    return "" if v in ("", "none", "nan") else v

def facet_stats(col):
    cov, excl, ndist = [], [], []
    for qid, g in E.groupby("query_id"):
        if len(g) < MIN: continue
        vals = [norm(v) for v in g[col].tolist()]
        present = [v for v in vals if v]
        cov.append(len(present) / len(vals))
        if not present:
            excl.append(1.0); ndist.append(0); continue
        plurality = collections.Counter(present).most_common(1)[0][0]
        kept = sum(1 for v in vals if v == plurality) / len(vals)  # hard equality drops missing/different
        excl.append(1.0 - kept)
        ndist.append(len(set(present)))
    return np.array(cov), np.array(excl), np.array(ndist)

def ci(a, B=5000):
    a = np.asarray(a, float); n = len(a)
    boots = [a[RNG.integers(0, n, n)].mean() for _ in range(B)]
    return round(float(a.mean()), 3), round(float(np.percentile(boots, 2.5)), 3), round(float(np.percentile(boots, 97.5)), 3)

out = {"dataset": "ESCI (data/esci/pool.parquet)", "relevance": "human esci_label == Exact",
       "min_relevant": MIN, "n_queries": int((E.groupby("query_id").size() >= MIN).sum())}
for col, name in [("product_brand", "brand"), ("product_color", "color")]:
    cov, excl, ndist = facet_stats(col)
    out[f"{name}_coverage"] = ci(cov)
    out[f"{name}_single_guess_exclusion"] = ci(excl)
    out[f"{name}_distinct_values_per_query_median"] = float(np.median(ndist))
json.dump(out, open(f"{RES}/esci_facet_exclusion.json", "w"), indent=2)

def f(k): m, lo, hi = out[k]; return f"{m} [{lo}, {hi}]"
print(f"ESCI human-Exact sets, {out['n_queries']} queries (>= {MIN} Exact products)")
for name in ["brand", "color"]:
    print(f"  {name}: coverage {f(name+'_coverage')} ; single-guess hard-exclusion {f(name+'_single_guess_exclusion')} ; median distinct values/query {out[name+'_distinct_values_per_query_median']:.0f}")
print("(compare WANDS category: coverage 0.93, single-guess exclusion 0.98)")
print("wrote", f"{RES}/esci_facet_exclusion.json")
