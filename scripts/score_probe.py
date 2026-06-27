#!/usr/bin/env python
"""Score planner outputs on the procedural probe. Labels are KNOWN BY CONSTRUCTION:
- spurious = a plan slot NOT in the query's known_slots (for A_entity, known_slots={},
  so any product_type is provably spurious — no holistic-relevance guessing).
- harm = an injected product_type excludes relevant_pids (relevant set defined by the
  generative seed, not WANDS grades), measured against product_class.
Reports the clean headline: on type-less entity queries, how often does the planner
FORCE a product_type, and what fraction of the (multi-class) relevant set does it cut?"""
import os, re, json, collections, argparse
import numpy as np, pandas as pd

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
RES = os.path.join(os.path.dirname(__file__), "..", "results")
def norm(s): return re.sub(r"[^a-z0-9 ]", " ", str(s).lower()).strip()
def depl(w): return w[:-1] if len(w) > 3 and w.endswith("s") else w
def tokset(s): return {depl(w) for w in norm(s).split() if w}
def matches(a, b):
    A, B = tokset(a), tokset(b)
    return bool(A) and (A == B or A <= B or B <= A)

def cluster_boot(per_q, n=2000, seed=13):
    g = [x for x in per_q if len(x)]
    if not g: return (float("nan"),)*3
    rng = np.random.default_rng(seed)
    st = [np.mean([v for qi in rng.integers(0, len(g), len(g)) for v in g[qi]]) for _ in range(n)]
    return float(np.mean([v for x in g for v in x])), float(np.percentile(st,2.5)), float(np.percentile(st,97.5))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", required=True); args = ap.parse_args()
    p = pd.read_csv(f"{DATA}/product.csv", sep="\t")
    pclass = {int(r.product_id): norm(r.product_class) for r in p.itertuples() if pd.notna(r.product_class)}
    rows = [json.loads(l) for l in open(f"{RES}/{args.tag}__probe_plans.jsonl")]

    rec = []
    for r in rows:
        fam = r["family"]; known = r["known_slots"]; plan = r["plan"]; rel = r["relevant_pids"]
        for slot, val in plan.items():
            slot = slot.lower()
            if slot not in ("product_type", "color", "material", "style", "brand"): continue
            kv = known.get(slot)
            spurious = (kv is None) or (not matches(val, kv))
            harm_excl = None
            if slot == "product_type" and rel:
                excl = sum(1 for pid in rel if not matches(val, pclass.get(pid, "")))
                harm_excl = excl / len(rel)
            rec.append(dict(query=r["query"], family=fam, slot=slot, value=val,
                            spurious=int(spurious), n_relevant=len(rel),
                            harm_excl_frac=harm_excl,
                            distinct_class=r["n_distinct_class"]))
    df = pd.DataFrame(rec)
    df.to_parquet(f"{RES}/{args.tag}__probe_scored.parquet")

    qs = [r["query"] for r in rows]
    A = [r for r in rows if r["family"] == "A_entity"]
    # headline A: fraction of type-less entity queries where planner FORCES a product_type
    forced = [int("product_type" in r["plan"]) for r in A]
    # of those, mean fraction of relevant set excluded
    pt_rows = df[(df.family=="A_entity") & (df.slot=="product_type")]
    excl_vals = pt_rows.harm_excl_frac.dropna().tolist()
    # B: over-injection rate (slots beyond known)
    B = [r for r in rows if r["family"] == "B_attr"]
    B_over = [sum(1 for s in r["plan"] if s.lower() not in r["known_slots"] and s.lower() in ("color","material","style","brand")) for r in B]

    def boot1(x,n=2000,seed=13):
        if not x: return (float("nan"),)*3
        x=np.array(x,float); rng=np.random.default_rng(seed)
        s=x[rng.integers(0,len(x),(n,len(x)))].mean(1)
        return float(x.mean()),float(np.percentile(s,2.5)),float(np.percentile(s,97.5))

    rep = dict(tag=args.tag, n_A=len(A), n_B=len(B),
               A_product_type_forced_rate=boot1(forced),
               A_mean_relevant_excluded_when_typed=boot1(excl_vals),
               A_n_typed=len(excl_vals),
               B_extra_slot_injections=int(sum(B_over)),
               B_over_injection_rate=boot1([int(x>0) for x in B_over]))
    json.dump(rep, open(f"{RES}/{args.tag}__probe_metrics.json","w"), indent=2, default=float)
    def f(n,c): return f"  {n:46s} {c[0]:.3f} [{c[1]:.3f},{c[2]:.3f}]"
    print(f"\n=== PROBE {args.tag}  (A_entity={len(A)}, B_attr={len(B)}) ===")
    print(f("A: type-less query gets a FORCED product_type", rep["A_product_type_forced_rate"]))
    print(f("A: mean relevant-set excluded when typed", rep["A_mean_relevant_excluded_when_typed"]))
    print(f("B: query gets an EXTRA (spurious) attribute", rep["B_over_injection_rate"]))
    print("  (A labels are spurious BY CONSTRUCTION: entity queries express no product_type)")

if __name__ == "__main__":
    main()
