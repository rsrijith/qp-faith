#!/usr/bin/env python
"""Score planner plans on the ESCI slice. Brand/color facets only (the only
populated structured fields in public ESCI). Harm restricted to SPURIOUS
injections (not grounded in the query) that exclude an Exact-graded product whose
brand/color is POPULATED and CONFLICTING. Cluster bootstrap. Brand is primary
(color is noisy in ESCI). Mirrors score_v2 semantics."""
import os, re, json, collections, argparse
import numpy as np, pandas as pd

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
RES = os.path.join(os.path.dirname(__file__), "..", "results")
STOP = set("a an the of for with and or in on to set sets piece pieces that have by your our".split())
def norm(s): return re.sub(r"[^a-z0-9 ]", " ", str(s).lower()).strip()
def depl(w): return w[:-1] if len(w) > 3 and w.endswith("s") else w
def tokset(s): return {depl(w) for w in norm(s).split() if w and w not in STOP}
def matches(a, b):
    A, B = tokset(a), tokset(b)
    return bool(A) and bool(B) and (A == B or A <= B or B <= A)
def grounded(val, qset):
    pv = tokset(val); return (not pv) or bool(pv & qset)

def cluster_boot(per_q, n=2000, seed=13):
    g = [x for x in per_q if len(x)]
    if not g: return (float("nan"),)*3
    rng = np.random.default_rng(seed)
    st = [np.mean([v for qi in rng.integers(0, len(g), len(g)) for v in g[qi]]) for _ in range(n)]
    return float(np.mean([v for x in g for v in x])), float(np.percentile(st, 2.5)), float(np.percentile(st, 97.5))

SLOTS = {"brand": "product_brand", "color": "product_color"}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", required=True); a = ap.parse_args()
    pool = pd.read_parquet(f"{DATA}/esci/pool.parquet")
    exact = pool[pool.esci_label == "Exact"]
    exact_by_q = {qid: g for qid, g in exact.groupby("query_id")}
    plans = [json.loads(l) for l in open(f"{RES}/esci-{a.tag}__plans.jsonl")]

    rows = []
    for r in plans:
        qid = r["query_id"]; qset = tokset(r["query"]); plan = r["plan"]
        grp = exact_by_q.get(qid)
        if grp is None or len(grp) == 0: continue
        for slot, col in SLOTS.items():
            if slot not in plan: continue
            val = plan[slot]; spur = not grounded(val, qset)
            vals = grp[col].dropna().astype(str)
            populated = len(vals)
            conflicting = sum(1 for v in vals if not matches(val, v))
            rows.append(dict(query_id=qid, slot=slot, value=val, spurious=int(spur),
                             n_exact=len(grp), populated=populated, conflicting=conflicting,
                             facet_coverage=(populated/len(grp) if len(grp) else 0),
                             harmful=int(spur and conflicting > 0)))
    df = pd.DataFrame(rows)
    if df.empty:
        print(f"esci-{a.tag}: no brand/color slots emitted"); return
    df.to_parquet(f"{RES}/esci-{a.tag}__scored.parquet")
    qids = sorted({r["query_id"] for r in plans})
    def per_q(mask):
        g = {q: [] for q in qids}
        for r in df[mask].itertuples(): g[r.query_id].append(1.0)
        return [g[q] for q in qids if g[q]]
    has_spur = [[1.0] if df[(df.query_id==q)].spurious.sum()>0 else [0.0] for q in qids]
    sp = df[df.spurious==1]
    harm_g = {q: [] for q in qids}
    for r in sp.itertuples(): harm_g[r.query_id].append(float(r.harmful))
    cwp_g = {q: [] for q in qids}
    for r in sp[sp.populated>0].itertuples(): cwp_g[r.query_id].append(r.conflicting/r.populated)
    cov_g = {q: [] for q in qids}
    for r in sp.itertuples(): cov_g[r.query_id].append(r.facet_coverage)
    rep = dict(tag=a.tag, n_queries=len(qids), slots=int(len(df)), spurious=int(df.spurious.sum()),
               harmful=int(df.harmful.sum()),
               frac_q_spurious=cluster_boot(has_spur),
               harmful_SCR=cluster_boot([harm_g[q] for q in qids if harm_g[q]]),
               conflict_within_pop=cluster_boot([cwp_g[q] for q in qids if cwp_g[q]]),
               coverage=cluster_boot([cov_g[q] for q in qids if cov_g[q]]),
               per_slot={s: dict(emitted=int((df.slot==s).sum()), spurious=int(df[df.slot==s].spurious.sum()),
                                 harmful=int(df[df.slot==s].harmful.sum())) for s in SLOTS})
    json.dump(rep, open(f"{RES}/esci-{a.tag}__metrics.json", "w"), indent=2, default=float)
    def f(n, c): return f"  {n:28s} {c[0]:.3f} [{c[1]:.2f},{c[2]:.2f}]"
    print(f"\n=== ESCI {a.tag} (n={len(qids)}) brand+color ===")
    print(f"  slots {len(df)} spurious {df.spurious.sum()} harmful {df.harmful.sum()}")
    print(f("frac-q spurious", rep["frac_q_spurious"]))
    print(f("Harmful-SCR", rep["harmful_SCR"]))
    print(f("conflict|populated", rep["conflict_within_pop"]))
    print(f("coverage (confound)", rep["coverage"]))
    print("  per-slot:", {s:(v['emitted'],v['spurious'],v['harmful']) for s,v in rep["per_slot"].items()})

if __name__ == "__main__":
    main()
