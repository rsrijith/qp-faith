#!/usr/bin/env python
"""
Score persisted planner outputs against the WANDS relevance-grounded harm label.
Reads results/<tag>__plans.jsonl (never re-runs inference). Deterministic.

Harm label (fix #2): an injected (spurious) attribute constraint is HARMFUL only
on POPULATED-AND-CONFLICTING facets -- an Exact-graded product that HAS the facet
populated AND whose value conflicts with the injected value. A merely-missing
facet is NOT harm. facet-coverage is reported as the confound bound.
"""
import os, sys, json, re, collections, argparse
import numpy as np
import pandas as pd

SEED = 13
DATA = os.path.join(os.path.dirname(__file__), "..", "data")
RES = os.path.join(os.path.dirname(__file__), "..", "results")

SLOT_FACETS = {
    "color":    ["color"],
    "material": ["primarymaterial", "material"],
    "style":    ["dsprimaryproductstyle", "style", "dssecondaryproductstyle"],
}
STOP = set("a an the of for with and or in on to set sets piece pieces".split())

def norm(s): return re.sub(r"[^a-z0-9 ]", " ", str(s).lower()).strip()
def words(s): return [w for w in norm(s).split() if w and w not in STOP]

def parse_features(s):
    d = collections.defaultdict(set)
    if not isinstance(s, str): return d
    for part in s.split("|"):
        if ":" not in part: continue
        k, v = part.split(":", 1)
        k = k.strip().lower(); v = norm(v)
        if k and v: d[k].add(v)
    return d

def value_matches(plan_val, prod_vals):
    pv = norm(plan_val)
    if not pv: return False
    return any(pv == v or pv in v or v in pv for v in prod_vals)

def grounded_in_query(plan_val, qw, qn):
    pv = norm(plan_val)
    if not pv: return True
    if pv in qn: return True
    return any(w in qw for w in pv.split() if w not in STOP)

def bootstrap_ci(vals, n=2000, seed=SEED):
    vals = np.asarray(vals, float)
    if len(vals) == 0: return (float("nan"),)*3
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(vals), size=(n, len(vals)))
    s = vals[idx].mean(1)
    return float(vals.mean()), float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    p = pd.read_csv(f"{DATA}/product.csv", sep="\t")
    lab = pd.read_csv(f"{DATA}/label.csv", sep="\t")
    exact = lab[lab.label == "Exact"]
    exact_by_q = exact.groupby("query_id")["product_id"].apply(list).to_dict()
    pf = p.set_index("product_id")
    feat = {pid: parse_features(pf.at[pid, "product_features"]) if pid in pf.index
            else collections.defaultdict(set) for pid in exact.product_id.unique()}
    # product_class column: single authoritative category per product, always
    # populated -> a clean harm slot (no facet-coverage confound).
    pclass = {pid: norm(pf.at[pid, "product_class"]) if pid in pf.index and pd.notna(pf.at[pid, "product_class"]) else ""
              for pid in exact.product_id.unique()}

    plans = [json.loads(l) for l in open(f"{RES}/{args.tag}__plans.jsonl")]
    rows = []
    for r in plans:
        qid = r["query_id"]; qn = norm(r["query"]); qw = set(words(r["query"]))
        plan = r["plan"]; pids = exact_by_q.get(qid, []); n_ex = len(pids)
        # product_type harm vs the authoritative product_class (always populated)
        if "product_type" in plan and n_ex:
            val = plan["product_type"]
            spur = not grounded_in_query(val, qw, qn)
            conflicting = sum(1 for pid in pids if not value_matches(val, {pclass[pid]} if pclass[pid] else set()))
            populated = n_ex  # product_class always populated
            rows.append(dict(query_id=qid, query=r["query"], slot="product_type", value=val,
                             spurious=int(spur), n_exact=n_ex, populated=populated,
                             conflicting=conflicting, facet_coverage=1.0,
                             harmful=int(spur and conflicting > 0),
                             excl_frac=(conflicting/n_ex if n_ex else 0.0)))
        for slot, facets in SLOT_FACETS.items():
            if slot not in plan: continue
            val = plan[slot]
            spur = not grounded_in_query(val, qw, qn)
            populated = conflicting = 0
            for pid in pids:
                fd = feat.get(pid, {})
                pv = set().union(*[fd.get(f, set()) for f in facets]) if any(f in fd for f in facets) else set()
                if pv:
                    populated += 1
                    if not value_matches(val, pv): conflicting += 1
            rows.append(dict(query_id=qid, query=r["query"], slot=slot, value=val,
                             spurious=int(spur), n_exact=n_ex, populated=populated,
                             conflicting=conflicting,
                             facet_coverage=(populated/n_ex if n_ex else 0.0),
                             harmful=int(spur and conflicting > 0),
                             excl_frac=(conflicting/n_ex if n_ex else 0.0)))
    df = pd.DataFrame(rows)
    df.to_parquet(f"{RES}/{args.tag}__scored.parquet")

    qids = sorted(set(r["query_id"] for r in plans)); nq = len(qids)
    byq = df.groupby("query_id")
    spur_per_q = [int(byq.get_group(g).spurious.sum()) if g in byq.groups else 0 for g in qids]
    has_spur = [int(s > 0) for s in spur_per_q]
    sp = df[df.spurious == 1]
    harmful_of_spur = sp.harmful.tolist()
    cov_of_spur = sp.facet_coverage.tolist()
    cwp = [(r.conflicting / r.populated) for r in sp.itertuples() if r.populated > 0]
    excl = [float(byq.get_group(g).excl_frac.max()) if g in byq.groups else 0.0 for g in qids]
    # also: harm restricted to spurious WITH a populated facet (so coverage can't mask it)
    sp_pop = sp[sp.populated > 0]
    harm_when_testable = sp_pop.harmful.tolist()

    rep = dict(model=args.tag, n_queries=nq, harm_slots=list(SLOT_FACETS),
               plan_slots_emitted=int(len(df)), total_spurious=int(df.spurious.sum()),
               frac_q_with_spurious=bootstrap_ci(has_spur),
               spurious_per_query=bootstrap_ci(spur_per_q),
               harmful_SCR_of_all_spurious=bootstrap_ci([float(x) for x in harmful_of_spur]) if harmful_of_spur else None,
               harmful_SCR_when_facet_populated=bootstrap_ci([float(x) for x in harm_when_testable]) if harm_when_testable else None,
               facet_coverage_confound=bootstrap_ci(cov_of_spur) if cov_of_spur else None,
               conflict_rate_within_populated=bootstrap_ci(cwp) if cwp else None,
               exact_recall_loss_at_plan=bootstrap_ci(excl),
               n_spurious_with_populated_facet=int(len(sp_pop)), seed=SEED)
    # per-slot spurious breakdown
    rep["per_slot"] = {s: dict(emitted=int((df.slot==s).sum()),
                               spurious=int(df[(df.slot==s)].spurious.sum()),
                               harmful=int(df[(df.slot==s)].harmful.sum()))
                       for s in sorted(df.slot.unique())}
    json.dump(rep, open(f"{RES}/{args.tag}__metrics.json", "w"), indent=2)

    def f(name, ci): return f"  {name:38s} {ci[0]:.3f}  [{ci[1]:.3f}, {ci[2]:.3f}]" if ci else f"  {name:38s} n/a"
    print(f"\n=== {args.tag}  (n={nq} queries) ===")
    print(f"  harm-bearing plan slots emitted: {len(df)}   spurious: {df.spurious.sum()}   spurious w/ populated facet: {len(sp_pop)}")
    print(f("frac queries w/ >=1 spurious", rep["frac_q_with_spurious"]))
    print(f("spurious per query", rep["spurious_per_query"]))
    print(f("Harmful-SCR (of ALL spurious)", rep["harmful_SCR_of_all_spurious"]))
    print(f("Harmful-SCR (when facet populated)", rep["harmful_SCR_when_facet_populated"]))
    print(f("facet-coverage (confound)", rep["facet_coverage_confound"]))
    print(f("conflict-rate | populated", rep["conflict_rate_within_populated"]))
    print(f("Exact-recall-loss@plan", rep["exact_recall_loss_at_plan"]))
    print("  per-slot (emitted/spurious/harmful):", {s: (v['emitted'], v['spurious'], v['harmful']) for s,v in rep["per_slot"].items()})

if __name__ == "__main__":
    main()
