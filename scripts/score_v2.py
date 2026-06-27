#!/usr/bin/env python
"""
score_v2 — re-scores persisted plans with the construct-validity fixes from the
2026-06-24 measurement-validity review. Reads results/<tag>__plans.jsonl.

Fixes vs score.py:
 1. Balanced-brace JSON re-extraction from the raw output + a per-model
    PLAN-PARSE-FAILURE RATE (sloppy-JSON models otherwise look falsely faithful).
 2. Token-set + synonym matching instead of naive substring (kills gold~marigold,
    table~tablecloth false-matches); plural/singular normalization.
 3. MULTI-CLASS Exact-set split for product_type harm: distinct product_class
    count + Shannon entropy, separating genuine wrong-class harm from
    heterogeneous-Exact-set artifacts (no single filter could retrieve them).
 4. Per-query CLUSTER bootstrap for every metric (per-row bootstrap was
    anti-conservative because a query emits several correlated rows).
Still deterministic. Hard-filter harm only here; soft-penalty/retriever
validation lives in score_soft.py (needs a ranker).
"""
import os, sys, re, json, math, collections, argparse
import numpy as np
import pandas as pd

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
RES = os.path.join(os.path.dirname(__file__), "..", "results")
SLOT_FACETS = {"color": ["color"], "material": ["primarymaterial", "material"],
               "style": ["dsprimaryproductstyle", "style", "dssecondaryproductstyle"]}
STOP = set("a an the of for with and or in on to set sets piece pieces that have".split())

# small curated synonym/equivalence map (closed-vocab; expand with the gold)
SYN = {
    "grey": "gray", "gray": "gray", "couch": "sofa", "sofa": "sofa",
    "rug": "rug", "carpet": "rug", "pillow": "pillow", "cushion": "pillow",
    "wood": "wood", "wooden": "wood", "timber": "wood",
    "metal": "metal", "metallic": "metal", "steel": "steel",
}
def syn(w): return SYN.get(w, w)
def depl(w): return w[:-1] if len(w) > 3 and w.endswith("s") else w  # crude singularize

def norm(s): return re.sub(r"[^a-z0-9 ]", " ", str(s).lower()).strip()
def toks(s): return [depl(syn(w)) for w in norm(s).split() if w and w not in STOP]
def tokset(s): return set(toks(s))

def balanced_json(text):
    """Extract the first balanced {...} (handles nesting/markdown). Returns (dict,parsed_ok)."""
    if not isinstance(text, str): return {}, False
    i = text.find("{")
    while i != -1:
        depth = 0
        for j in range(i, len(text)):
            if text[j] == "{": depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[i:j+1])
                        if isinstance(obj, dict):
                            flat = {k: v for k, v in obj.items() if isinstance(v, (str, int, float)) and str(v).strip()}
                            return {k: str(v) for k, v in flat.items()}, True
                    except Exception:
                        break
        i = text.find("{", i + 1)
    return {}, False

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
    """token-set match: plan tokens are a subset of a product value's tokens
    (after synonym+plural norm), OR exact set equality. Stricter than substring."""
    pset = tokset(plan_val)
    if not pset: return False
    for v in prod_vals:
        vset = tokset(v)
        if pset == vset or pset <= vset or vset <= pset:
            return True
    return False

def grounded(plan_val, qset):
    pv = tokset(plan_val)
    return (not pv) or bool(pv & qset)

def cluster_boot(per_query_vals, n=2000, seed=13):
    """per-query cluster bootstrap. per_query_vals: list of lists (one per query)."""
    flat_mean = lambda groups: np.mean([x for g in groups for x in g]) if any(groups) else float("nan")
    groups = [g for g in per_query_vals if len(g) > 0]
    if not groups: return (float("nan"),)*3
    rng = np.random.default_rng(seed)
    qs = np.arange(len(groups))
    stats = []
    for _ in range(n):
        samp = rng.integers(0, len(groups), size=len(groups))
        vals = [x for qi in samp for x in groups[qi]]
        if vals: stats.append(np.mean(vals))
    pt = flat_mean(groups)
    return float(pt), float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", required=True); args = ap.parse_args()
    p = pd.read_csv(f"{DATA}/product.csv", sep="\t")
    lab = pd.read_csv(f"{DATA}/label.csv", sep="\t")
    exact = lab[lab.label == "Exact"]
    exact_by_q = exact.groupby("query_id")["product_id"].apply(list).to_dict()
    pf = p.set_index("product_id")
    feat = {pid: parse_features(pf.at[pid, "product_features"]) if pid in pf.index else {} for pid in exact.product_id.unique()}
    pclass = {pid: norm(pf.at[pid, "product_class"]) if pid in pf.index and pd.notna(pf.at[pid, "product_class"]) else "" for pid in exact.product_id.unique()}

    plans = [json.loads(l) for l in open(f"{RES}/{args.tag}__plans.jsonl")]
    n_parse_fail = 0
    rows = []
    for r in plans:
        qid = r["query_id"]; qset = tokset(r["query"])
        # re-extract from raw with the balanced parser; fall back to stored plan
        plan2, ok = balanced_json(r.get("raw", ""))
        if not ok and not r.get("plan"): n_parse_fail += 1
        plan = plan2 if ok else r.get("plan", {})
        pids = exact_by_q.get(qid, []); n_ex = len(pids)
        # multi-class entropy of the Exact set
        classes = [pclass[pid] for pid in pids if pclass.get(pid)]
        cc = collections.Counter(classes); distinct = len(cc)
        tot = sum(cc.values()) or 1
        ent = -sum((c/tot)*math.log2(c/tot) for c in cc.values()) if cc else 0.0

        if "product_type" in plan and n_ex:
            val = plan["product_type"]; spur = not grounded(val, qset)
            conflicting = sum(1 for pid in pids if not value_matches(val, {pclass[pid]} if pclass.get(pid) else set()))
            rows.append(dict(query_id=qid, query=r["query"], slot="product_type", value=val,
                             spurious=int(spur), n_exact=n_ex, populated=n_ex, conflicting=conflicting,
                             facet_coverage=1.0, harmful=int(spur and conflicting > 0),
                             excl_frac=conflicting/n_ex, exact_distinct_class=distinct, exact_class_entropy=ent,
                             clean_harm=int(spur and conflicting > 0 and distinct == 1)))
        for slot, facets in SLOT_FACETS.items():
            if slot not in plan: continue
            val = plan[slot]; spur = not grounded(val, qset)
            populated = conflicting = 0
            for pid in pids:
                fd = feat.get(pid, {})
                pv = set().union(*[fd.get(f, set()) for f in facets]) if any(f in fd for f in facets) else set()
                if pv:
                    populated += 1
                    if not value_matches(val, pv): conflicting += 1
            rows.append(dict(query_id=qid, query=r["query"], slot=slot, value=val, spurious=int(spur),
                             n_exact=n_ex, populated=populated, conflicting=conflicting,
                             facet_coverage=(populated/n_ex if n_ex else 0.0),
                             harmful=int(spur and conflicting > 0), excl_frac=(conflicting/n_ex if n_ex else 0.0),
                             exact_distinct_class=distinct, exact_class_entropy=ent,
                             clean_harm=int(spur and conflicting > 0 and slot == "product_type")))
    df = pd.DataFrame(rows)
    df.to_parquet(f"{RES}/{args.tag}__scored_v2.parquet")

    qids = sorted(set(r["query_id"] for r in plans))
    # per-query grouped values for cluster bootstrap
    def per_q(colfilter):
        g = {q: [] for q in qids}
        for r in df[colfilter].itertuples():
            g[r.query_id].append(1.0)
        # queries with no qualifying row contribute an empty group (excluded)
        return [g[q] for q in qids if g[q]]
    # frac queries with >=1 spurious: per-query indicator (cluster trivially = bernoulli)
    has_spur = [[1.0] if (df[(df.query_id==q)].spurious.sum()>0) else [0.0] for q in qids]
    sp = df[df.spurious==1]
    # harmful among spurious, clustered by query
    harm_groups = {q: [] for q in qids}
    for r in sp.itertuples(): harm_groups[r.query_id].append(float(r.harmful))
    harm_pq = [harm_groups[q] for q in qids if harm_groups[q]]
    # conflict-rate within populated, clustered
    cwp_groups = {q: [] for q in qids}
    for r in sp[sp.populated>0].itertuples(): cwp_groups[r.query_id].append(r.conflicting/r.populated)
    cwp_pq = [cwp_groups[q] for q in qids if cwp_groups[q]]
    cov_groups = {q: [] for q in qids}
    for r in sp.itertuples(): cov_groups[r.query_id].append(r.facet_coverage)
    cov_pq = [cov_groups[q] for q in qids if cov_groups[q]]

    pt = df[df.slot=="product_type"]; pth = pt[pt.harmful==1]
    rep = dict(tag=args.tag, n_queries=len(qids), parse_fail=n_parse_fail,
               parse_fail_rate=round(n_parse_fail/len(plans),4),
               total_slots=int(len(df)), total_spurious=int(df.spurious.sum()), total_harmful=int(df.harmful.sum()),
               frac_q_with_spurious=cluster_boot(has_spur),
               harmful_SCR_of_spurious=cluster_boot(harm_pq),
               conflict_rate_within_populated=cluster_boot(cwp_pq),
               facet_coverage_confound=cluster_boot(cov_pq),
               product_type_harmful=int(pth.harmful.sum()),
               product_type_clean_harm_singleclass=int(pth.clean_harm.sum()),
               product_type_harm_multiclass_artifact=int((pth.exact_distinct_class>1).sum()),
               per_slot={s: dict(emitted=int((df.slot==s).sum()), spurious=int(df[df.slot==s].spurious.sum()),
                                 harmful=int(df[df.slot==s].harmful.sum())) for s in sorted(df.slot.unique())})
    json.dump(rep, open(f"{RES}/{args.tag}__metrics_v2.json","w"), indent=2, default=float)
    def f(n,ci): return f"  {n:34s} {ci[0]:.3f} [{ci[1]:.3f},{ci[2]:.3f}]"
    print(f"\n=== {args.tag} (v2, n={len(qids)}) parse-fail={n_parse_fail} ({rep['parse_fail_rate']*100:.1f}%) ===")
    print(f"  slots {len(df)}  spurious {df.spurious.sum()}  harmful {df.harmful.sum()}")
    print(f("frac q w/ spurious (clustered)", rep["frac_q_with_spurious"]))
    print(f("Harmful-SCR of spurious", rep["harmful_SCR_of_spurious"]))
    print(f("conflict-rate|populated", rep["conflict_rate_within_populated"]))
    print(f("facet-coverage confound", rep["facet_coverage_confound"]))
    print(f"  product_type harmful={rep['product_type_harmful']}  "
          f"clean(single-class)={rep['product_type_clean_harm_singleclass']}  "
          f"multiclass-artifact={rep['product_type_harm_multiclass_artifact']}")
    print("  per-slot:", {s:(v['emitted'],v['spurious'],v['harmful']) for s,v in rep["per_slot"].items()})

if __name__ == "__main__":
    main()
