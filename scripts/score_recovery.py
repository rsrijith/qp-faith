#!/usr/bin/env python
"""
score_recovery — the actionable hook (the mitigation method). Shows the spurious
product_type injection is both HARMFUL and FIXABLE with a trivial detector.

For each WANDS query we BM25-rank its graded candidate pool and evaluate three
policies, reporting nDCG@10 (grades as gains) and Exact-recall@10:
  NOFILTER  — rank the pool, ignore the plan.
  FILTER    — apply the planner's product_type as a hard category filter, then rank.
  GATED     — apply the filter ONLY when a cheap detector says the query is
              CONCRETE (head token is a known product noun); on ambiguous/entity
              queries, suppress the injected product_type.

Claim to support: on AMBIGUOUS queries FILTER craters recall and GATED restores it;
on CONCRETE queries FILTER helps (removes irrelevants) and GATED keeps that benefit.
So the fix recovers harm without giving up legitimate filtering. Per-query cluster
bootstrap CIs.
"""
import os, re, json, math, collections, argparse
import numpy as np, pandas as pd
from rank_bm25 import BM25Okapi

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
RES = os.path.join(os.path.dirname(__file__), "..", "results")
GAIN = {"Exact": 2.0, "Partial": 1.0, "Irrelevant": 0.0}
STOP = set("a an the of for with and or in on to set sets piece pieces that have by your our".split())
def norm(s): return re.sub(r"[^a-z0-9 ]", " ", str(s).lower()).strip()
def depl(w): return w[:-1] if len(w) > 3 and w.endswith("s") else w
def tokset(s): return {depl(w) for w in norm(s).split() if w and w not in STOP}
def matches(a, b):
    A, B = tokset(a), tokset(b)
    return bool(A) and (A == B or A <= B or B <= A)
def parse_features(s):
    d = collections.defaultdict(set)
    if isinstance(s, str):
        for part in s.split("|"):
            if ":" in part:
                k, v = part.split(":", 1); k = k.strip().lower(); v = norm(v)
                if k and v: d[k].add(v)
    return d
def ndcg(order, gains, k=10):
    dcg = sum(gains.get(p, 0)/math.log2(i+2) for i, p in enumerate(order[:k]))
    ideal = sorted(gains.values(), reverse=True)
    idcg = sum(g/math.log2(i+2) for i, g in enumerate(ideal[:k]))
    return dcg/idcg if idcg > 0 else 0.0
def exact_recall_at(order, gains, k=10):
    ex = [p for p, g in gains.items() if g == 2.0]
    if not ex: return None
    return len([p for p in order[:k] if gains.get(p) == 2.0]) / len(ex)

def cluster_boot_pairs(vals, n=2000, seed=13):
    v = np.array([x for x in vals if x is not None], float)
    if len(v) == 0: return (float("nan"),)*3
    rng = np.random.default_rng(seed)
    s = v[rng.integers(0, len(v), (n, len(v)))].mean(1)
    return float(v.mean()), float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    q = pd.read_csv(f"{DATA}/query.csv", sep="\t")
    p = pd.read_csv(f"{DATA}/product.csv", sep="\t").set_index("product_id")
    lab = pd.read_csv(f"{DATA}/label.csv", sep="\t")
    lab_by_q = {qid: g[["product_id", "label"]].values.tolist() for qid, g in lab.groupby("query_id")}
    plans = {r["query_id"]: r["plan"] for r in (json.loads(l) for l in open(f"{RES}/{args.tag}__plans.jsonl"))}

    # detector vocab: product-class + producttype head nouns
    class_tokens = set()
    for c in p["product_class"].dropna(): class_tokens |= tokset(c)
    for s in p["product_features"].dropna():
        for part in str(s).split("|"):
            if part.lower().strip().startswith("producttype") and ":" in part:
                class_tokens |= tokset(part.split(":", 1)[1])
    def is_concrete(query):
        return any(t in class_tokens for t in tokset(query))

    out = collections.defaultdict(lambda: collections.defaultdict(list))  # seg -> policy -> list
    for r in q.itertuples():
        qid = int(r.query_id); pool = lab_by_q.get(qid, [])
        if len(pool) < 10: continue
        plan = plans.get(qid, {}); pt = plan.get("product_type")
        qwords = tokset(r.query); concrete = is_concrete(r.query)
        seg = "concrete" if concrete else "ambiguous"
        pids = [int(pid) for pid, _ in pool]
        gains = {int(pid): GAIN.get(lab_, 0.0) for pid, lab_ in pool}
        docs, cls = [], {}
        for pid in pids:
            try:
                docs.append((norm(p.at[pid, "product_name"]) + " " + norm(p.at[pid, "product_features"])).split())
                cls[pid] = norm(p.at[pid, "product_class"])
            except KeyError:
                docs.append(["x"]); cls[pid] = ""
        base = np.array(BM25Okapi(docs).get_scores(list(qwords) or ["x"]))
        order_nofilter = [pids[i] for i in np.argsort(-base)]

        def filtered_order():
            if not pt: return order_nofilter
            keep = [i for i, pid in enumerate(pids) if matches(pt, cls[pid])]
            if not keep: keep = list(range(len(pids)))  # empty filter -> degrade gracefully
            ks = sorted(keep, key=lambda i: -base[i])
            return [pids[i] for i in ks]
        order_filter = filtered_order()
        order_gated = order_filter if concrete else order_nofilter

        for pol, order in [("NOFILTER", order_nofilter), ("FILTER", order_filter), ("GATED", order_gated)]:
            out[seg][pol+"_ndcg"].append(ndcg(order, gains))
            out[seg][pol+"_recall"].append(exact_recall_at(order, gains))
            out["all"][pol+"_ndcg"].append(ndcg(order, gains))
            out["all"][pol+"_recall"].append(exact_recall_at(order, gains))

    rep = {"tag": args.tag}
    print(f"\n=== RECOVERY {args.tag} ===")
    print(f"{'segment':10s} {'metric':7s} {'NOFILTER':>20s} {'FILTER':>20s} {'GATED':>20s}")
    for seg in ["all", "concrete", "ambiguous"]:
        n = len([x for x in out[seg]['FILTER_ndcg']])
        rep[seg] = {"n": n}
        for metric in ["ndcg", "recall"]:
            cells = {}
            line = f"{seg:10s} {metric:7s} "
            for pol in ["NOFILTER", "FILTER", "GATED"]:
                ci = cluster_boot_pairs(out[seg][pol+"_"+metric])
                cells[pol] = ci
                line += f"{ci[0]:.3f}[{ci[1]:.2f},{ci[2]:.2f}] ".rjust(21)
            rep[seg][metric] = cells
            print(line)
    json.dump(rep, open(f"{RES}/{args.tag}__recovery.json", "w"), indent=2, default=float)
    # headline recovery numbers
    a = rep["ambiguous"]
    print(f"\n  AMBIGUOUS Exact-recall@10: FILTER {a['recall']['FILTER'][0]:.3f} -> GATED {a['recall']['GATED'][0]:.3f} "
          f"(recovered {a['recall']['GATED'][0]-a['recall']['FILTER'][0]:+.3f})")
    c = rep["concrete"]
    print(f"  CONCRETE nDCG@10:          NOFILTER {c['ndcg']['NOFILTER'][0]:.3f} -> FILTER {c['ndcg']['FILTER'][0]:.3f} "
          f"(filter benefit {c['ndcg']['FILTER'][0]-c['ndcg']['NOFILTER'][0]:+.3f}); GATED keeps it ({c['ndcg']['GATED'][0]:.3f})")

if __name__ == "__main__":
    main()
