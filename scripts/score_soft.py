#!/usr/bin/env python
"""
score_soft — the 4th validity fix: harm under a SOFT-penalty retrieval model, not
just a hard conjunctive filter. Addresses the methods reviewer's C1 ("real planners
use attributes as soft boosts, so hard-filter exclusion overstates harm").

For each WANDS query we build BM25 over its graded candidate pool (product text),
rank it, and measure nDCG@10 using WANDS grades as gains (Exact=2, Partial=1,
Irrelevant=0). Then we apply each model's plan as a SOFT penalty: products that
CONFLICT with a spurious injected attribute have their BM25 score multiplied by
LAMBDA (demoted, not deleted). Soft-harm = nDCG@10 drop.

This brackets the harm: hard-filter recall-loss is the upper bound (score.py),
soft-penalty nDCG-drop is the realistic-system lower bound (here).
"""
import os, re, json, math, collections, argparse
import numpy as np, pandas as pd
from rank_bm25 import BM25Okapi

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
RES = os.path.join(os.path.dirname(__file__), "..", "results")
GAIN = {"Exact": 2.0, "Partial": 1.0, "Irrelevant": 0.0}
LAMBDA = 0.5  # soft demotion factor for conflicting products
SLOT_FACETS = {"color": ["color"], "material": ["primarymaterial", "material"],
               "style": ["dsprimaryproductstyle", "style", "dssecondaryproductstyle"],
               "product_type": ["__class__"]}
def norm(s): return re.sub(r"[^a-z0-9 ]", " ", str(s).lower()).strip()
def depl(w): return w[:-1] if len(w) > 3 and w.endswith("s") else w
def tokset(s): return {depl(w) for w in norm(s).split() if w}
def matches(a, vals):
    A = tokset(a)
    return bool(A) and any(A == tokset(v) or A <= tokset(v) or tokset(v) <= A for v in vals)

def parse_features(s):
    d = collections.defaultdict(set)
    if not isinstance(s, str): return d
    for part in s.split("|"):
        if ":" not in part: continue
        k, v = part.split(":", 1)
        k = k.strip().lower(); v = norm(v)
        if k and v: d[k].add(v)
    return d

def ndcg(order, gains, k=10):
    dcg = sum(gains[p] / math.log2(i + 2) for i, p in enumerate(order[:k]))
    ideal = sorted(gains.values(), reverse=True)
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal[:k]))
    return dcg / idcg if idcg > 0 else 0.0

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", required=True)
    ap.add_argument("--max_q", type=int, default=480); args = ap.parse_args()
    q = pd.read_csv(f"{DATA}/query.csv", sep="\t")
    p = pd.read_csv(f"{DATA}/product.csv", sep="\t").set_index("product_id")
    lab = pd.read_csv(f"{DATA}/label.csv", sep="\t")
    lab_by_q = {qid: g[["product_id", "label"]].values.tolist() for qid, g in lab.groupby("query_id")}
    plans = {r["query_id"]: r["plan"] for r in (json.loads(l) for l in open(f"{RES}/{args.tag}__plans.jsonl"))}

    deltas = []
    for r in q.head(args.max_q).itertuples():
        qid = int(r.query_id)
        pool = lab_by_q.get(qid, [])
        if len(pool) < 10: continue
        plan = plans.get(qid, {})
        qwords = tokset(r.query)
        pids = [int(pid) for pid, _ in pool]
        gains = {int(pid): GAIN.get(lab_, 0.0) for pid, lab_ in pool}
        # BM25 over product text
        docs = []
        feats = {}
        for pid in pids:
            try:
                name = norm(p.at[pid, "product_name"]); pf = p.at[pid, "product_features"]
                cls = norm(p.at[pid, "product_class"])
            except KeyError:
                name, pf, cls = "", "", ""
            feats[pid] = (parse_features(pf), cls)
            docs.append((name + " " + norm(pf)).split())
        bm = BM25Okapi(docs)
        base = np.array(bm.get_scores(list(qwords) or ["x"]))
        # soft penalty: demote products conflicting with a SPURIOUS injected attr
        pen = np.ones(len(pids))
        for slot, facets in SLOT_FACETS.items():
            if slot not in plan: continue
            val = plan[slot]
            if qwords & tokset(val): continue  # grounded -> not spurious
            for i, pid in enumerate(pids):
                fd, cls = feats[pid]
                if slot == "product_type":
                    vals = {cls} if cls else set()
                else:
                    vals = set().union(*[fd.get(f, set()) for f in facets]) if any(f in fd for f in facets) else set()
                if vals and not matches(val, vals):
                    pen[i] *= LAMBDA
        order_base = [pids[i] for i in np.argsort(-base)]
        order_soft = [pids[i] for i in np.argsort(-(base * pen))]
        d = ndcg(order_base, gains) - ndcg(order_soft, gains)
        deltas.append(d)

    deltas = np.array(deltas)
    rng = np.random.default_rng(13)
    s = deltas[rng.integers(0, len(deltas), (2000, len(deltas)))].mean(1)
    rep = dict(tag=args.tag, n_queries=int(len(deltas)), lambda_=LAMBDA,
               mean_ndcg10_drop_soft=float(deltas.mean()),
               ci=[float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))],
               frac_queries_with_any_drop=float((deltas > 1e-6).mean()))
    json.dump(rep, open(f"{RES}/{args.tag}__soft_metrics.json", "w"), indent=2)
    print(f"\n=== SOFT {args.tag} (n={len(deltas)}, lambda={LAMBDA}) ===")
    print(f"  mean nDCG@10 drop (soft penalty): {deltas.mean():.4f}  "
          f"[{np.percentile(s,2.5):.4f}, {np.percentile(s,97.5):.4f}]")
    print(f"  queries with any drop: {(deltas>1e-6).mean():.3f}")
    print("  (compare to hard-filter Exact-recall-loss; soft is the realistic-system lower bound)")

if __name__ == "__main__":
    main()
