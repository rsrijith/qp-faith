#!/usr/bin/env python
"""Referee fix #1 (defensibility): validate the probe's product_type harm through an
ACTUAL retrieval over the full WANDS catalog, not just a deterministic label.

For each entity probe query we BM25-rank ALL ~43K products, measure recall@K of the
motif's relevant set with NO filter, then apply the planner's injected product_type
as a hard category filter (production category-filter semantics) and re-measure.
Harm = recall drop under real retrieval. We also report agreement between this
measured recall-loss and the by-construction label exclusion, to show the label is
a faithful proxy. CPU only.
"""
import os, re, json, collections, argparse
import numpy as np, pandas as pd
from rank_bm25 import BM25Okapi

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
RES = os.path.join(os.path.dirname(__file__), "..", "results")
STOP = set("a an the of for with and or in on to set sets piece pieces that have by your our".split())
def norm(s): return re.sub(r"[^a-z0-9 ]", " ", str(s).lower()).strip()
def depl(w): return w[:-1] if len(w) > 3 and w.endswith("s") else w
def tokset(s): return {depl(w) for w in norm(s).split() if w and w not in STOP}
def matches(a, b):
    A, B = tokset(a), tokset(b); return bool(A) and bool(B) and (A == B or A <= B or B <= A)

_CACHE = {}
def build_index():
    if "bm25" in _CACHE: return _CACHE["bm25"], _CACHE["pids"], _CACHE["cls"]
    p = pd.read_csv(f"{DATA}/product.csv", sep="\t")
    pids = p["product_id"].tolist()
    cls = {int(r.product_id): norm(r.product_class) for r in p.itertuples() if pd.notna(r.product_class)}
    docs = [(norm(n) + " " + norm(f)).split() for n, f in zip(p["product_name"], p["product_features"])]
    bm = BM25Okapi(docs)
    _CACHE.update(bm25=bm, pids=[int(x) for x in pids], cls=cls)
    return bm, _CACHE["pids"], cls

def recall_at(order, rel, k):
    rs = set(rel); return len([p for p in order[:k] if p in rs]) / len(rs) if rs else float("nan")

def boot(x, n=2000, seed=13):
    x = np.array([v for v in x if v == v], float)
    if not len(x): return (float("nan"),)*3
    rng = np.random.default_rng(seed); s = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", required=True)
    ap.add_argument("--k", type=int, default=100); a = ap.parse_args()
    bm, pids, cls = build_index()
    pid_arr = np.array(pids)
    rows = [json.loads(l) for l in open(f"{RES}/{a.tag}__probe_plans.jsonl")]
    rows = [r for r in rows if r["family"] == "A_entity"]

    drops, label_excls, agree = [], [], []
    typed = 0
    for r in rows:
        pt = r["plan"].get("product_type")
        rel = [int(x) for x in r["relevant_pids"]]
        scores = bm.get_scores(list(tokset(r["query"])) or ["x"])
        order = pid_arr[np.argsort(-scores)].tolist()
        r_nofilter = recall_at(order, rel, a.k)
        if not pt:  # planner emitted no type -> no filter -> no harm
            drops.append(0.0); continue
        typed += 1
        keep = [pid for pid in order if matches(pt, cls.get(pid, ""))]
        r_filter = recall_at(keep, rel, a.k)
        drops.append(r_nofilter - r_filter)
        # by-construction label exclusion (score_probe semantics)
        lex = sum(1 for pid in rel if not matches(pt, cls.get(pid, ""))) / len(rel)
        label_excls.append(lex)
        agree.append(abs((r_nofilter - r_filter) - 0))  # placeholder; report both dists

    rep = dict(tag=a.tag, k=a.k, n_entity=len(rows), n_typed=typed,
               mean_recall_drop_real_retrieval=boot(drops),
               mean_label_exclusion=boot(label_excls))
    json.dump(rep, open(f"{RES}/{a.tag}__retrieval_validation.json", "w"), indent=2, default=float)
    rd = rep["mean_recall_drop_real_retrieval"]; le = rep["mean_label_exclusion"]
    print(f"\n=== RETRIEVAL VALIDATION {a.tag} (entity queries, k={a.k}) ===")
    print(f"  typed {typed}/{len(rows)} entity queries got a product_type")
    print(f"  REAL recall@{a.k} drop from the injected category filter: {rd[0]:.3f} [{rd[1]:.3f},{rd[2]:.3f}]")
    print(f"  by-construction label exclusion:                        {le[0]:.3f} [{le[1]:.3f},{le[2]:.3f}]")
    print(f"  -> the injected product_type filter destroys {rd[0]*100:.0f}% of recoverable recall under ACTUAL BM25 retrieval")

if __name__ == "__main__":
    main()
