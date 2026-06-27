#!/usr/bin/env python
"""Second-domain probe scorer: forced-type rate + real-retrieval harm on the
Amazon/ESCI catalog. No product_class in ESCI, so the injected product_type is
applied as a TITLE-token category filter (keep products whose title contains the
type token) over a BM25 ranking of the full catalog. Recall@K drop = harm."""
import os, re, json, argparse
import numpy as np, pandas as pd
from rank_bm25 import BM25Okapi

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
RES = os.path.join(os.path.dirname(__file__), "..", "results")
STOP = set("a an the of for with and or in on to set sets by your our".split())
def norm(s): return re.sub(r"[^a-z0-9 ]", " ", str(s).lower()).strip()
def depl(w): return w[:-1] if len(w) > 3 and w.endswith("s") else w
def tokset(s): return {depl(w) for w in norm(s).split() if w and w not in STOP}

_C = {}
def index(catalog=None):
    if "bm" in _C: return _C["bm"], _C["pids"], _C["titles"]
    cat = pd.read_parquet(catalog or f"{DATA}/probe_esci/catalog.parquet")
    pids = cat["product_id"].tolist(); titles = cat["ntitle"].tolist()
    _C.update(bm=BM25Okapi([t.split() for t in titles]), pids=pids,
              titles={pids[i]: titles[i] for i in range(len(pids))})
    return _C["bm"], _C["pids"], _C["titles"]

def boot(x, n=2000, seed=13):
    x = np.array([v for v in x if v == v], float)
    if not len(x): return (float("nan"),)*3
    rng = np.random.default_rng(seed); s = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))

def recall_at(order, rel, k):
    rs = set(rel); return len([p for p in order[:k] if p in rs]) / len(rs) if rs else float("nan")

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", required=True); ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--catalog", default=None)
    a = ap.parse_args()
    bm, pids, titles = index(a.catalog); pid_arr = np.array(pids)
    rows = [json.loads(l) for l in open(f"{RES}/{a.tag}__probe_plans.jsonl")]
    rows = [r for r in rows if r["family"] == "A_entity"]
    forced, drops = [], []
    for r in rows:
        pt = r["plan"].get("product_type")
        forced.append(1.0 if pt else 0.0)
        if not pt: drops.append(0.0); continue
        rel = r["relevant_pids"]
        scores = bm.get_scores(list(tokset(r["query"])) or ["x"])
        order = pid_arr[np.argsort(-scores)].tolist()
        r_no = recall_at(order, rel, a.k)
        ptok = tokset(pt)
        keep = [pid for pid in order if ptok & tokset(titles.get(pid, ""))]
        if not keep: keep = order
        r_f = recall_at(keep, rel, a.k)
        drops.append(r_no - r_f)
    rep = dict(tag=a.tag, n_entity=len(rows), k=a.k,
               forced_type_rate=boot(forced), real_recall_drop=boot(drops))
    json.dump(rep, open(f"{RES}/{a.tag}__esci_probe_metrics.json", "w"), indent=2, default=float)
    ft = rep["forced_type_rate"]; rd = rep["real_recall_drop"]
    print(f"\n=== ESCI-PROBE {a.tag} (n={len(rows)} entity, k={a.k}) ===")
    print(f"  type-forced rate:        {ft[0]:.2f} [{ft[1]:.2f},{ft[2]:.2f}]")
    print(f"  real recall@{a.k} drop:    {rd[0]:.2f} [{rd[1]:.2f},{rd[2]:.2f}]")

if __name__ == "__main__":
    main()
