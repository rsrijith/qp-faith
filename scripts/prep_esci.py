#!/usr/bin/env python
"""Prepare an ESCI secondary slice (brand/color only, per the plan). US locale,
sample ~480 queries with >=10 labeled products and >=1 Exact. Persists:
  data/esci/queries.jsonl  [{query_id, query}]
  data/esci/pool.parquet   per (query_id, product_id, esci_label, brand, color)
Deterministic (blake2b sampling)."""
import os, json, hashlib
import pandas as pd
from huggingface_hub import hf_hub_download

OUT = os.path.join(os.path.dirname(__file__), "..", "data", "esci")
os.makedirs(OUT, exist_ok=True)
N_Q = 480

def h(s):
    return int.from_bytes(hashlib.blake2b(str(s).encode(), digest_size=8).digest(), "big")

def main():
    f = hf_hub_download("tasksource/esci",
                        "data/test-00000-of-00004-d48474212b95f33b.parquet", repo_type="dataset")
    df = pd.read_parquet(f)
    df = df[df.product_locale == "us"].copy()
    # queries with enough graded products and >=1 Exact
    g = df.groupby("query_id")
    ok = []
    for qid, grp in g:
        if len(grp) >= 10 and (grp.esci_label == "Exact").sum() >= 1:
            ok.append(qid)
    ok = sorted(ok, key=lambda x: h(x))[:N_Q]   # stable sample
    sub = df[df.query_id.isin(set(ok))].copy()

    q = sub[["query_id", "query"]].drop_duplicates("query_id")
    with open(f"{OUT}/queries.jsonl", "w") as fh:
        for r in q.itertuples():
            fh.write(json.dumps({"query_id": int(r.query_id), "query": str(r.query)}) + "\n")
    pool = sub[["query_id", "product_id", "esci_label", "product_brand", "product_color"]].copy()
    pool.to_parquet(f"{OUT}/pool.parquet")
    man = dict(n_queries=int(q.shape[0]), n_pairs=int(pool.shape[0]),
               brand_pop=float(pool.product_brand.notna().mean()),
               color_pop=float(pool.product_color.notna().mean()),
               label_dist=pool.esci_label.value_counts().to_dict(),
               sample_queries=q["query"].head(15).tolist())
    json.dump(man, open(f"{OUT}/manifest.json", "w"), indent=2, default=str)
    print(json.dumps(man, indent=2, default=str))

if __name__ == "__main__":
    main()
