#!/usr/bin/env python
"""Dense-retrieval cell (validity reviewer m3): repeat the entity-probe product_type
HARD-vs-SOFT harm under a DENSE retriever instead of BM25, to test whether the
hard/soft bracket is a BM25 artifact. Embeds the full WANDS catalog with a
sentence-transformer, cosine-ranks, applies the injected product_type as a hard
category filter vs a soft (x0.5 score) penalty, reports recall@100. Cached embeddings."""
import os, re, json, argparse
import numpy as np, pandas as pd

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
RES = os.path.join(os.path.dirname(__file__), "..", "results")
EMB = f"{RES}/dense_catalog_bge.npy"
MODEL = "BAAI/bge-small-en-v1.5"
def norm(s): return re.sub(r"[^a-z0-9 ]", " ", str(s).lower()).strip()
def depl(w): return w[:-1] if len(w) > 3 and w.endswith("s") else w
def tokset(s): return {depl(w) for w in norm(s).split() if w}
def matches(val, v): A=tokset(val); B=tokset(v); return bool(A) and bool(B) and (A==B or A<=B or B<=A)

def get_index():
    from sentence_transformers import SentenceTransformer
    p = pd.read_csv(f"{DATA}/product.csv", sep="\t")
    pids = [int(x) for x in p["product_id"]]
    cls = {int(r.product_id): norm(r.product_class) for r in p.itertuples() if pd.notna(r.product_class)}
    m = SentenceTransformer(MODEL)
    if os.path.exists(EMB):
        emb = np.load(EMB)
    else:
        texts = [(str(n) + ". " + str(f))[:512] for n, f in zip(p["product_name"], p["product_features"])]
        emb = m.encode(texts, batch_size=256, normalize_embeddings=True, show_progress_bar=True).astype(np.float32)
        np.save(EMB, emb)
    return m, emb, np.array(pids), cls

def recall(order, rel, k=100):
    rs = set(rel); return len([x for x in order[:k] if x in rs]) / len(rs) if rs else float("nan")
def boot(x):
    x = np.array(x, float); rng = np.random.default_rng(13); s = x[rng.integers(0,len(x),(2000,len(x)))].mean(1)
    return x.mean(), np.percentile(s,2.5), np.percentile(s,97.5)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tags", nargs="+", required=True); a = ap.parse_args()
    m, emb, pid_arr, cls = get_index()
    LAMBDA = 0.5
    print(f"{'model':12s} {'base_recall':>11s} {'HARD_drop':>10s} {'SOFT_drop':>10s}")
    for tag in a.tags:
        rows = [r for r in (json.loads(l) for l in open(f"{RES}/{tag}__probe_plans.jsonl")) if r["family"]=="A_entity"]
        base=[]; hard=[]; soft=[]
        for r in rows:
            pt = r["plan"].get("product_type")
            if not pt: continue
            rel = [int(x) for x in r["relevant_pids"]]
            q = m.encode([r["query"]], normalize_embeddings=True)[0]
            sc = emb @ q
            order = pid_arr[np.argsort(-sc)].tolist()
            rb = recall(order, rel)
            keep = [pid_arr[i] for i in np.argsort(-sc) if matches(pt, cls.get(int(pid_arr[i]), ""))]
            rh = recall(keep if keep else order, rel)
            pen = np.array([1.0 if matches(pt, cls.get(int(pid_arr[i]), "")) else LAMBDA for i in range(len(pid_arr))])
            order_s = pid_arr[np.argsort(-(sc*pen))].tolist()
            rs_ = recall(order_s, rel)
            base.append(rb); hard.append(rb-rh); soft.append(rb-rs_)
        b=boot(base); h=boot(hard); s=boot(soft)
        rep=dict(tag=tag, model="bge-small", base_recall=b, hard_drop=h, soft_drop=s)
        json.dump(rep, open(f"{RES}/{tag}__dense.json","w"), default=float, indent=2)
        print(f"{tag:12s} {b[0]:11.2f} {h[0]:6.2f} [{h[1]:.2f},{h[2]:.2f}] {s[0]:5.2f} [{s[1]:.2f},{s[2]:.2f}]")

if __name__ == "__main__":
    main()
