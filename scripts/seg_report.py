#!/usr/bin/env python
"""
Segment analysis: split WANDS queries into CONCRETE (head token is a known
product noun) vs AMBIGUOUS (entity / abstract / non-product-noun), then report
spurious-injection and harm within each segment. Reads a scored parquet.

Hypothesis from the pilot: spurious product_type injection (and its catastrophic
over-constraining harm) concentrates in the AMBIGUOUS segment.
"""
import os, sys, re, json, collections, argparse
import numpy as np
import pandas as pd

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
RES = os.path.join(os.path.dirname(__file__), "..", "results")
STOP = set("a an the of for with and or in on to set sets piece pieces that have".split())

def norm(s): return re.sub(r"[^a-z0-9 ]", " ", str(s).lower()).strip()

def build_product_vocab(p):
    vocab = collections.Counter()
    for pc in p["product_class"].dropna():
        for w in norm(pc).split():
            if len(w) > 2 and w not in STOP: vocab[w] += 1
    # producttype facet values too
    for s in p["product_features"].dropna():
        for part in str(s).split("|"):
            if part.lower().strip().startswith("producttype"):
                v = part.split(":", 1)[1] if ":" in part else ""
                for w in norm(v).split():
                    if len(w) > 2 and w not in STOP: vocab[w] += 1
    # keep tokens that appear as a product noun in >=3 products (drop noise)
    return {w for w, c in vocab.items() if c >= 3}

def classify(query, vocab):
    toks = [w for w in norm(query).split() if w not in STOP]
    if not toks: return "ambiguous"
    # concrete if ANY content token is a known product noun
    return "concrete" if any(t in vocab for t in toks) else "ambiguous"

def boot(vals, n=2000, seed=13):
    vals = np.asarray(vals, float)
    if len(vals) == 0: return (float("nan"),)*3
    rng = np.random.default_rng(seed)
    s = vals[rng.integers(0, len(vals), size=(n, len(vals)))].mean(1)
    return float(vals.mean()), float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    p = pd.read_csv(f"{DATA}/product.csv", sep="\t")
    q = pd.read_csv(f"{DATA}/query.csv", sep="\t")
    df = pd.read_parquet(f"{RES}/{args.tag}__scored.parquet")

    vocab = build_product_vocab(p)
    seg = {int(qid): classify(qstr, vocab) for qid, qstr in zip(q["query_id"], q["query"])}
    df["segment"] = df.query_id.map(seg)

    qseg = {qid: seg[qid] for qid in df.query_id.unique()}
    n_by_seg = collections.Counter(seg[qid] for qid in df.query_id.unique())

    print(f"\n=== segment report: {args.tag} ===")
    print(f"queries scored: {df.query_id.nunique()}  "
          f"(concrete={n_by_seg['concrete']}, ambiguous={n_by_seg['ambiguous']})")
    print(f"{'segment':10s} {'queries':>8s} {'slots':>6s} {'spur':>5s} {'harм':>5s} "
          f"{'SCR(query)':>22s} {'Harmful-SCR(spur)':>22s}")
    for s in ["concrete", "ambiguous"]:
        d = df[df.segment == s]
        qids = sorted(d.query_id.unique())
        byq = d.groupby("query_id")
        has_spur = [int(byq.get_group(g).spurious.sum() > 0) for g in qids]
        sp = d[d.spurious == 1]
        hsr = sp.harmful.tolist()
        scr = boot(has_spur); h = boot([float(x) for x in hsr]) if hsr else (float("nan"),)*3
        print(f"{s:10s} {len(qids):8d} {len(d):6d} {int(d.spurious.sum()):5d} "
              f"{int(d.harmful.sum()):5d} "
              f"{scr[0]:.3f} [{scr[1]:.3f},{scr[2]:.3f}]   "
              f"{h[0]:.3f} [{h[1]:.3f},{h[2]:.3f}]")
    # show ambiguous-segment harmful examples
    amb_harm = df[(df.segment == "ambiguous") & (df.harmful == 1)]
    print(f"\nambiguous-segment harmful injections ({len(amb_harm)}):")
    for r in amb_harm.sort_values("conflicting", ascending=False).head(15).itertuples():
        print(f"  '{r.query}' -> {r.slot}='{r.value}'  excl {r.conflicting}/{r.n_exact}")

if __name__ == "__main__":
    main()
