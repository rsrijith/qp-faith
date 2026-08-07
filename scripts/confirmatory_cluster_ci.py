#!/usr/bin/env python
"""Fix for methodology-review W1+W2: compute the confirmatory pooled harm with the
SAME motif-cluster bootstrap used for the WANDS headline (resample motif clusters,
each carrying its 4 model deltas), not a degenerate n=4 between-model bootstrap.

Also addresses W2 (instrument confound): re-score the WANDS 55 motifs with the SAME
title-based filter used in electronics/apparel, so the home-vs-rest gradient can be
read at a matched instrument instead of confounding curated-vs-title with breadth.

Outputs results/confirmatory_cluster_ci.json and prints the corrected numbers."""
import os, re, json, sys
import numpy as np, pandas as pd
from rank_bm25 import BM25Okapi
sys.path.insert(0, os.path.dirname(__file__))

RES = os.path.join(os.path.dirname(__file__), "..", "results")
DATA = os.path.join(os.path.dirname(__file__), "..", "data")
MODELS = ["qwen4", "llama4", "mistral4", "gemma4"]
STOP = set("a an the of for with and or in on to set sets by your our".split())
def norm(s): return re.sub(r"[^a-z0-9 ]", " ", str(s).lower()).strip()
def depl(w): return w[:-1] if len(w) > 3 and w.endswith("s") else w
def tokset(s): return {depl(w) for w in norm(s).split() if w and w not in STOP}
def recall_at(order, rel, k=100):
    rs = set(rel); return len([p for p in order[:k] if p in rs]) / len(rs) if rs else float("nan")

def build_index(catalog_path):
    cat = pd.read_parquet(catalog_path)
    pids = cat["product_id"].tolist(); titles = cat["ntitle"].tolist()
    bm = BM25Okapi([t.split() for t in titles])
    return bm, np.array(pids), {pids[i]: titles[i] for i in range(len(pids))}

def per_motif_drops(plans_path, bm, pid_arr, titles, k=100):
    """title-based hard filter (same instrument as validate_esci_probe). Returns
    {query: drop} for rows that emitted a product_type."""
    out = {}
    for line in open(plans_path):
        r = json.loads(line)
        if r.get("family") != "A_entity": continue
        pt = r["plan"].get("product_type")
        if not pt: continue  # only count emitted-type rows (matches scorer's drop list incl 0s)
        rel = r["relevant_pids"]
        scores = bm.get_scores(list(tokset(r["query"])) or ["x"])
        order = pid_arr[np.argsort(-scores)].tolist()
        r_no = recall_at(order, rel, k)
        ptok = tokset(pt)
        keep = [pid for pid in order if ptok & tokset(titles.get(pid, ""))] or order
        out[r["query"]] = r_no - recall_at(keep, rel, k)
    return out

def cluster_bootstrap(model_drops, n=5000, seed=13):
    """model_drops: {model: {query: drop}}. Resample the motif clusters; each cluster
    carries its per-model deltas; bootstrap stat = mean over motifs of (mean over models)."""
    motifs = sorted(set().union(*[set(d) for d in model_drops.values()]))
    M = np.array([[model_drops[m].get(q, np.nan) for m in MODELS] for q in motifs])  # motif x model
    per_motif = np.nanmean(M, axis=1)  # mean over models per motif
    per_motif = per_motif[~np.isnan(per_motif)]
    rng = np.random.default_rng(seed)
    s = per_motif[rng.integers(0, len(per_motif), (n, len(per_motif)))].mean(1)
    return float(per_motif.mean()), float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5)), len(per_motif)

out = {}
# 1. confirmatory domains: correct motif-cluster bootstrap (expansion)
for dom, catn in [("electronics", "catalog.parquet"), ("apparel", "catalog.parquet")]:
    bm, pid_arr, titles = build_index(f"{DATA}/probe_domain/{dom}/{catn}")
    md = {m: per_motif_drops(f"{RES}/dom-{dom}-{m}-expansion__probe_plans.jsonl", bm, pid_arr, titles) for m in MODELS}
    mean, lo, hi, nmot = cluster_bootstrap(md)
    out[dom] = {"pooled_expansion_harm_cluster": [mean, lo, hi], "n_motif_clusters": nmot}
    print(f"{dom:12} cluster-bootstrap pooled harm: {mean:.2f} [{lo:.2f},{hi:.2f}]  (n={nmot} motif clusters)")

# 2. W2: re-score WANDS 55 motifs with the TITLE-based instrument (instrument-matched to elec/apparel)
#    WANDS plans tagged probe-<model>-expansion; relevant_pids over WANDS catalog (product.csv).
wcat = pd.read_csv(f"{DATA}/product.csv", sep="\t")
wcat["ntitle"] = wcat["product_name"].map(norm)
wbm = BM25Okapi([t.split() for t in wcat["ntitle"].tolist()])
wpid = np.array(wcat["product_id"].tolist()); wtitles = {wcat["product_id"].iloc[i]: wcat["ntitle"].iloc[i] for i in range(len(wcat))}
wmd = {}
for m in MODELS:
    p = f"{RES}/probe-{m}-expansion__probe_plans.jsonl"
    if not os.path.exists(p): p = f"{RES}/probe-{m}-exp__probe_plans.jsonl"
    if not os.path.exists(p): print(f"  (WANDS plans for {m} not found at {p})"); continue
    wmd[m] = per_motif_drops(p, wbm, wpid, wtitles)
if wmd:
    mean, lo, hi, nmot = cluster_bootstrap(wmd)
    out["wands_title_instrument"] = {"pooled_expansion_harm_cluster": [mean, lo, hi], "n_motif_clusters": nmot}
    print(f"{'WANDS(title)':12} cluster-bootstrap pooled harm: {mean:.2f} [{lo:.2f},{hi:.2f}]  (n={nmot})  <- instrument-matched to elec/apparel")
print(f"\nWANDS curated-facet reference (from paper): 0.74 [0.64,0.83]")
json.dump(out, open(f"{RES}/confirmatory_cluster_ci.json", "w"), indent=2)
print("wrote results/confirmatory_cluster_ci.json")
