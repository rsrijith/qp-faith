#!/usr/bin/env python
"""D-earn: EARN the ESCI headline with a REAL retrieval over the full catalog.

The submitted paper measures ESCI harm as an exclusion metric on the human-Exact
LABEL set (scripts/esci_routing.py): "of the query's human-Exact products, what
fraction has a brand/color that fails the planner's emitted filter." R1 comment 3
reads that as a construction on the label set rather than a measured retrieval
degradation. This script converts it into the measured quantity:

  1. Build a BM25 index over the FULL ESCI-US product catalog (small=482K / large=1.22M).
  2. For each real ESCI query, retrieve the top-K products with BM25 (a real ranker).
  3. Apply the planner's routed brand/color value as an UPSTREAM hard pre-filter
     (full-hard-filter semantics: drop every product whose structured brand/color
     does not match the emitted value).
  4. Measure recall@K and nDCG@K on the human-Exact products, WITH vs WITHOUT the
     filter. The drop is the realized retrieval loss.

Because the filter removes documents BEFORE ranking, the Exact products it drops
cannot be recovered by any downstream reranker operating on the surviving candidate
set (cf. Appendix C.10's cross-encoder, which cannot recover the loss). So the
measured recall@K loss is a ceiling on any reranker's recovery.

Consistency with the submitted analysis (so this is an extension, not a new dataset):
same 4 local planners, same {expansion, conservative} modes, same {brand, color}
facets, same norm()/facet_match()/in_query() logic, same MIN_EXACT=5 subset, seed 13,
same query-clustered bootstrap. qrels come from the same data/esci/pool.parquet human
judgments the paper's ESCI numbers are built on; product text for the catalog comes
from the full ESCI corpus (HF cache, offline). $0, CPU-only (bm25s).

Restartable: caches the catalog parquet, the bm25s index, and the top-K retrieval to
disk; re-runs skip completed stages. Recompute / resume:
  ./.venv/bin/python scripts/esci_dearn.py --version small --k 1000
"""
import os, re, json, argparse, time
import numpy as np, pandas as pd

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
DATA = os.path.join(HERE, "..", "data")
DEARN = os.path.join(RES, "dearn")
os.makedirs(DEARN, exist_ok=True)

RNG = np.random.default_rng(13)
MODELS = ["qwen4", "llama4", "mistral4", "gemma4"]
MIN_EXACT = 5
GAIN = {"Exact": 1.0, "Substitute": 0.1, "Complement": 0.01, "Irrelevant": 0.0}
REPORT_K = [10, 100]

# --- normalization / matching: identical to scripts/esci_routing.py ---
def norm(s): return re.sub(r"[^a-z0-9 ]", " ", str(s).lower()).strip()
def head(s):
    t = norm(s).split(); return t[-1] if t else ""
def toks(s): return set(norm(s).split())
def facet_match(emitted, val):
    e = norm(emitted); v = norm(val)
    if not e or not v: return False
    return e == v or e in v or v in e or bool(toks(e) & toks(v))
def in_query(emitted, q):
    e = head(emitted)
    return bool(e) and (e in toks(q) or norm(emitted) in norm(q))

# R1 comment 3 entailment audit: a planner that emits a non-committal placeholder
# (brand="unknown", color="any", "varies", "unspecified", ...) is DECLINING to guess a
# value, not routing a spurious one. A realistic system would not apply such a value as a
# hard equality filter. We treat these as no emission (no filter, not a spurious fire).
NONCOMMITTAL = {"any", "none", "n a", "na", "unknown", "unspecified", "not specified",
 "varies", "various", "varied", "generic", "unbranded", "brandless", "no brand", "no color",
 "no colour", "multiple", "assorted", "misc", "miscellaneous", "not applicable",
 "not available", "other", "standard", "default", "null", "nil", "tbd", "undefined"}
def is_committal(v):
    n = norm(v)
    return bool(n) and n not in NONCOMMITTAL


def build_catalog(version):
    """Unique ESCI-US products -> (product_id, text, brand, color). Unions in every
    judged product for our 480 queries so all Exact targets are retrievable. Cached."""
    path = os.path.join(DEARN, f"catalog_{version}.parquet")
    if os.path.exists(path):
        df = pd.read_parquet(path)
        print(f"[catalog] loaded {len(df):,} products from cache ({version})")
        return df
    from datasets import load_dataset
    print(f"[catalog] building {version} from full ESCI (HF cache), streaming...")
    ds = load_dataset("tasksource/esci")
    ours = {json.loads(l)["query_id"] for l in open(f"{DATA}/esci/queries.jsonl")}
    cols = ["product_id", "product_locale", "product_text", "product_title",
            "product_brand", "product_color", "query_id", "small_version"]
    # stream, accumulating only unique US products -> bounded memory (no giant DataFrame)
    prod = {}          # pid -> [text, brand, color, is_small]
    judged_pids = set()
    seen = 0
    for split in ds:
        for b in ds[split].select_columns(cols).iter(batch_size=20000):
            n = len(b["product_id"])
            for i in range(n):
                if b["product_locale"][i] != "us":
                    continue
                pid = b["product_id"][i]
                if b["query_id"][i] in ours:
                    judged_pids.add(pid)
                sm = 1 if b["small_version"][i] else 0
                if pid not in prod:
                    txt = b["product_text"][i] or b["product_title"][i] or ""
                    prod[pid] = [str(txt), str(b["product_brand"][i] or ""),
                                 str(b["product_color"][i] or ""), sm]
                elif sm:
                    prod[pid][3] = 1
            seen += n
        print(f"[catalog]   scanned {seen:,} rows; {len(prod):,} unique US products so far")
    if version == "small":
        pids = [p for p, v in prod.items() if v[3] == 1 or p in judged_pids]
    else:
        pids = list(prod.keys())
    df = pd.DataFrame({
        "product_id": pids,
        "text": [prod[p][0] for p in pids],
        "brand": [prod[p][1] for p in pids],
        "color": [prod[p][2] for p in pids],
    })
    df.to_parquet(path)
    print(f"[catalog] built {len(df):,} unique products ({version}); judged in catalog: "
          f"{len(judged_pids & set(pids)):,}/{len(judged_pids):,}")
    return df


def build_index(version, texts):
    """bm25s index over the catalog text. Cached to disk (save/load)."""
    import bm25s, Stemmer
    idx_dir = os.path.join(DEARN, f"bm25_{version}")
    stemmer = Stemmer.Stemmer("english")
    if os.path.exists(os.path.join(idx_dir, "data.csc.index.npy")) or os.path.exists(os.path.join(idx_dir, "params.index.json")):
        print(f"[index] loading bm25s index from cache ({version})")
        retriever = bm25s.BM25.load(idx_dir, mmap=True)
        return retriever, stemmer
    print(f"[index] tokenizing {len(texts):,} docs...")
    t0 = time.time()
    corpus_tokens = bm25s.tokenize(texts, stemmer=stemmer, stopwords="en", show_progress=True)
    print(f"[index] tokenized in {time.time()-t0:.0f}s; indexing...")
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens, show_progress=True)
    retriever.save(idx_dir)
    print(f"[index] built + saved in {time.time()-t0:.0f}s")
    return retriever, stemmer


def retrieve_all(version, K, retriever, stemmer, catalog):
    """Top-K product_ids per query for the MIN_EXACT>=5 subset. Cached (restartable)."""
    import bm25s
    path = os.path.join(DEARN, f"retrieval_{version}_k{K}.jsonl")
    done = {}
    if os.path.exists(path):
        for l in open(path):
            r = json.loads(l); done[r["query_id"]] = r["pids"]
    # qrels + the >=5-Exact query subset from pool.parquet (same as the paper)
    pool = pd.read_parquet(f"{DATA}/esci/pool.parquet")
    qtext = {int(r.query_id): str(r["query"]) for _, r in
             pd.DataFrame([json.loads(l) for l in open(f"{DATA}/esci/queries.jsonl")]).iterrows()}
    exact_ct = pool[pool.esci_label == "Exact"].groupby("query_id").size()
    qids = sorted(int(q) for q in exact_ct[exact_ct >= MIN_EXACT].index)
    pid_order = catalog.product_id.values
    todo = [q for q in qids if q not in done]
    print(f"[retrieve] {len(done)} cached, {len(todo)} to do (K={K}, {len(qids)} queries total)")
    if todo:
        with open(path, "a") as f:
            for i in range(0, len(todo), 64):
                batch = todo[i:i + 64]
                qtoks = bm25s.tokenize([qtext[q] for q in batch], stemmer=stemmer,
                                       stopwords="en", show_progress=False)
                idxs, _ = retriever.retrieve(qtoks, k=K, show_progress=False)
                for j, q in enumerate(batch):
                    pids = [str(pid_order[ix]) for ix in idxs[j]]
                    f.write(json.dumps({"query_id": q, "pids": pids}) + "\n")
                    f.flush(); done[q] = pids
                print(f"  retrieved {min(i+64,len(todo))}/{len(todo)}")
    return done, qids, qtext, pool


def dcg(gains):
    return sum(g / np.log2(i + 2) for i, g in enumerate(gains))

def ndcg_at(ranked_labels, ideal_gains, k):
    d = dcg([GAIN.get(l, 0.0) for l in ranked_labels[:k]])
    idcg = dcg(sorted(ideal_gains, reverse=True)[:k])
    return d / idcg if idcg > 0 else 0.0


def score(version, K, retrieval, qids, qtext, pool, catalog):
    """Baseline vs routed-hard-filter recall@k / nDCG@k, per model/mode/facet."""
    brand_of = dict(zip(catalog.product_id.astype(str), catalog.brand))
    color_of = dict(zip(catalog.product_id.astype(str), catalog.color))
    val_of = {"brand": brand_of, "color": color_of}
    # per-query qrels: pid -> label, and the Exact pid set
    qrels = {}
    for qid, g in pool.groupby("query_id"):
        qi = int(qid)
        qrels[qi] = {str(p): l for p, l in zip(g.product_id, g.esci_label)}

    def cluster_ci(vals, B=5000):
        a = np.array(vals, float); n = len(a)
        if n == 0: return (0.0, 0.0, 0.0)
        b = [a[RNG.integers(0, n, n)].mean() for _ in range(B)]
        return (round(float(a.mean()), 4), round(float(np.percentile(b, 2.5)), 4),
                round(float(np.percentile(b, 97.5)), 4))

    def hier_ci(per_model_vals, B=5000):
        """Two-stage bootstrap matching scripts/hier_bootstrap.py: resample the 4 models
        with replacement, then resample each drawn model's queries with replacement.
        The query-clustered interval treats the 4 models as fixed and so understates the
        uncertainty; this one carries the between-model variance (R1 comment 7)."""
        arrs = [np.array(v, float) for v in per_model_vals if len(v)]
        if not arrs: return (0.0, 0.0, 0.0)
        point = float(np.mean([a.mean() for a in arrs]))
        boots = []
        for _ in range(B):
            drawn = [arrs[i] for i in RNG.integers(0, len(arrs), len(arrs))]
            boots.append(np.mean([a[RNG.integers(0, len(a), len(a))].mean() for a in drawn]))
        return (round(point, 4), round(float(np.percentile(boots, 2.5)), 4),
                round(float(np.percentile(boots, 97.5)), 4))

    out = {"design": f"real BM25 retrieval over the full ESCI-US catalog ({version}); "
                     f"top-K={K}; routed brand/color applied as an upstream hard pre-filter; "
                     f"recall@k / nDCG@k on human-Exact products; 4 local planners; "
                     f"MIN_EXACT={MIN_EXACT}; qrels = data/esci/pool.parquet; seed 13.",
           "version": version, "K": K, "n_queries": len(qids),
           "catalog_size": int(len(catalog))}

    for mode, label in [("exp", "expansion"), ("con", "conservative")]:
        out[label] = {}
        for facet in ["brand", "color"]:
            vlook = val_of[facet]
            # accumulate per (model,query): baseline & filtered recall/ndcg at each K
            per_model = {m: {f"base_r@{k}": [] for k in REPORT_K} for m in MODELS}
            for m in MODELS:
                for k in REPORT_K:
                    per_model[m][f"filt_r@{k}"] = []
                    per_model[m][f"loss_r@{k}"] = []
                    per_model[m][f"sponly_loss_r@{k}"] = []   # loss attributable to SPURIOUS filters only
                    per_model[m][f"base_ndcg@{k}"] = []
                    per_model[m][f"filt_ndcg@{k}"] = []
                    per_model[m][f"loss_ndcg@{k}"] = []
                    per_model[m][f"sponly_loss_ndcg@{k}"] = []   # nDCG loss attributable to SPURIOUS filters only
                per_model[m]["spur"] = []
                per_model[m][f"spur_loss_r@{max(REPORT_K)}"] = []
                fn = f"{RES}/esci-{m}-{mode}__plans.jsonl"
                if not os.path.exists(fn):
                    continue
                plans = {}
                for l in open(fn):
                    r = json.loads(l); plans[int(r["query_id"])] = r.get("plan") or {}
                for qid in qids:
                    if qid not in retrieval:
                        continue
                    ranked = retrieval[qid]
                    rel = qrels.get(qid, {})
                    exact_pids = {p for p, lab in rel.items() if lab == "Exact"}
                    n_exact = len(exact_pids)
                    if n_exact == 0:
                        continue
                    ideal = [GAIN.get(l, 0.0) for l in rel.values()]
                    ranked_labels = [rel.get(p, "Irrelevant") for p in ranked]
                    # baseline metrics
                    base_r = {k: len(set(ranked[:k]) & exact_pids) / n_exact for k in REPORT_K}
                    base_n = {k: ndcg_at(ranked_labels, ideal, k) for k in REPORT_K}
                    # filter
                    p = plans.get(qid, {})
                    emitted_raw = str(p.get(facet, "")).strip()
                    # non-committal placeholder = the planner declined to guess -> no constraint
                    emitted = emitted_raw if is_committal(emitted_raw) else ""
                    q = qtext.get(qid, "")
                    spurious = bool(emitted) and not in_query(emitted, q)
                    if emitted:
                        fr = [pid for pid in ranked if facet_match(emitted, vlook.get(pid, ""))]
                    else:
                        fr = ranked  # no constraint emitted -> no filter
                    fr_labels = [rel.get(p_, "Irrelevant") for p_ in fr]
                    filt_r = {k: len(set(fr[:k]) & exact_pids) / n_exact for k in REPORT_K}
                    filt_n = {k: ndcg_at(fr_labels, ideal, k) for k in REPORT_K}
                    for k in REPORT_K:
                        per_model[m][f"base_r@{k}"].append(base_r[k])
                        per_model[m][f"filt_r@{k}"].append(filt_r[k])
                        per_model[m][f"loss_r@{k}"].append(base_r[k] - filt_r[k])
                        # spurious-only: the loss attributable to MIS-ROUTING (0 when the filter is a legitimate in-query value)
                        per_model[m][f"sponly_loss_r@{k}"].append((base_r[k] - filt_r[k]) if spurious else 0.0)
                        per_model[m][f"base_ndcg@{k}"].append(base_n[k])
                        per_model[m][f"filt_ndcg@{k}"].append(filt_n[k])
                        per_model[m][f"loss_ndcg@{k}"].append(base_n[k] - filt_n[k])
                        # spurious-only nDCG loss (0 when the filter is a legitimate in-query value)
                        per_model[m][f"sponly_loss_ndcg@{k}"].append((base_n[k] - filt_n[k]) if spurious else 0.0)
                    per_model[m]["spur"].append(1.0 if spurious else 0.0)
                    if spurious:
                        kk = max(REPORT_K)
                        per_model[m][f"spur_loss_r@{kk}"].append(base_r[kk] - filt_r[kk])
            # pooled across models
            pooled = {}
            for k in REPORT_K:
                pooled[f"base_recall@{k}"] = cluster_ci([x for m in MODELS for x in per_model[m][f"base_r@{k}"]])
                pooled[f"filtered_recall@{k}"] = cluster_ci([x for m in MODELS for x in per_model[m][f"filt_r@{k}"]])
                pooled[f"recall_loss@{k}"] = cluster_ci([x for m in MODELS for x in per_model[m][f"loss_r@{k}"]])
                pooled[f"spurious_only_recall_loss@{k}"] = cluster_ci([x for m in MODELS for x in per_model[m][f"sponly_loss_r@{k}"]])
                # R1 c7 / DA: the lead figure also gets the hierarchical (models + queries) interval
                pooled[f"spurious_only_recall_loss@{k}_hierarchical"] = hier_ci([per_model[m][f"sponly_loss_r@{k}"] for m in MODELS])
                pooled[f"base_ndcg@{k}"] = cluster_ci([x for m in MODELS for x in per_model[m][f"base_ndcg@{k}"]])
                pooled[f"ndcg_loss@{k}"] = cluster_ci([x for m in MODELS for x in per_model[m][f"loss_ndcg@{k}"]])
                pooled[f"spurious_only_ndcg_loss@{k}"] = cluster_ci([x for m in MODELS for x in per_model[m][f"sponly_loss_ndcg@{k}"]])
            kk = max(REPORT_K)
            pooled[f"spurious_subset_recall_loss@{kk}"] = cluster_ci([x for m in MODELS for x in per_model[m][f"spur_loss_r@{kk}"]])
            pooled["spurious_routing_rate"] = cluster_ci([x for m in MODELS for x in per_model[m]["spur"]])
            pooled["by_model"] = {
                m: {f"recall_loss@{max(REPORT_K)}": round(float(np.mean(per_model[m][f"loss_r@{max(REPORT_K)}"])), 4)
                    if per_model[m][f"loss_r@{max(REPORT_K)}"] else None,
                    "spurious_routing_rate": round(float(np.mean(per_model[m]["spur"])), 4)
                    if per_model[m]["spur"] else None}
                for m in MODELS}
            out[label][facet] = pooled
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", choices=["small", "large"], default="small")
    ap.add_argument("--k", type=int, default=1000)
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()

    catalog = build_catalog(args.version)
    retriever, stemmer = build_index(args.version, catalog.text.tolist())
    retrieval, qids, qtext, pool = retrieve_all(args.version, args.k, retriever, stemmer, catalog)
    out = score(args.version, args.k, retrieval, qids, qtext, pool, catalog)

    path = os.path.join(RES, f"esci_dearn_{args.version}_k{args.k}.json")
    json.dump(out, open(path, "w"), indent=2)

    def f(t): m, lo, hi = t; return f"{m} [{lo}, {hi}]"
    print(f"\n=== D-earn ESCI ({args.version} catalog = {out['catalog_size']:,} products, "
          f"n={out['n_queries']} queries, top-K={args.k}) ===")
    for label in ["expansion", "conservative"]:
        print(f"\n--- {label} ---")
        for facet in ["brand", "color"]:
            d = out[label][facet]
            print(f"  {facet}: spurious-routing {f(d['spurious_routing_rate'])}")
            for k in REPORT_K:
                print(f"    recall@{k}: base {f(d[f'base_recall@{k}'])} -> filtered "
                      f"{f(d[f'filtered_recall@{k}'])}  (loss {f(d[f'recall_loss@{k}'])})")
                print(f"    nDCG@{k}:   loss {f(d[f'ndcg_loss@{k}'])}")
            kk = max(REPORT_K)
            print(f"    spurious-subset recall@{kk} loss: {f(d[f'spurious_subset_recall_loss@{kk}'])}")
    print("\nwrote", path)


if __name__ == "__main__":
    main()
