#!/usr/bin/env python
"""R3 item 2: why is the leaf-category realized loss LOWER than the title-based one?

Reviewer 3 could not reconcile the two recall@100 losses Section 5.7 reports for the same two
400K-product catalogs (title 0.48/0.38 vs leaf 0.14/0.27) and assumed the two instruments define
the relevant set differently. They do not. scripts/clean_category_harm.py reads `relevant_pids`
from the same cached plan file for both scorings and varies only `sigmap`, the per-product signal
the emitted product_type is token-matched against. The retrieval, the metric, the motif set, the
planners and the bootstrap are all held fixed.

So the gap has to come from the filter. This script instruments it, mirroring per_motif_drops()
line for line:

  (a) FALLBACK. `keep = [...] or order` means that when the emitted product_type matches NO
      product's signal anywhere in the ranking, the scorer applies no filter and records a loss of
      exactly 0. Product titles are long free text, so a title match almost always exists; leaf
      category names are a small controlled vocabulary (974 distinct labels in electronics, 1,263
      in apparel, against ~267K and ~114K distinct title tokens), so an emitted type such as
      "toy" or "jewelry" can match nothing at all. In electronics it matches nothing for 31
      distinct emitted types. Every such cell enters the mean as a zero.

  (b) SELECTIVITY. Where the leaf filter does bite, it may still keep a different share of the
      motif's relevant set than the title filter does ON THOSE SAME CELLS. Comparing each
      instrument on its own fired subset would compare different cell sets, so the title figure
      is computed on the leaf-fired cells, not on all of them.

The unit is the (motif, planner) cell, which is what makes the decomposition exact:

    mean_loss_all = fire_rate x mean_loss_fired

verified to 6 dp for every arm. The paper's own cluster bootstrap resamples motifs, so the
motif-mean and the cell-mean differ; the cell unit is used here because it is the one the
identity closes on and the one the fallback rate is naturally expressed in.

Also writes results/r3_instrument_diagnostic_{domain}.csv, one row per (motif, planner,
instrument), carrying the planner name so a row can be attributed without relying on file order.

Reuses cached plans and catalogs. No new inference. $0.
Run: ./.venv/bin/python scripts/r3_instrument_diagnostic.py
"""
import os, re, json
import numpy as np, pandas as pd
from rank_bm25 import BM25Okapi

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
ROOT = os.path.join(HERE, "..", "data", "probe_domain")
MODELS = ["qwen4", "llama4", "mistral4", "gemma4"]
STOP = set("a an the of for with and or in on to set sets by your our".split())

# --- identical to clean_category_harm.py -------------------------------------------------
def norm(s): return re.sub(r"[^a-z0-9 ]", " ", str(s).lower()).strip()
def depl(w): return w[:-1] if len(w) > 3 and w.endswith("s") else w
def tokset(s): return {depl(w) for w in norm(s).split() if w and w not in STOP}
def recall_at(order, rel, k=100):
    rs = set(rel); return len([p for p in order[:k] if p in rs]) / len(rs) if rs else float("nan")


def build_index(catalog_path):
    cat = pd.read_parquet(catalog_path)
    pids = cat["product_id"].astype(str).tolist(); titles = cat["ntitle"].tolist()
    bm = BM25Okapi([t.split() for t in titles])
    return bm, np.array(pids), {pids[i]: titles[i] for i in range(len(pids))}
# -----------------------------------------------------------------------------------------


def collect(plans_path, model, bm, pid_arr, sigmaps, order_cache, k=100):
    """One row per (motif, planner, instrument). Mirrors per_motif_drops(), but records whether
    the fallback fired instead of silently returning a 0 loss."""
    rows, skipped = [], 0
    for line in open(plans_path):
        r = json.loads(line)
        if r.get("family") != "A_entity":
            skipped += 1; continue
        pt = (r.get("plan") or {}).get("product_type")
        if not pt:
            skipped += 1; continue
        rel = r["relevant_pids"]
        if isinstance(rel, str):
            rel = json.loads(rel)
        q = r["query"]
        if q not in order_cache:
            scores = bm.get_scores(list(tokset(q)) or ["x"])
            order_cache[q] = pid_arr[np.argsort(-scores)].tolist()
        order = order_cache[q]
        r_no = recall_at(order, rel, k)
        ptok = tokset(pt)
        for name, sig in sigmaps.items():
            matched = [pid for pid in order if ptok & tokset(sig.get(pid, ""))]
            fired = len(matched) > 0      # False => `or order` fallback => loss forced to 0
            keep = matched or order
            rows.append({"motif": q, "model": model, "instrument": name, "product_type": pt,
                         "fired": fired, "n_kept": len(matched),
                         "loss": r_no - recall_at(keep, rel, k), "base_recall": r_no})
    return rows, skipped


def label_level_exclusion(plans_path, leaf, leafvocab_tokens, k=100):
    """Label-level exclusion, mirroring per_motif_exclusion() in clean_category_harm.py, split by
    whether the emitted type matches ANY leaf label in the catalog.

    Reviewer 3 (R3) asked how the label-level figure treats a type that matches no leaf label. It has
    no `or order` fallback, so such a type scores exactly 1.0: every relevant product with a label
    fails to match. That is the OPPOSITE convention to the realized loss, where the same event means
    no filter and a loss of 0. This function reports the published all-cells figure alongside the
    figure restricted to matchable types, so the clause added at proof can say which way the
    convention pushes the number and by how much.
    """
    allc, matchable = [], []
    for line in open(plans_path):
        r = json.loads(line)
        if r.get("family") != "A_entity":
            continue
        pt = (r.get("plan") or {}).get("product_type")
        if not pt:
            continue
        rel = r["relevant_pids"]
        if isinstance(rel, str):
            rel = json.loads(rel)
        ptok = tokset(pt)
        miss = [0.0 if (ptok & tokset(leaf.get(str(p), ""))) else 1.0
                for p in rel if leaf.get(str(p), "")]
        if not miss:
            continue
        v = float(np.mean(miss))
        allc.append(v)
        if ptok & leafvocab_tokens:
            matchable.append(v)
    return allc, matchable


def summarize(df, dom, coverage):
    t = df[df.instrument == "title"].sort_values(["motif", "model"]).reset_index(drop=True)
    l = df[df.instrument == "category_leaf"].sort_values(["motif", "model"]).reset_index(drop=True)
    assert len(t) == len(l) and (t.motif == l.motif).all() and (t.model == l.model).all()
    m = l.fired.values                    # cells where the LEAF filter fires
    fire = float(m.mean())
    out = {
        "n_cells": int(len(l)),
        "catalog_category_coverage": round(float(coverage), 3),
        "leaf_fallback_rate": round(1 - fire, 3),
        "leaf_fire_rate": round(fire, 3),
        "n_leaf_fired_cells": int(m.sum()),
        "title_loss_all_cells": round(float(t.loss.mean()), 3),
        "leaf_loss_all_cells": round(float(l.loss.mean()), 3),
        # both instruments on the SAME cells: the ones where the leaf filter actually fires
        "title_loss_on_leaf_fired": round(float(t.loss.values[m].mean()), 3),
        "leaf_loss_on_leaf_fired": round(float(l.loss.values[m].mean()), 3),
        "paired_leaf_minus_title": round(float((l.loss.values[m] - t.loss.values[m]).mean()), 3),
        # exactness of  mean_loss_all = fire_rate x mean_loss_fired
        "identity_residual": round(abs(fire * float(l.loss.values[m].mean())
                                       - float(l.loss.mean())), 9),
    }
    print(f"\n=== {dom} (expansion), unit = (motif, planner) cell, n={out['n_cells']} ===")
    print(f"  leaf fallback rate            {out['leaf_fallback_rate']:.3f}   "
          f"(title {1 - t.fired.mean():.3f})")
    print(f"  loss over all cells           title {out['title_loss_all_cells']:.3f}   "
          f"leaf {out['leaf_loss_all_cells']:.3f}")
    print(f"  loss on the {out['n_leaf_fired_cells']:>3} leaf-fired cells  "
          f"title {out['title_loss_on_leaf_fired']:.3f}   leaf {out['leaf_loss_on_leaf_fired']:.3f}"
          f"   (paired {out['paired_leaf_minus_title']:+.3f})")
    print(f"  identity fire x fired = all   residual {out['identity_residual']:.2e}")
    return out


def main():
    out = {"design": "R3 item 2: fallback vs selectivity as the cause of the title-vs-leaf "
                     "realized-loss gap on the two 400K domains. Expansion prompt, 4 planners, "
                     "same relevant sets and retrieval as clean_category_harm.py. Unit is the "
                     "(motif, planner) cell; both instruments are compared on the cells where the "
                     "leaf filter fires.", "by_domain": {}}
    for dom in ["electronics", "apparel"]:
        ddir = os.path.join(ROOT, dom)
        cats = pd.read_parquet(os.path.join(ddir, "categories.parquet"))
        catpath = dict(zip(cats.product_id.astype(str), cats.category_path.fillna("")))
        leaf = {p: (path.split(" > ")[-1] if path else "") for p, path in catpath.items()}
        bm, pid_arr, titles = build_index(os.path.join(ddir, "catalog.parquet"))
        coverage = np.mean([1.0 if catpath.get(p, "") else 0.0 for p in titles])
        sigmaps = {"title": titles, "category_leaf": leaf}
        rows, order_cache = [], {}
        for m in MODELS:
            pth = f"{RES}/dom-{dom}-{m}-expansion__probe_plans.jsonl"
            if not os.path.exists(pth):
                raise SystemExit(f"missing cached plans: {pth}")
            r, skipped = collect(pth, m, bm, pid_arr, sigmaps, order_cache)
            assert skipped == 0, f"{pth}: {skipped} rows dropped by the family/product_type guards"
            rows += r
        df = pd.DataFrame(rows)
        assert df.groupby(["motif", "model", "instrument"]).size().max() == 1, "duplicate cell"
        out["by_domain"][dom] = summarize(df, dom, coverage)

        # R3 proof item 1: how the label-level figure treats an unmatchable emitted type
        leafvocab_tokens = set().union(*[tokset(v) for v in set(leaf.values()) if v]) or set()
        allc, matchable = [], []
        for m_ in MODELS:
            pth = f"{RES}/dom-{dom}-{m_}-expansion__probe_plans.jsonl"
            a_, b_ = label_level_exclusion(pth, leaf, leafvocab_tokens)
            allc += a_; matchable += b_
        out["by_domain"][dom]["label_exclusion_all_cells"] = round(float(np.mean(allc)), 3)
        out["by_domain"][dom]["label_exclusion_matchable_only"] = round(float(np.mean(matchable)), 3)
        out["by_domain"][dom]["n_label_cells"] = len(allc)
        out["by_domain"][dom]["n_label_cells_matchable"] = len(matchable)
        print(f"  label-level exclusion          all {np.mean(allc):.4f} (n={len(allc)})   "
              f"matchable-type only {np.mean(matchable):.4f} (n={len(matchable)})")
        df.to_csv(f"{RES}/r3_instrument_diagnostic_{dom}.csv", index=False)
    json.dump(out, open(f"{RES}/r3_instrument_diagnostic.json", "w"), indent=2)
    print("\nwrote", f"{RES}/r3_instrument_diagnostic.json")


if __name__ == "__main__":
    main()
