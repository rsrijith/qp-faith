#!/usr/bin/env python
"""R1.3 (Reviewer 1, comment 3): make the ESCI analysis per-model and show its
sensitivity to the >=5-Exact selection, with the matching/normalization rules spelled
out. No new inference; reads the same cached plans + pool as esci_routing.py.

Adds over esci_routing.py:
  (d) per-model routing / per-FIRE exclusion / end-to-end loss reported as primary
      (not only the 4-model pooled average).
  (sensitivity) the whole thing recomputed at MIN_EXACT in {3,5,7,10} so a reader can
      see the >=5 cutoff is not doing the work.
  (terminology) the file states this is a COUNTERFACTUAL exclusion analysis on judged
      products (within each query's human-Exact pool), NOT a full-catalog retrieval
      harm. The full-catalog retrieval version is the separate D-earn run.

Writes results/esci_sensitivity.json. Recompute:
  ./.venv/bin/python scripts/esci_sensitivity.py
"""
import os, re, json
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
DATA = os.path.join(HERE, "..", "data")
RNG = np.random.default_rng(13)
MODELS = ["qwen4", "llama4", "mistral4", "gemma4"]
THRESHOLDS = [3, 5, 7, 10]


def norm(s):
    return re.sub(r"[^a-z0-9 ]", " ", str(s).lower()).strip()


def head(s):
    t = norm(s).split()
    return t[-1] if t else ""


def toks(s):
    return set(norm(s).split())


def facet_match(emitted, val):
    e, v = norm(emitted), norm(val)
    if not e or not v:
        return False
    return e == v or e in v or v in e or bool(toks(e) & toks(v))


def in_query(emitted, q):
    e = head(emitted)
    return bool(e) and (e in toks(q) or norm(emitted) in norm(q))

# R1 comment 3: non-committal placeholders are not spurious guesses (see esci_dearn.py).
NONCOMMITTAL = {"any", "none", "n a", "na", "unknown", "unspecified", "not specified",
 "varies", "various", "varied", "generic", "unbranded", "brandless", "no brand", "no color",
 "no colour", "multiple", "assorted", "misc", "miscellaneous", "not applicable",
 "not available", "other", "standard", "default", "null", "nil", "tbd", "undefined"}
def is_committal(v):
    n = norm(v)
    return bool(n) and n not in NONCOMMITTAL


def cluster_ci(vals, B=5000):
    a = np.array(vals, float)
    n = len(a)
    if n == 0:
        return (None, None, None)
    b = [a[RNG.integers(0, n, n)].mean() for _ in range(B)]
    return (round(float(a.mean()), 3), round(float(np.percentile(b, 2.5)), 3),
            round(float(np.percentile(b, 97.5)), 3))


def load():
    pool = pd.read_parquet(f"{DATA}/esci/pool.parquet")
    E = pool[pool.esci_label == "Exact"]
    exact = {}
    for qid, g in E.groupby("query_id"):
        exact[qid] = [(norm(b), norm(c)) for b, c in zip(g.product_brand, g.product_color)]
    qtext = {int(r.query_id): str(r["query"]) for _, r in
             pd.DataFrame([json.loads(l) for l in open(f"{DATA}/esci/queries.jsonl")]).iterrows()}
    return exact, qtext


def analyze(mode, min_exact, exact, qtext):
    res = {}
    for facet in ["brand", "color"]:
        pm = {m: {"spur": [], "excl": [], "loss": []} for m in MODELS}
        for m in MODELS:
            fn = f"{RES}/esci-{m}-{mode}__plans.jsonl"
            if not os.path.exists(fn):
                continue
            for l in open(fn):
                r = json.loads(l)
                qid = int(r["query_id"])
                p = r.get("plan") or {}
                if not isinstance(p, dict) or qid not in exact or len(exact[qid]) < min_exact:
                    continue
                q = qtext.get(qid, "")
                emitted_raw = str(p.get(facet, "")).strip()
                emitted = emitted_raw if is_committal(emitted_raw) else ""
                spurious = bool(emitted) and not in_query(emitted, q)
                pm[m]["spur"].append(1.0 if spurious else 0.0)
                if spurious:
                    kept = np.mean([1.0 if facet_match(emitted, ev[0 if facet == "brand" else 1]) else 0.0
                                    for ev in exact[qid]])
                    pm[m]["excl"].append(1.0 - kept)
                    pm[m]["loss"].append(1.0 - kept)
                else:
                    pm[m]["loss"].append(0.0)
        res[facet] = {
            "by_model": {m: {
                "spurious_routing_rate": round(float(np.mean(pm[m]["spur"])), 3) if pm[m]["spur"] else None,
                "per_fire_exclusion": round(float(np.mean(pm[m]["excl"])), 3) if pm[m]["excl"] else None,
                "end_to_end_loss": round(float(np.mean(pm[m]["loss"])), 3) if pm[m]["loss"] else None,
                "n_fires": len(pm[m]["excl"]),
            } for m in MODELS},
            "pooled": {
                "spurious_routing_rate": cluster_ci([x for m in MODELS for x in pm[m]["spur"]]),
                "per_fire_exclusion": cluster_ci([x for m in MODELS for x in pm[m]["excl"]]),
                "end_to_end_loss": cluster_ci([x for m in MODELS for x in pm[m]["loss"]]),
            },
            "n_queries": len(pm[MODELS[0]]["spur"]),
        }
    return res


def main():
    exact, qtext = load()
    out = {
        "framing": "COUNTERFACTUAL exclusion analysis on judged products: exclusion is measured "
                   "within each query's human-Exact pool, not by retrieval over the full catalog "
                   "(that is the separate D-earn full-catalog run). 4 local planners, cached plans.",
        "normalization_rules": {
            "case": "all values lowercased.",
            "punctuation": "non-alphanumeric characters collapsed to spaces (norm()).",
            "missing_facet_value": "an empty product brand/color never matches an emitted value "
                                   "(counts against 'kept'); an empty EMITTED value is not a fire.",
            "multi_valued_fields": "token-overlap match, so a value overlapping any token of a "
                                   "multi-token product value counts as a match (lenient; biases "
                                   "toward UNDER-counting exclusion).",
            "aliases_abbreviations": "NOT normalized here (e.g. 'iphone case' -> 'Apple' is not "
                                     "inferred). The conservative routing rate bounds this; the "
                                     "human entailment audit (R1.3e) quantifies the residual.",
            "spurious_definition": "emitted value whose head token is absent from the query text.",
        },
        "sensitivity_min_exact": {},
    }
    for th in THRESHOLDS:
        out["sensitivity_min_exact"][str(th)] = {
            lab: analyze(mode, th, exact, qtext) for mode, lab in [("exp", "expansion"), ("con", "conservative")]
        }
    json.dump(out, open(f"{RES}/esci_sensitivity.json", "w"), indent=2)

    print("ESCI sensitivity to the min-Exact cutoff (expansion; per-fire exclusion is on human-Exact pools)")
    for th in THRESHOLDS:
        d = out["sensitivity_min_exact"][str(th)]["expansion"]
        b, c = d["brand"], d["color"]
        print(f"  min_exact={th:>2} (n={b['n_queries']:>3}): "
              f"brand route {b['pooled']['spurious_routing_rate'][0]} loss {b['pooled']['end_to_end_loss'][0]} | "
              f"color route {c['pooled']['spurious_routing_rate'][0]} loss {c['pooled']['end_to_end_loss'][0]}")
    print("\nper-model at min_exact=5 (expansion), brand:")
    for m in MODELS:
        bm = out["sensitivity_min_exact"]["5"]["expansion"]["brand"]["by_model"][m]
        print(f"  {m:<9} route {bm['spurious_routing_rate']}  per-fire-excl {bm['per_fire_exclusion']}  "
              f"end-to-end {bm['end_to_end_loss']}  (n_fires={bm['n_fires']})")
    print("\nwrote", f"{RES}/esci_sensitivity.json")


if __name__ == "__main__":
    main()
