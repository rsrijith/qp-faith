#!/usr/bin/env python
"""R1.7 (Reviewer 1, comment 7): the pooled ESCI interval clusters over queries while
treating the 4 selected models as fixed, so it looks more precise than the cross-model
evidence warrants. This recomputes the pooled end-to-end loss with a HIERARCHICAL
bootstrap over models AND queries, reports it beside the query-only interval, defines
macro vs micro, and checks the pooled-routing x pooled-per-fire product against direct
per-query aggregation. No new inference. Recompute:
  ./.venv/bin/python scripts/hier_bootstrap.py
"""
import os, re, json
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
DATA = os.path.join(HERE, "..", "data")
RNG = np.random.default_rng(13)
MODELS = ["qwen4", "llama4", "mistral4", "gemma4"]
MIN_EXACT = 5


def norm(s):
    return re.sub(r"[^a-z0-9 ]", " ", str(s).lower()).strip()


def head(s):
    t = norm(s).split(); return t[-1] if t else ""


def toks(s):
    return set(norm(s).split())


def facet_match(e, v):
    e, v = norm(e), norm(v)
    return bool(e) and bool(v) and (e == v or e in v or v in e or bool(toks(e) & toks(v)))


def in_query(e, q):
    h = head(e); return bool(h) and (h in toks(q) or norm(e) in norm(q))


# R1 comment 3: non-committal placeholders are not spurious guesses (see esci_dearn.py).
NONCOMMITTAL = {"any", "none", "n a", "na", "unknown", "unspecified", "not specified",
 "varies", "various", "varied", "generic", "unbranded", "brandless", "no brand", "no color",
 "no colour", "multiple", "assorted", "misc", "miscellaneous", "not applicable",
 "not available", "other", "standard", "default", "null", "nil", "tbd", "undefined"}
def is_committal(v):
    n = norm(v); return bool(n) and n not in NONCOMMITTAL


def per_query_loss(mode, facet):
    """Return {model: {qid: end_to_end_loss}} over the >=MIN_EXACT queries."""
    pool = pd.read_parquet(f"{DATA}/esci/pool.parquet")
    E = pool[pool.esci_label == "Exact"]
    exact = {}
    for qid, g in E.groupby("query_id"):
        exact[qid] = [(norm(b), norm(c)) for b, c in zip(g.product_brand, g.product_color)]
    qtext = {int(r.query_id): str(r["query"]) for _, r in
             pd.DataFrame([json.loads(l) for l in open(f"{DATA}/esci/queries.jsonl")]).iterrows()}
    out = {m: {} for m in MODELS}
    for m in MODELS:
        fn = f"{RES}/esci-{m}-{mode}__plans.jsonl"
        if not os.path.exists(fn):
            continue
        for l in open(fn):
            r = json.loads(l); qid = int(r["query_id"]); p = r.get("plan") or {}
            if not isinstance(p, dict) or qid not in exact or len(exact[qid]) < MIN_EXACT:
                continue
            q = qtext.get(qid, ""); emitted_raw = str(p.get(facet, "")).strip()
            emitted = emitted_raw if is_committal(emitted_raw) else ""
            if emitted and not in_query(emitted, q):
                kept = np.mean([1.0 if facet_match(emitted, ev[0 if facet == "brand" else 1]) else 0.0
                                for ev in exact[qid]])
                out[m][qid] = 1.0 - kept
            else:
                out[m][qid] = 0.0
    return out


def query_only_ci(pooled_vals, B=5000):
    a = np.array(pooled_vals, float)
    b = [a[RNG.integers(0, len(a), len(a))].mean() for _ in range(B)]
    return round(float(a.mean()), 3), round(float(np.percentile(b, 2.5)), 3), round(float(np.percentile(b, 97.5)), 3)


def hierarchical_ci(by_model, B=5000):
    """Two-stage: resample the 4 models with replacement, then resample each drawn
    model's queries with replacement; the statistic is the mean of the resampled
    per-model query-means (models weighted equally = the fixed-model estimand's target,
    but now carrying between-model variance)."""
    model_qvals = {m: np.array(list(by_model[m].values()), float) for m in MODELS if by_model[m]}
    ms = [m for m in MODELS if m in model_qvals]
    point = float(np.mean([model_qvals[m].mean() for m in ms]))
    boots = []
    for _ in range(B):
        drawn = [ms[i] for i in RNG.integers(0, len(ms), len(ms))]
        means = []
        for m in drawn:
            qv = model_qvals[m]
            means.append(qv[RNG.integers(0, len(qv), len(qv))].mean())
        boots.append(np.mean(means))
    return round(point, 3), round(float(np.percentile(boots, 2.5)), 3), round(float(np.percentile(boots, 97.5)), 3)


def main():
    out = {
        "estimand_note": "Two pooled summaries are reported. The QUERY-ONLY interval treats the 4 "
                         "models as fixed and clusters over queries (what the submitted paper reported). "
                         "The HIERARCHICAL interval resamples models AND queries, so it reflects the "
                         "much larger between-model variation. Per-model estimates (esci_sensitivity.json) "
                         "remain the primary evidence; the hierarchical interval is the honest pooled CI.",
        "definitions": {
            "macro_average": "mean over queries of the per-query end-to-end loss (each query weighted "
                             "equally); this is what both intervals below summarize.",
            "micro_average": "mean over query-model FIRES of the per-fire exclusion (each fire weighted "
                             "equally); reported in esci_sensitivity.json as per_fire_exclusion.",
            "end_to_end_loss": "per query, spurious ? per-fire-exclusion : 0 (absolute recall difference "
                               "on the human-Exact pool, not a relative reduction).",
        },
        "results": {},
    }
    for facet in ["brand", "color"]:
        bm = per_query_loss("exp", facet)
        pooled_vals = [v for m in MODELS for v in bm[m].values()]
        qonly = query_only_ci(pooled_vals)
        hier = hierarchical_ci(bm)
        # reconcile pooled routing x pooled per-fire vs direct per-query macro loss
        route = float(np.mean([1.0 if v > 0 else 0.0 for v in pooled_vals]))  # fires / queries
        fires = [v for v in pooled_vals if v > 0]
        perfire = float(np.mean(fires)) if fires else 0.0
        out["results"][facet] = {
            "macro_end_to_end_loss_query_only_ci": qonly,
            "macro_end_to_end_loss_hierarchical_ci": hier,
            "reconciliation": {
                "pooled_routing_rate": round(route, 3),
                "pooled_per_fire_exclusion": round(perfire, 3),
                "product": round(route * perfire, 3),
                "direct_per_query_macro_loss": qonly[0],
                "match": abs(route * perfire - qonly[0]) < 0.005,
            },
        }
    json.dump(out, open(f"{RES}/hier_bootstrap.json", "w"), indent=2)
    for facet in ["brand", "color"]:
        r = out["results"][facet]
        q = r["macro_end_to_end_loss_query_only_ci"]; h = r["macro_end_to_end_loss_hierarchical_ci"]
        rec = r["reconciliation"]
        print(f"{facet}: query-only {q[0]} [{q[1]}, {q[2]}]  vs  hierarchical {h[0]} [{h[1]}, {h[2]}]  "
              f"(width {q[2]-q[1]:.3f} -> {h[2]-h[1]:.3f})")
        print(f"       reconcile: routing {rec['pooled_routing_rate']} x per-fire {rec['pooled_per_fire_exclusion']} "
              f"= {rec['product']} vs direct {rec['direct_per_query_macro_loss']}  match={rec['match']}")
    print("wrote", f"{RES}/hier_bootstrap.json")


if __name__ == "__main__":
    main()
