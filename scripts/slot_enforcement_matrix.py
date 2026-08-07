#!/usr/bin/env python
"""R1.4 (Reviewer 1, comment 4): the COMPLETE slot x enforcement matrix.

Reviewer 1 asked us to disentangle slot identity from enforcement mode (Figure 3
confounds the two), to report all five slots under identical missing-value rules with
effect sizes and CIs, and to formally define the three exclusion semantics. This
computes, on the 55 WANDS type-less entity motifs (the clean instrument), the label-
level selectivity of each slot's ACTUAL spuriously-emitted value under each enforcement
semantics, per planner, aggregated over the 55-motif cluster (the resampling unit the
rest of the paper uses). No new inference; reads only the cached probe plans
(results/probe-*-exp__probe_plans.jsonl) + data/product.csv.

Value matching reuses validate_enforcement.py's token logic (deplural + subset/equality),
so the category column reconciles with the manuscript's measured enforcement sweep.

Formal definitions. A planner emits a single value v for a slot on a type-less query
(so v is spurious by construction). R is the motif's relevant set; V(p) is product p's
value-set for the slot, empty when the facet is missing:
  full-hard-filter exclusion = 1 - |{p in R : v matches V(p)}| / |R|
        strict equality: a missing facet does NOT match v, so it is EXCLUDED.
  conflict-only exclusion    = |{p in R : V(p) != {} and v does not match V(p)}| / |R|
        a missing facet is RETAINED; only a present, conflicting value excludes.
  soft-penalty loss          = 0 realized recall loss by construction (a soft signal
        demotes, never removes). The measured soft demotion for the category slot is
        small and is reported separately (enforcement sweep / *_soft_metrics.json).

The coverage point R1 flagged falls straight out of the two hard definitions: under
full-hard-filter LOW coverage RAISES exclusion (missing rows dropped); under conflict-
only LOW coverage LOWERS it (missing rows kept). So coverage's sign depends on the
operator; the driver common to both is enforcement mode x relevant-set spread, not the
slot label. Motif-clustered bootstrap CIs (seed 13). Writes
results/slot_enforcement_matrix.json.
"""
import os, re, json, glob, collections
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
RES = os.path.join(HERE, "..", "results")
RNG = np.random.default_rng(13)

# emitted-plan key -> product-side facet keys (category is the product_class column)
SLOT_PLAN_KEY = {"category": "product_type", "brand": "brand", "color": "color",
                 "material": "material", "style": "style"}
SLOT_FEATURE_KEYS = {
    "brand":    {"brand", "fashionbrand"},
    "color":    {"color"},
    "material": {"primarymaterial", "material"},
    "style":    {"style", "dsprimaryproductstyle", "dssecondaryproductstyle"},
}
SLOTS = ["category", "brand", "color", "material", "style"]


def norm(s):
    return re.sub(r"[^a-z0-9 ]", " ", str(s).lower()).strip()


def depl(w):
    return w[:-1] if len(w) > 3 and w.endswith("s") else w


def tokset(s):
    return {depl(w) for w in norm(s).split() if w}


def matches(val, prod_val):
    """Same token logic validate_enforcement.py uses: nonempty subset/equality either way."""
    A = tokset(val)
    B = tokset(prod_val)
    return bool(A) and bool(B) and (A == B or A <= B or B <= A)


def value_matches_set(val, vset):
    return any(matches(val, pv) for pv in vset)


def parse_features(s):
    d = collections.defaultdict(set)
    if isinstance(s, str):
        for part in s.split("|"):
            if ":" in part:
                k, v = part.split(":", 1)
                vn = norm(v)
                if vn:
                    d[k.strip().lower()].add(vn)
    return d


def slot_valueset(row, slot):
    if slot == "category":
        v = row["product_class"]
        return {norm(v)} if pd.notna(v) and norm(v) else set()
    feats = row["_feats"]
    out = set()
    for k in SLOT_FEATURE_KEYS[slot]:
        out |= feats.get(k, set())
    return out


def cluster_ci(motif_means, B=5000):
    a = np.array([m for m in motif_means if m is not None], float)
    n = len(a)
    if n == 0:
        return (None, None, None, 0)
    boots = [a[RNG.integers(0, n, n)].mean() for _ in range(B)]
    return (round(float(a.mean()), 3), round(float(np.percentile(boots, 2.5)), 3),
            round(float(np.percentile(boots, 97.5)), 3), n)


def main():
    p = pd.read_csv(f"{DATA}/product.csv", sep="\t")
    p["_feats"] = p["product_features"].map(parse_features)
    by_pid = {int(r.product_id): r for _, r in p.iterrows()}

    plan_files = sorted(glob.glob(f"{RES}/probe-*-exp__probe_plans.jsonl"))
    # exclude the ESCI-domain probe (ASIN pids), keep WANDS (integer pids)
    tags = []
    for f in plan_files:
        try:
            r0 = json.loads(open(f).readline())
        except Exception:
            continue
        if not r0.get("relevant_pids"):
            continue
        if not str(r0["relevant_pids"][0]).lstrip("-").isdigit():
            continue  # ESCI/ASIN
        tags.append(f)

    # per slot: motif -> list of per-(planner) exclusion values, plus coverage/spread
    per = {s: {"full_hard": collections.defaultdict(list),
               "conflict_only": collections.defaultdict(list)} for s in SLOTS}
    cov_spread = {s: {"coverage": [], "spread": []} for s in SLOTS}
    planners = []

    for f in tags:
        tag = os.path.basename(f).replace("__probe_plans.jsonl", "")
        planners.append(tag)
        for line in open(f):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("family") != "A_entity":
                continue
            plan = r.get("plan") or {}
            if not isinstance(plan, dict):
                continue
            rel = [int(x) for x in r["relevant_pids"]]
            rows = [by_pid[pid] for pid in rel if pid in by_pid]
            R = len(rows)
            if R == 0:
                continue
            motif = r["query"]
            for s in SLOTS:
                v = str(plan.get(SLOT_PLAN_KEY[s], "") or "").strip()
                if not v:
                    continue  # slot not emitted by this planner on this motif
                vsets = [slot_valueset(row, s) for row in rows]
                match = [value_matches_set(v, vs) for vs in vsets]
                present = [vs for vs in vsets if vs]
                full_hard = 1.0 - sum(match) / R
                conflict = sum(1 for vs, mt in zip(vsets, match) if vs and not mt) / R
                per[s]["full_hard"][motif].append(full_hard)
                per[s]["conflict_only"][motif].append(conflict)
                cov_spread[s]["coverage"].append(len(present) / R)
                # spread = distinct normalized facet values among relevant present products
                distinct = len(set().union(*present)) if present else 0
                cov_spread[s]["spread"].append(distinct)

    matrix = {}
    for s in SLOTS:
        # motif-level mean (average planners within a motif), then cluster-bootstrap over motifs
        fh_motif = [float(np.mean(v)) for v in per[s]["full_hard"].values()]
        co_motif = [float(np.mean(v)) for v in per[s]["conflict_only"].values()]
        matrix[s] = {
            "coverage_mean": round(float(np.mean(cov_spread[s]["coverage"])), 3)
            if cov_spread[s]["coverage"] else None,
            "relevant_set_spread_mean_distinct_values": round(float(np.mean(cov_spread[s]["spread"])), 2)
            if cov_spread[s]["spread"] else None,
            "full_hard_filter_exclusion": cluster_ci(fh_motif),
            "conflict_only_exclusion": cluster_ci(co_motif),
            "soft_penalty_realized_recall_loss": 0.0,
            "n_motifs_emitted": len(fh_motif),
            "n_planner_motif_pairs": len(cov_spread[s]["coverage"]),
        }

    out = {
        "instrument": "55 WANDS type-less entity motifs; relevant sets from the cached probe "
                      "plans (probe-*-exp__probe_plans.jsonl, expansion prompt). Any emitted slot "
                      "value on a type-less query is spurious by construction.",
        "planners": planners,
        "matching": "validate_enforcement.py token logic (deplural + nonempty subset/equality).",
        "value_source": "the planner's ACTUAL emitted value per slot per motif (not a guess).",
        "missing_value_rule": "identical across slots: full-hard-filter excludes a missing facet; "
                              "conflict-only retains it.",
        "soft_penalty_note": "soft-penalty realized recall@100 loss is 0 by construction (candidates "
                             "demoted, not removed). Measured soft demotion for the category slot is "
                             "small: mean nDCG@10 drop 0.006-0.025 across the four local planners "
                             "(expansion, dense weight lambda=0.5, results/*_soft_metrics.json); the "
                             "enforcement sweep (results/probe-*__enforce_dense.json) shows realized "
                             "recall@100 loss rising from ~0 at soft lambdas to ~0.72-0.76 only at the "
                             "hard endpoint lambda=0.",
        "matrix": matrix,
    }
    json.dump(out, open(f"{RES}/slot_enforcement_matrix.json", "w"), indent=2)

    def f(t):
        return f"{t[0]} [{t[1]}, {t[2]}] (m={t[3]})" if t and t[0] is not None else "n/a"
    print(f"planners ({len(planners)}):", ", ".join(planners))
    print(f"\n{'slot':<10}{'cov':>6}{'spread':>8}   {'full-hard':>26}{'conflict-only':>26}{'soft':>6}")
    for s in SLOTS:
        d = matrix[s]
        print(f"{s:<10}{str(d['coverage_mean']):>6}{str(d['relevant_set_spread_mean_distinct_values']):>8}   "
              f"{f(d['full_hard_filter_exclusion']):>26}{f(d['conflict_only_exclusion']):>26}"
              f"{d['soft_penalty_realized_recall_loss']:>6}")
    print("\nwrote", f"{RES}/slot_enforcement_matrix.json")
    print("VALIDATION: category full-hard-filter exclusion should reproduce the manuscript's ~0.98.")


if __name__ == "__main__":
    main()
