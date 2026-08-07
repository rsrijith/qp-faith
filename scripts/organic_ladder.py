#!/usr/bin/env python
"""R1.2 robustness + R1.8 reasoning check on ORGANIC WANDS queries.

The probe ladder (score_conditions.py) measures category routing on the 55 constructed
type-less motifs. R1.2/R2.2 ask whether that susceptibility replicates on real organic
queries rather than only on the procedurally-generated probe. Here we score the same four
ecological prompts across all nine planners on the 61 genuinely type-less ORGANIC WANDS
queries (auto_reference.explicit.product_type absent), reporting the category
FALSE-APPLICATION rate (spurious product_type emitted) with parse-failure separated and a
query-clustered bootstrap. As a control we also report the rate on the 419 typed organic
queries, where emitting a category is CORRECT (so a high rate there is expected, not harm).

R1.8: gpt-5.2 is a reasoning model; gpt-4o is not. Both are OpenAI planners run through the
identical path, so the gpt-5.2-vs-gpt-4o column contrast is a reasoning-depth sensitivity
read on identical prompts.

Uses only cached sweep plans; $0. Run: ./.venv/bin/python scripts/organic_ladder.py
"""
import os, re, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
DATA = os.path.join(HERE, "..", "data")
RNG = np.random.default_rng(13)

CONDS = ["stock_selfquery", "default_extract", "practitioner_shopping", "practitioner_narrow"]
TAGS = ["qwen4", "llama4", "mistral4", "gemma4", "haiku", "sonnet",
        "together-llama70", "gpt4o", "gpt52"]


def raw_has_json(raw):
    if not raw:
        return False
    i = raw.find("{")
    if i < 0:
        return False
    depth = 0
    for j in range(i, len(raw)):
        if raw[j] == "{":
            depth += 1
        elif raw[j] == "}":
            depth -= 1
            if depth == 0:
                try:
                    json.loads(raw[i:j + 1]); return True
                except Exception:
                    return False
    return False


def emitted_pt(plan):
    if not isinstance(plan, dict):
        return False
    v = plan.get("product_type")
    if v is None:
        return False
    return str(v).strip().lower() not in ("", "none", "null", "n/a", "na", "nan")


def cluster_ci(vals, B=5000):
    a = np.array(vals, float); n = len(a)
    if n == 0:
        return (None, None, None)
    b = [a[RNG.integers(0, n, n)].mean() for _ in range(B)]
    return (round(float(a.mean()), 3), round(float(np.percentile(b, 2.5)), 3),
            round(float(np.percentile(b, 97.5)), 3))


def main():
    ref = [json.loads(l) for l in open(f"{DATA}/gold/auto_reference.jsonl")]
    typeless = {int(r["query_id"]) for r in ref if not (r.get("explicit") or {}).get("product_type")}
    typed = {int(r["query_id"]) for r in ref if (r.get("explicit") or {}).get("product_type")}

    out = {"design": "category false-application on 61 type-less ORGANIC WANDS queries (control: "
                     "419 typed), four ecological prompts x 9 planners; parse-failure separated; "
                     "query-clustered bootstrap; seed 13.",
           "n_typeless": len(typeless), "n_typed": len(typed), "by_condition": {}}

    for cond in CONDS:
        out["by_condition"][cond] = {}
        for tag in TAGS:
            fn = f"{RES}/cond-{cond}-{tag}__wands_plans.jsonl"
            if not os.path.exists(fn):
                continue
            recs = {}
            for l in open(fn):
                try:
                    r = json.loads(l)
                    recs[int(r["query_id"])] = (r.get("plan") or {}, raw_has_json(r.get("raw", "")))
                except Exception:
                    pass
            tl = [q for q in typeless if q in recs]
            ty = [q for q in typed if q in recs]
            tl_ok = [q for q in tl if recs[q][1]]
            ty_ok = [q for q in ty if recs[q][1]]
            fa = [1.0 if emitted_pt(recs[q][0]) else 0.0 for q in tl_ok]
            ta = [1.0 if emitted_pt(recs[q][0]) else 0.0 for q in ty_ok]
            pf = 1.0 - (len(tl_ok) + len(ty_ok)) / (len(tl) + len(ty)) if (len(tl) + len(ty)) else None
            out["by_condition"][cond][tag] = {
                "false_application_typeless": cluster_ci(fa),
                "correct_application_typed": cluster_ci(ta),
                "parse_failure_rate": round(pf, 3) if pf is not None else None,
                "n_typeless_parseable": len(tl_ok)}

    json.dump(out, open(f"{RES}/organic_ladder.json", "w"), indent=2)

    print(f"R1.2 ROBUSTNESS: category FALSE-APPLICATION on {out['n_typeless']} type-less ORGANIC WANDS queries")
    print("(spurious product_type on a genuinely type-less query; parse-fail separated)")
    print(f"{'condition':<22}" + "".join(f"{t[:8]:>9}" for t in TAGS))
    for cond in CONDS:
        cells = []
        for t in TAGS:
            d = out["by_condition"][cond].get(t, {})
            v = d.get("false_application_typeless", (None,))[0]
            cells.append(f"{v:>9.2f}" if v is not None else f"{'-':>9}")
        print(f"{cond:<22}" + "".join(cells))
    print("\nR1.8 reasoning read — gpt52 (reasoning) vs gpt4o (not), false-application on type-less:")
    for cond in CONDS:
        g52 = out["by_condition"][cond].get("gpt52", {}).get("false_application_typeless", (None,))[0]
        g4o = out["by_condition"][cond].get("gpt4o", {}).get("false_application_typeless", (None,))[0]
        print(f"  {cond:<22} gpt4o {g4o}   gpt52 {g52}")
    print("\nParse-failure flags (any planner >0.10 under a condition):")
    for cond in CONDS:
        for t in TAGS:
            pf = out["by_condition"][cond].get(t, {}).get("parse_failure_rate")
            if pf is not None and pf > 0.10:
                print(f"  {cond} / {t}: parse_failure {pf}")
    print("\nwrote", f"{RES}/organic_ladder.json")


if __name__ == "__main__":
    main()
