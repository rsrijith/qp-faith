#!/usr/bin/env python
"""R1.5 mitigation-by-type: evaluate the omit-instruction fix on the COMPLETE organic
WANDS query set (480), separated by query type and model, with the metrics the reviewer
asked for: category-extraction precision/recall, FALSE OMISSION (on genuinely-typed
queries) and FALSE APPLICATION (on type-less queries).

Ground truth is the paper's automatic query-attribute reference
(data/gold/auto_reference.jsonl): a query is TYPED if its `explicit` block contains a
product_type, else TYPE-LESS. On WANDS/480 this is 419 typed + 61 type-less.

We compare two conditions on the SAME 480 queries, per model:
  - expansion       : the aggressive baseline (`{model}-exp__plans.jsonl`)
  - omit_instruction: the proposed fix     (`cond-omit_instruction-{model}__wands_plans.jsonl`)

Definitions (emission-based, which is what false-omission / false-application mean):
  emit(q)          = the plan carries a non-empty product_type
  false_application= P(emit | type-less)   [want LOW; this is the harm the fix targets]
  false_omission   = P(not emit | typed)   [want LOW; the fix must not break typed queries]
  category_recall  = P(emit | typed)        = 1 - false_omission
  category_precision = P(typed | emit) = TP/(TP+FP), TP=emit&typed, FP=emit&type-less
We also report value_recall = P(emitted product_type token-matches the reference type | typed),
a stricter check that the retained category is the right one.

Query-clustered bootstrap (seed 13). Uses only cached plans; $0. Hosted models are
scored too when both their expansion and omit plans are present (skipped otherwise).
Run: ./.venv/bin/python scripts/mitigation_by_type.py
"""
import os, re, json, glob
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
DATA = os.path.join(HERE, "..", "data")
RNG = np.random.default_rng(13)

LOCAL = ["qwen4", "llama4", "mistral4", "gemma4"]
# hosted expansion baselines on organic WANDS are the framework runs fr-{tag}-exp;
# their omit runs are cond-omit_instruction-{tag}__wands. Scored only if both exist.
HOSTED = ["haiku", "sonnet"]


def norm(s):
    return re.sub(r"[^a-z0-9 ]", " ", str(s).lower()).strip()

def toks(s):
    return set(norm(s).split())

def emitted_pt(plan):
    if not isinstance(plan, dict):
        return False
    v = plan.get("product_type")
    if v is None:
        return False
    return str(v).strip().lower() not in ("", "none", "null", "n/a", "na", "nan")

def pt_value(plan):
    return str((plan or {}).get("product_type", "")) if isinstance(plan, dict) else ""


def raw_has_json(raw):
    """A generation is PARSEABLE iff its raw text contains a brace-balanced JSON object that
    json.loads-es (even {}). A prose/chain-of-thought generation with no such object is a
    JSON-FORMAT FAILURE, which must NOT be counted as a category omission (score_v2.py caveat:
    a model emitting fewer slots through parse failure is not more faithful)."""
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
                    json.loads(raw[i:j + 1])
                    return True
                except Exception:
                    return False
    return False


def load_plans(path):
    """Return {query_id: {"plan": dict, "parseable": bool}} or None if absent."""
    if not os.path.exists(path):
        return None
    out = {}
    for l in open(path):
        try:
            r = json.loads(l)
            out[int(r["query_id"])] = {"plan": r.get("plan") or {},
                                       "parseable": raw_has_json(r.get("raw", ""))}
        except Exception:
            pass
    return out


def cluster_ci(vals, B=5000):
    a = np.array(vals, float)
    n = len(a)
    if n == 0:
        return (None, None, None)
    b = [a[RNG.integers(0, n, n)].mean() for _ in range(B)]
    return (round(float(a.mean()), 4), round(float(np.percentile(b, 2.5)), 4),
            round(float(np.percentile(b, 97.5)), 4))


def score_condition(plans, typed_ids, typeless_ids, ref_type):
    """Return the R1.5 metric bundle for one (model, condition). Emission metrics are computed
    ONLY over PARSEABLE generations; JSON-format failures are reported separately so a model
    that drops the category by breaking format is not credited with a clean omission."""
    typed_all = sorted(typed_ids & set(plans))
    typeless_all = sorted(typeless_ids & set(plans))
    n_all = len(typed_all) + len(typeless_all)
    n_fail = sum(1 for q in typed_all + typeless_all if not plans[q]["parseable"])
    # parseable subset
    typed = [q for q in typed_all if plans[q]["parseable"]]
    typeless = [q for q in typeless_all if plans[q]["parseable"]]
    emit_typed = [1.0 if emitted_pt(plans[q]["plan"]) else 0.0 for q in typed]
    emit_typeless = [1.0 if emitted_pt(plans[q]["plan"]) else 0.0 for q in typeless]
    val_correct = []
    for q in typed:
        if emitted_pt(plans[q]["plan"]):
            ok = bool(toks(pt_value(plans[q]["plan"])) & toks(ref_type.get(q, "")))
            val_correct.append(1.0 if ok else 0.0)
        else:
            val_correct.append(0.0)
    tp = sum(emit_typed)
    fp = sum(emit_typeless)
    precision = (tp / (tp + fp)) if (tp + fp) > 0 else None
    return {
        "parse_failure_rate": round(n_fail / n_all, 4) if n_all else None,
        "n_parseable_typed": len(typed), "n_parseable_typeless": len(typeless),
        "false_application": cluster_ci(emit_typeless),        # emit on type-less (harm)
        "false_omission": cluster_ci([1.0 - x for x in emit_typed]),  # not-emit on typed
        "category_recall": cluster_ci(emit_typed),             # emit on typed
        "value_recall": cluster_ci(val_correct),               # emit AND right type
        "category_precision": round(precision, 4) if precision is not None else None,
    }


def main():
    ref = [json.loads(l) for l in open(f"{DATA}/gold/auto_reference.jsonl")]
    typed_ids, typeless_ids, ref_type = set(), set(), {}
    for r in ref:
        q = int(r["query_id"])
        pt = (r.get("explicit") or {}).get("product_type")
        if pt:
            typed_ids.add(q); ref_type[q] = str(pt)
        else:
            typeless_ids.add(q)

    out = {"design": "R1.5 mitigation-by-type on 480 organic WANDS queries; ground truth = "
                     "auto_reference.explicit.product_type (typed vs type-less); expansion "
                     "(aggressive) vs omit_instruction (fix); emission-based false-omission / "
                     "false-application; query-clustered bootstrap; seed 13.",
           "n_typed": len(typed_ids), "n_typeless": len(typeless_ids), "by_model": {}}

    def exp_path(tag, hosted):
        return f"{RES}/{'fr-' if hosted else ''}{tag}-exp__plans.jsonl"

    rows = []
    for tag, hosted in [(m, False) for m in LOCAL] + [(m, True) for m in HOSTED]:
        exp = load_plans(exp_path(tag, hosted))
        omit = load_plans(f"{RES}/cond-omit_instruction-{tag}__wands_plans.jsonl")
        if exp is None or omit is None:
            continue
        out["by_model"][tag] = {
            "expansion": score_condition(exp, typed_ids, typeless_ids, ref_type),
            "omit_instruction": score_condition(omit, typed_ids, typeless_ids, ref_type),
        }
        rows.append(tag)

    # pooled across the local models (apples-to-apples: paper's expansion level vs fix),
    # over the PARSEABLE subset only, with a pooled parse-failure rate reported alongside
    def pool(cond):
        fa, fo, fail, tot = [], [], 0, 0
        for tag in LOCAL:
            exp = load_plans(exp_path(tag, False))
            omit = load_plans(f"{RES}/cond-omit_instruction-{tag}__wands_plans.jsonl")
            src = exp if cond == "expansion" else omit
            if src is None:
                continue
            for q in sorted(typeless_ids & set(src)):
                tot += 1
                if not src[q]["parseable"]:
                    fail += 1; continue
                fa.append(1.0 if emitted_pt(src[q]["plan"]) else 0.0)
            for q in sorted(typed_ids & set(src)):
                tot += 1
                if not src[q]["parseable"]:
                    fail += 1; continue
                fo.append(0.0 if emitted_pt(src[q]["plan"]) else 1.0)
        return {"false_application_pooled": cluster_ci(fa),
                "false_omission_pooled": cluster_ci(fo),
                "parse_failure_rate_pooled": round(fail / tot, 4) if tot else None}
    out["pooled_local"] = {"expansion": pool("expansion"),
                           "omit_instruction": pool("omit_instruction")}

    json.dump(out, open(f"{RES}/mitigation_by_type.json", "w"), indent=2)

    def f(t):
        if t is None or t[0] is None:
            return "  n/a  "
        return f"{t[0]:.3f} [{t[1]:.3f},{t[2]:.3f}]"
    print(f"R1.5 MITIGATION-BY-TYPE on organic WANDS ({out['n_typed']} typed / {out['n_typeless']} type-less)")
    print("(emission metrics over PARSEABLE generations only; parse-fail reported separately)")
    print(f"{'model':<10}{'cond':<17}{'parseFail':<10}{'false_appl(typeless)':<24}{'false_omis(typed)':<22}{'cat_prec':<9}")
    for tag in rows:
        for cond in ["expansion", "omit_instruction"]:
            d = out["by_model"][tag][cond]
            pf = f"{d['parse_failure_rate']:.3f}" if d['parse_failure_rate'] is not None else "n/a"
            print(f"{tag:<10}{cond:<17}{pf:<10}{f(d['false_application']):<24}{f(d['false_omission']):<22}"
                  f"{('%.3f'%d['category_precision']) if d['category_precision'] is not None else 'n/a':<9}")
    print("\nPOOLED (4 local):")
    for cond in ["expansion", "omit_instruction"]:
        p = out["pooled_local"][cond]
        print(f"  {cond:<17} parse_fail {p['parse_failure_rate_pooled']}  "
              f"false_application {f(p['false_application_pooled'])}  "
              f"false_omission {f(p['false_omission_pooled'])}")
    print("\nwrote", f"{RES}/mitigation_by_type.json")


if __name__ == "__main__":
    main()
