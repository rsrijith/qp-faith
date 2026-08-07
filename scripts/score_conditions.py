#!/usr/bin/env python
"""R1.2 scorer: the susceptibility ladder. Reads the cond-*__probe_plans.jsonl files the
sweep writes and computes, per (condition, planner), the CATEGORY routing rate on the 55
type-less entity motifs = fraction of motifs where the planner emits a non-empty
product_type (spurious by construction). Also reports the other-slot emission rates.

This is the R1.2 evidence separating susceptibility (aggressive prompt) from default
behavior (stock / practitioner prompts): the same planners route ~100% under the
aggressive prompt and near 0% under stock/default prompts. Safe to run while the sweep is
still writing (bad/partial trailing lines are skipped). Writes results/score_conditions.json.

Recompute: ./.venv/bin/python scripts/score_conditions.py
"""
import os, glob, json, collections

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
SLOTS = ["product_type", "color", "material", "style", "brand", "price_range"]


def emitted(plan, s):
    """Robust across models: a slot counts as emitted only for a non-empty, non-null string
    value (guards against a model leaving `"product_type": null` in the JSON, which a naive
    truthiness check would miscount)."""
    if not isinstance(plan, dict):
        return False
    v = plan.get(s)
    if v is None:
        return False
    return str(v).strip().lower() not in ("", "none", "null", "n/a", "na", "nan")


def load(fn):
    rows = []
    for line in open(fn):
        try:
            rows.append(json.loads(line))
        except Exception:
            pass  # partial trailing line while the sweep writes; skip
    return rows


def main():
    files = sorted(glob.glob(f"{RES}/cond-*__probe_plans.jsonl"))
    # table[condition][tag] = {slot_emission_rates, n_entity}
    table = collections.defaultdict(dict)
    for fn in files:
        base = os.path.basename(fn).replace("__probe_plans.jsonl", "")[len("cond-"):]
        # Filename is cond-{condition}-{tag}. Condition names use underscores and never
        # contain a hyphen (stock_selfquery, practitioner_narrow, default_extract, ...), while
        # a tag MAY contain a hyphen (together-llama70). So split on the FIRST hyphen: the
        # condition is the prefix, the tag is everything after it (hyphenated tags intact).
        cond, tag = base.split("-", 1)
        rows = [r for r in load(fn) if r.get("family") == "A_entity"]
        n = len(rows)
        if n == 0:
            continue
        emit = {}
        for s in SLOTS:
            emit[s] = round(sum(1 for r in rows if emitted(r.get("plan"), s)) / n, 3)
        table[cond][tag] = {"category_routing_rate": emit["product_type"],
                            "emission": emit, "n_entity": n}

    out = {"metric": "category_routing_rate = fraction of the 55 type-less entity motifs where the "
                     "planner emits a non-empty product_type (spurious by construction).",
           "by_condition": {c: table[c] for c in sorted(table)}}
    json.dump(out, open(f"{RES}/score_conditions.json", "w"), indent=2)

    # print the ladder: rows = conditions, cols = planners
    conds = sorted(table)
    tags = sorted({t for c in table for t in table[c]})
    print(f"CATEGORY ROUTING RATE on 55 type-less motifs (R1.2 susceptibility ladder)")
    print(f"{'condition':<22}" + "".join(f"{t:>10}" for t in tags))
    for c in conds:
        cells = []
        for t in tags:
            v = table[c].get(t, {}).get("category_routing_rate")
            cells.append(f"{v:>10}" if v is not None else f"{'-':>10}")
        print(f"{c:<22}" + "".join(cells))
    print(f"\n(partial while the sweep runs; n_entity per cell in results/score_conditions.json)")
    print("wrote", f"{RES}/score_conditions.json")


if __name__ == "__main__":
    main()
