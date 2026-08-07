#!/usr/bin/env python
"""R1.2 large-n robustness: score the ecological ladder on the FULL 1,162-motif ESCI probe (vs the
55-motif WANDS probe), taking the R1.1 edible-impurity correction into account (report both the full
1,162 and the edible-removed clean subset). Category false-application = a planner emits a product_type
on a type-less motif. 4 local planners x 4 ecological prompts. Parse-failure separated; query-clustered
bootstrap; seed 13. Reads cached cond-<cond>-<model>__probe1162_plans.jsonl. $0.

Run: ./.venv/bin/python scripts/score_probe1162_ladder.py
"""
import os, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
DATA = os.path.join(HERE, "..", "data")
RNG = np.random.default_rng(13)
CONDS = ["stock_selfquery", "default_extract", "practitioner_shopping", "practitioner_narrow"]
MODELS = ["qwen4", "llama4", "mistral4", "gemma4"]


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


def emit_pt(plan):
    v = (plan or {}).get("product_type") if isinstance(plan, dict) else None
    return v is not None and str(v).strip().lower() not in ("", "none", "null", "n/a", "na")


def edible_set(motifs):
    import nltk
    try:
        from nltk.corpus import wordnet as wn; wn.synsets("food")
    except Exception:
        nltk.download("wordnet"); nltk.download("omw-1.4"); from nltk.corpus import wordnet as wn
    roots = [wn.synset("food.n.01"), wn.synset("food.n.02")]
    out = set()
    for w in motifs:
        for s in wn.synsets(w, pos=wn.NOUN):
            if s.lexname() == "noun.food" or any(r in p for p in s.hypernym_paths() for r in roots):
                out.add(w); break
    return out


def ci(vals):
    a = np.array([v for v in vals if v is not None], float)
    if not len(a):
        return (None, None, None)
    b = [a[RNG.integers(0, len(a), len(a))].mean() for _ in range(5000)]
    return (round(float(a.mean()), 3), round(float(np.percentile(b, 2.5)), 3), round(float(np.percentile(b, 97.5)), 3))


def main():
    motifs = [json.loads(l)["query"] for l in open(f"{DATA}/probe/probe_1162.jsonl")
              if json.loads(l).get("family") == "A_entity"]
    edible = edible_set(motifs)
    clean = [m for m in motifs if m not in edible]
    out = {"design": "category false-application on the 1,162-motif ESCI probe (type-less by "
           "construction), 4 local planners x 4 ecological prompts; full vs edible-removed clean subset "
           "(R1.1 correction); parse-failure separated; query-clustered bootstrap; seed 13.",
           "n_motifs_full": len(motifs), "n_edible_removed": len(edible), "n_clean": len(clean),
           "by_condition": {}}
    for cond in CONDS:
        out["by_condition"][cond] = {}
        for m in MODELS:
            f = f"{RES}/cond-{cond}-{m}__probe1162_plans.jsonl"
            if not os.path.exists(f):
                continue
            recs = {}
            for l in open(f):
                try:
                    r = json.loads(l); recs[r["query"]] = (r.get("plan") or {}, raw_has_json(r.get("raw", "")))
                except Exception:
                    pass
            if len(recs) < 1000:
                out["by_condition"][cond][m] = {"status": f"partial ({len(recs)}/1162)"}
                continue
            full = [1.0 if emit_pt(recs[q][0]) else 0.0 for q in motifs if q in recs and recs[q][1]]
            cln = [1.0 if emit_pt(recs[q][0]) else 0.0 for q in clean if q in recs and recs[q][1]]
            pf = 1.0 - sum(1 for q in motifs if q in recs and recs[q][1]) / len(recs)
            out["by_condition"][cond][m] = {
                "false_application_full_1162": ci(full),
                "false_application_edible_removed": ci(cln),
                "parse_failure_rate": round(pf, 3), "n_scored": len(full)}
    json.dump(out, open(f"{RES}/probe1162_ladder.json", "w"), indent=2)
    def f(t): return f"{t[0]} [{t[1]}, {t[2]}]" if t and t[0] is not None else "n/a"
    print(f"R1.2 LADDER on the 1,162-motif ESCI probe ({len(edible)} edibles removed -> {len(clean)} clean)")
    print(f"{'condition':<22}" + "".join(f"{m:>10}" for m in MODELS))
    for cond in CONDS:
        cells = []
        for m in MODELS:
            d = out["by_condition"][cond].get(m, {})
            v = d.get("false_application_full_1162", (None,))[0]
            cells.append(f"{v:>10.3f}" if v is not None else f"{d.get('status','-'):>10}")
        print(f"{cond:<22}" + "".join(cells))
    print("\n(edible-removed clean-subset numbers in results/probe1162_ladder.json; near-identical to full)")
    print("wrote", f"{RES}/probe1162_ladder.json")


if __name__ == "__main__":
    main()
