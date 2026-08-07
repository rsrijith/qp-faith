#!/usr/bin/env python
"""R1.1 step 3: score the author's adjudicated motif sample.

Reads results/r1_1/ADJUDICATION_SAMPLE.csv after the `human_label` column is filled and computes:
  - LLM-vs-human agreement (raw + Cohen's kappa), overall and on the random rows;
  - the confirmed TYPE-LESS PURITY of the probe (fraction of motifs the human confirms type-less),
    per source, on the random rows, with a bootstrap CI;
  - the verified-clean subset (human == type-less) and the impurities to exclude;
  - a re-report of the WANDS expansion category-routing rate on the verified-clean subset vs the
    full 55, from the cached probe plans (the headline-instrument sensitivity R1.1 asks for).

Run (after filling human_label): ./.venv/bin/python scripts/score_adjudication.py
"""
import os, csv, json, collections
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
OUTDIR = os.path.join(RES, "r1_1")
RNG = np.random.default_rng(13)
LABELS = ["type-less", "genuine-product-type", "ambiguous"]
MODELS = ["qwen4", "llama4", "mistral4", "gemma4", "haiku", "sonnet", "together-llama70", "gpt4o", "gpt52"]


def kappa(a, b):
    cats = LABELS
    n = len(a)
    if n == 0:
        return None
    po = np.mean([x == y for x, y in zip(a, b)])
    ca = collections.Counter(a); cb = collections.Counter(b)
    pe = sum((ca.get(c, 0) / n) * (cb.get(c, 0) / n) for c in cats)
    return round((po - pe) / (1 - pe), 3) if pe < 1 else 1.0


def boot_frac(flags, B=5000):
    a = np.array(flags, float); n = len(a)
    if n == 0:
        return (None, None, None)
    b = [a[RNG.integers(0, n, n)].mean() for _ in range(B)]
    return (round(float(a.mean()), 3), round(float(np.percentile(b, 2.5)), 3), round(float(np.percentile(b, 97.5)), 3))


def emitted_pt(plan):
    if not isinstance(plan, dict):
        return False
    v = plan.get("product_type")
    return v is not None and str(v).strip().lower() not in ("", "none", "null", "n/a", "na", "nan")


def wands_routing_on_subset(clean_motifs):
    """Expansion category-routing rate on the WANDS entity probe, full 55 vs clean subset."""
    full, clean = [], []
    for m in MODELS:
        p = f"{RES}/probe-{m}-exp__probe_plans.jsonl"
        if not os.path.exists(p):
            continue
        for l in open(p):
            r = json.loads(l)
            if r.get("family") != "A_entity":
                continue
            e = 1.0 if emitted_pt(r.get("plan")) else 0.0
            full.append(e)
            if r["query"] in clean_motifs:
                clean.append(e)
    def ci(v):
        v = np.array(v, float)
        if not len(v): return (None, None, None)
        b = [v[RNG.integers(0, len(v), len(v))].mean() for _ in range(5000)]
        return (round(float(v.mean()), 3), round(float(np.percentile(b, 2.5)), 3), round(float(np.percentile(b, 97.5)), 3))
    return {"full_55": ci(full), "clean_subset": ci(clean), "n_clean_motifs": len(clean_motifs)}


def main():
    path = os.path.join(OUTDIR, "ADJUDICATION_SAMPLE.csv")
    rows = list(csv.DictReader(open(path)))
    filled = [r for r in rows if r.get("human_label", "").strip().lower() in LABELS]
    if len(filled) < 0.5 * len(rows):
        print(f"Only {len(filled)}/{len(rows)} rows have a valid human_label. "
              f"Fill the human_label column (type-less / genuine-product-type / ambiguous) first.")
        raise SystemExit(1)
    for r in filled:
        r["human_label"] = r["human_label"].strip().lower()

    out = {"n_rows": len(rows), "n_adjudicated": len(filled)}
    # agreement
    ll = [r["llm_label"] for r in filled]
    hh = [r["human_label"] for r in filled]
    out["agreement_raw"] = round(float(np.mean([a == b for a, b in zip(ll, hh)])), 3)
    out["agreement_kappa"] = kappa(ll, hh)
    rand = [r for r in filled if r.get("random_flag") == "1"]
    out["agreement_kappa_random"] = kappa([r["llm_label"] for r in rand], [r["human_label"] for r in rand])

    # confirmed type-less purity on random rows, per source
    out["typeless_purity"] = {}
    for src in ["WANDS", "ESCI"]:
        rr = [r for r in rand if r["source"] == src]
        out["typeless_purity"][src] = {
            "n_random": len(rr),
            "human_typeless_frac": boot_frac([1.0 if r["human_label"] == "type-less" else 0.0 for r in rr])}
    # human label breakdown
    out["human_label_counts"] = {src: dict(collections.Counter(
        r["human_label"] for r in filled if r["source"] == src)) for src in ["WANDS", "ESCI"]}

    # verified-clean subset + impurities
    clean_wands = {r["motif"] for r in filled if r["source"] == "WANDS" and r["human_label"] == "type-less"}
    impure_wands = [(r["motif"], r["human_label"]) for r in filled
                    if r["source"] == "WANDS" and r["human_label"] != "type-less"]
    out["wands_clean_motif_count"] = len(clean_wands)
    out["wands_impurities"] = impure_wands
    # re-report the WANDS routing headline on the clean subset
    out["wands_expansion_routing"] = wands_routing_on_subset(clean_wands)

    json.dump(out, open(os.path.join(OUTDIR, "adjudication_result.json"), "w"), indent=2)
    def f(t): return f"{t[0]} [{t[1]}, {t[2]}]" if t and t[0] is not None else "n/a"
    print(f"adjudicated {len(filled)}/{len(rows)} rows")
    print(f"LLM-human agreement: raw {out['agreement_raw']}, kappa {out['agreement_kappa']} "
          f"(random rows kappa {out['agreement_kappa_random']})")
    for src in ["WANDS", "ESCI"]:
        p = out["typeless_purity"][src]
        print(f"  {src} type-less purity (random n={p['n_random']}): {f(p['human_typeless_frac'])}")
    print(f"WANDS impurities (human != type-less): {out['wands_impurities']}")
    wr = out["wands_expansion_routing"]
    print(f"WANDS expansion routing: full-55 {f(wr['full_55'])} vs clean-subset "
          f"(n={wr['n_clean_motifs']}) {f(wr['clean_subset'])}")
    print("wrote", os.path.join(OUTDIR, "adjudication_result.json"))


if __name__ == "__main__":
    main()
