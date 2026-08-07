#!/usr/bin/env python
"""Significance evidence: how common are type-less queries in REAL e-commerce query sets?

Establishes the failure SURFACE is a real slice of traffic, not a procedural-probe curiosity.
WANDS: from the paper's automatic reference (type-less = no explicit product_type). ESCI: LLM-
classify the real shopper queries (gpt-4o-mini) type-less / genuine-product-type / ambiguous.
Reports the type-less fraction with a bootstrap CI per set. $0-cheap. Restartable.

Run: set -a; . ../GroundLM/.env; set +a; ./.venv/bin/python scripts/typeless_prevalence.py
"""
import os, json, re, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
DATA = os.path.join(HERE, "..", "data")
OUTDIR = os.path.join(RES, "prevalence")
os.makedirs(OUTDIR, exist_ok=True)
RNG = np.random.default_rng(13)

SYSTEM = (
    "You classify a REAL e-commerce shopper search query. Decide whether the query, as typed, names "
    "or clearly implies a specific PRODUCT TYPE / category, or is type-less (a bare entity, motif, "
    "theme, brand-only, or attribute with no product category). Output STRICT JSON "
    '{"label":"..."} with label one of: '
    "type-less (no product category expressed or implied: 'dinosaur', 'aloe', 'nike', 'red'); "
    "genuine-product-type (names/implies a product category: 'bathroom fan', 'running shoes', "
    "'collagen capsules', 'baby gate'); ambiguous (could be read either way). Judge as a shopper would.")


def load_env():
    envp = os.path.join(HERE, "..", "..", "GroundLM", ".env")
    if os.path.exists(envp):
        for line in open(envp):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def ci(flags):
    a = np.array(flags, float)
    b = [a[RNG.integers(0, len(a), len(a))].mean() for _ in range(5000)]
    return (round(float(a.mean()), 3), round(float(np.percentile(b, 2.5)), 3), round(float(np.percentile(b, 97.5)), 3))


def classify_esci():
    load_env()
    from openai import OpenAI
    client = OpenAI()
    queries = [(int(json.loads(l)["query_id"]), json.loads(l)["query"])
               for l in open(f"{DATA}/esci/queries.jsonl")]
    path = os.path.join(OUTDIR, "esci_query_labels.jsonl")
    done = {}
    if os.path.exists(path):
        for l in open(path):
            r = json.loads(l); done[r["query_id"]] = r["label"]
    todo = [(qid, q) for qid, q in queries if qid not in done]
    lock = threading.Lock()

    def one(item):
        qid, q = item
        try:
            r = client.chat.completions.create(model="gpt-4o-mini", temperature=0, max_tokens=20,
                messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": f'Query: "{q}"\nJSON:'}])
            t = r.choices[0].message.content
            lab = json.loads(t[t.find("{"):t.rfind("}") + 1]).get("label", "").strip().lower()
            if lab not in ("type-less", "genuine-product-type", "ambiguous"):
                lab = "ambiguous"
            return {"query_id": qid, "query": q, "label": lab}
        except Exception as e:
            return {"query_id": qid, "query": q, "label": "ERROR", "err": str(e)[:100]}

    if todo:
        with open(path, "a") as f, ThreadPoolExecutor(max_workers=8) as ex:
            for fut in as_completed({ex.submit(one, it): it for it in todo}):
                r = fut.result()
                with lock:
                    f.write(json.dumps(r) + "\n"); f.flush(); done[r["query_id"]] = r["label"]
    return done


def main():
    # WANDS from the automatic reference (type-less = no explicit product_type)
    ref = [json.loads(l) for l in open(f"{DATA}/gold/auto_reference.jsonl")]
    wands_tl = [1.0 if not (r.get("explicit") or {}).get("product_type") else 0.0 for r in ref]
    # ESCI via LLM classification
    esci = classify_esci()
    esci_tl = [1.0 if v == "type-less" else 0.0 for v in esci.values() if v != "ERROR"]

    out = {"design": "type-less prevalence in real query sets; WANDS from auto_reference, ESCI via "
           "gpt-4o-mini classification; bootstrap CIs, seed 13.",
           "WANDS": {"n": len(wands_tl), "typeless_fraction": ci(wands_tl)},
           "ESCI": {"n": len(esci_tl), "typeless_fraction": ci(esci_tl),
                    "label_breakdown": {k: sum(1 for v in esci.values() if v == k)
                                        for k in ["type-less", "genuine-product-type", "ambiguous", "ERROR"]}}}
    json.dump(out, open(f"{RES}/typeless_prevalence.json", "w"), indent=2)
    def f(t): return f"{t[0]} [{t[1]}, {t[2]}]"
    print("=== type-less query prevalence in REAL query sets ===")
    print(f"  WANDS (n={out['WANDS']['n']}): {f(out['WANDS']['typeless_fraction'])}")
    print(f"  ESCI  (n={out['ESCI']['n']}): {f(out['ESCI']['typeless_fraction'])}  breakdown {out['ESCI']['label_breakdown']}")
    print("wrote", f"{RES}/typeless_prevalence.json")


if __name__ == "__main__":
    main()
