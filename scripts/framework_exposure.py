#!/usr/bin/env python
"""Significance (the decisive one): does the category mis-routing fire in a REAL, shipped RAG query
constructor, not just under our stress prompt? We run LangChain's actual SelfQueryRetriever query
constructor (`load_query_constructor_runnable`, its shipped prompt) over the 55 type-less motifs with
a realistic product-attribute schema, and check whether the emitted StructuredQuery filter contains a
spurious product_type/category comparison.

Two catalog descriptions, both realistic deployment choices:
  neutral  : "E-commerce catalog of furniture and home-goods products."
  narrowing: a description that asks the constructor to help the shopper narrow to what they want
             (the common production tweak).

This converts "artifact of our aggressive prompt" into "fires in code people ship." Reports the
category-filter rate per (model, description) with a bootstrap CI. Reads OPENAI_API_KEY. Restartable.

Run: set -a; . ../GroundLM/.env; set +a; ./.venv/bin/python scripts/framework_exposure.py
"""
import os, json, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
DATA = os.path.join(HERE, "..", "data")
OUTDIR = os.path.join(RES, "framework")
os.makedirs(OUTDIR, exist_ok=True)
RNG = np.random.default_rng(13)
MODELS = ["gpt-4o", "gpt-4o-mini"]
DESCS = {
    "neutral": "E-commerce catalog of furniture and home-goods products.",
    "narrowing": ("E-commerce catalog of furniture and home-goods products. Infer the filters that best "
                  "narrow the catalog to the products this shopper most likely wants."),
}


def load_env():
    envp = os.path.join(HERE, "..", "..", "GroundLM", ".env")
    if os.path.exists(envp):
        for line in open(envp):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def attr_info():
    from langchain.chains.query_constructor.schema import AttributeInfo
    return [
        AttributeInfo(name="product_type", description="The product category/type, e.g. sofa, lamp, rug, toy", type="string"),
        AttributeInfo(name="color", description="The product color", type="string"),
        AttributeInfo(name="material", description="The product material", type="string"),
        AttributeInfo(name="style", description="The product style", type="string"),
        AttributeInfo(name="brand", description="The product brand", type="string"),
        AttributeInfo(name="price_range", description="The product price range", type="string"),
    ]


def filter_has_producttype(f):
    """Traverse a LangChain StructuredQuery filter (Comparison/Operation tree) for a product_type comparison."""
    if f is None:
        return False
    attr = getattr(f, "attribute", None)
    if attr is not None:
        return str(attr) == "product_type"
    args = getattr(f, "arguments", None)
    if args:
        return any(filter_has_producttype(a) for a in args)
    return False


def ci(vals):
    a = np.array([v for v in vals if v is not None], float)
    if not len(a):
        return (None, None, None)
    b = [a[RNG.integers(0, len(a), len(a))].mean() for _ in range(5000)]
    return (round(float(a.mean()), 3), round(float(np.percentile(b, 2.5)), 3), round(float(np.percentile(b, 97.5)), 3))


def main():
    load_env()
    from langchain.chains.query_constructor.base import load_query_constructor_runnable
    from langchain_openai import ChatOpenAI
    ai = attr_info()
    motifs = [json.loads(l)["query"] for l in open(f"{DATA}/probe/probe.jsonl")
              if json.loads(l).get("family") == "A_entity"]
    path = os.path.join(OUTDIR, "framework_plans.jsonl")
    done = {}
    if os.path.exists(path):
        for l in open(path):
            r = json.loads(l); done[(r["model"], r["desc"], r["motif"])] = r

    runnables = {}
    for m in MODELS:
        for dk, dv in DESCS.items():
            runnables[(m, dk)] = load_query_constructor_runnable(
                ChatOpenAI(model=m, temperature=0), dv, ai)

    jobs = [(m, dk, mo) for m in MODELS for dk in DESCS for mo in motifs if (m, dk, mo) not in done]
    print(f"LangChain SelfQuery constructor: {len(MODELS)} models x {len(DESCS)} descs x {len(motifs)} motifs; "
          f"{len(done)} cached, {len(jobs)} to run", flush=True)
    lock = threading.Lock()

    def one(job):
        m, dk, mo = job
        try:
            sq = runnables[(m, dk)].invoke({"query": mo})
            has = filter_has_producttype(getattr(sq, "filter", None))
            return {"model": m, "desc": dk, "motif": mo, "has_category_filter": bool(has),
                    "filter": str(getattr(sq, "filter", None))[:200]}
        except Exception as e:
            return {"model": m, "desc": dk, "motif": mo, "has_category_filter": None, "err": str(e)[:150]}

    if jobs:
        with open(path, "a") as f, ThreadPoolExecutor(max_workers=3) as ex:
            n = 0
            for fut in as_completed({ex.submit(one, j): j for j in jobs}):
                r = fut.result()
                with lock:
                    f.write(json.dumps(r) + "\n"); f.flush(); done[(r["model"], r["desc"], r["motif"])] = r
                n += 1
                if n % 40 == 0:
                    print(f"  {n}/{len(jobs)}", flush=True)

    out = {"design": "LangChain SelfQueryRetriever shipped query constructor; category-filter rate on 55 "
           "type-less motifs; neutral vs narrowing catalog description; temp 0.", "by_model": {}}
    for m in MODELS:
        out["by_model"][m] = {}
        for dk in DESCS:
            vals = [done[(m, dk, mo)]["has_category_filter"] for mo in motifs if (m, dk, mo) in done]
            vals = [1.0 if v else (0.0 if v is not None else None) for v in vals]
            out["by_model"][m][dk] = {"category_filter_rate": ci(vals), "n_error": sum(1 for v in vals if v is None)}
    json.dump(out, open(f"{RES}/framework_exposure.json", "w"), indent=2)
    def f(t): return f"{t[0]} [{t[1]}, {t[2]}]" if t and t[0] is not None else "n/a"
    print("=== LangChain shipped query constructor: spurious category-filter rate (55 type-less motifs) ===")
    for m in MODELS:
        for dk in DESCS:
            d = out["by_model"][m][dk]
            print(f"  {m:<14} {dk:<11} category-filter {f(d['category_filter_rate'])}  (err={d['n_error']})")
    print("wrote", f"{RES}/framework_exposure.json")


if __name__ == "__main__":
    main()
