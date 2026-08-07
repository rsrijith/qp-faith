#!/usr/bin/env python
"""Run QP-Faith's 55 type-less WANDS motif queries through LangChain's actual
SelfQueryRetriever query-constructor (the deployed LLM->structured-metadata-filter
pattern) and measure (a) the rate at which it emits a hard product_class/category
filter and (b) the recall@100 loss when that filter is applied as a hard pre-filter.

Two conditions, matching the paper's prompt ladder inside a real framework:
  default    -- LangChain's stock query-constructor context (conservative-like).
  aggressive -- the same, with a precision-narrowing directive added to the
                document description (a common deployment tweak).

This converts the paper's RAG-metadata-filter exposure from a conjecture into a
measured result inside a widely-deployed framework. We do not modify LangChain's
constructor logic; only the document description differs across conditions."""
import os, re, json, argparse
import numpy as np, pandas as pd
from rank_bm25 import BM25Okapi

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "data"); RES = os.path.join(HERE, "..", "results")

def load_key():
    for line in open(os.path.join(HERE, "..", "..", "GroundLM", ".env")):
        m = re.match(r"\s*(?:export\s+)?ANTHROPIC_API_KEY\s*=\s*(.+)\s*$", line)
        if m: os.environ["ANTHROPIC_API_KEY"] = m.group(1).strip().strip('"').strip("'")

STOP = set("a an the of for with and or in on to set sets piece pieces by your our".split())
def norm(s): return re.sub(r"[^a-z0-9 ]", " ", str(s).lower()).strip()
def depl(w): return w[:-1] if len(w) > 3 and w.endswith("s") else w
def tokset(s): return {depl(w) for w in norm(s).split() if w and w not in STOP}
def recall_at(order, rel, k=100):
    rs = set(rel); return len([p for p in order[:k] if p in rs]) / len(rs) if rs else float("nan")
def boot(x, n=5000, seed=13):
    x = np.array([v for v in x if v == v], float)
    if not len(x): return (float("nan"),)*3
    rng = np.random.default_rng(seed); s = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))

DEFAULT_DOC = "Furniture and home goods products sold in an online store."
# aggressive analog of the paper's expansion prompt, expressed inside LangChain's own
# query-constructor: instruct the constructor to always populate product_class.
AGGRESSIVE_DOC = ("Furniture and home goods. IMPORTANT: to narrow the catalog as much as possible, "
                  "you MUST always infer and populate product_class with your single best-guess "
                  "category for EVERY query, including thematic, decorative, or one-word motif "
                  "queries. Never leave product_class unfiltered.")

def _attrs():
    from langchain_classic.chains.query_constructor.schema import AttributeInfo
    return [
        AttributeInfo(name="product_class", description="the product category/type, e.g. 'sofa', 'rug', 'wall art', 'bedding', 'toy'", type="string"),
        AttributeInfo(name="color", description="the product color", type="string"),
        AttributeInfo(name="material", description="the product material", type="string"),
    ]

def build_constructor(condition):
    """Both conditions use LangChain's real query-constructor; they differ only in the
    document description (the deployment-controlled context), matching the paper's
    conservative-vs-expansion prompt ladder inside the framework."""
    from langchain_classic.chains.query_constructor.base import (
        load_query_constructor_runnable, get_query_constructor_prompt, StructuredQueryOutputParser)
    from langchain_anthropic import ChatAnthropic
    llm = ChatAnthropic(model=os.environ.get("LC_MODEL", "claude-haiku-4-5-20251001"), temperature=0)
    if condition == "default":
        return load_query_constructor_runnable(llm, DEFAULT_DOC, _attrs())
    prompt = get_query_constructor_prompt(AGGRESSIVE_DOC, _attrs())
    return prompt | llm | StructuredQueryOutputParser.from_components()

def emitted_class(sq):
    """return the product_class value if the structured query filters on it, else None."""
    f = sq.filter
    if f is None: return None
    def walk(node):
        if node is None: return None
        if hasattr(node, "attribute") and node.attribute == "product_class":
            return str(node.value)
        for arg in getattr(node, "arguments", []) or []:
            r = walk(arg)
            if r: return r
        return None
    return walk(f)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--condition", choices=["default", "aggressive"], required=True)
    a = ap.parse_args()
    load_key()
    p = pd.read_csv(f"{DATA}/product.csv", sep="\t")
    p["nname"] = p["product_name"].map(norm); p["nclass"] = p["product_class"].map(norm)
    pids = p["product_id"].tolist(); names = p["nname"].tolist(); classes = p["nclass"].tolist()
    cls_of = {pids[i]: classes[i] for i in range(len(pids))}
    bm = BM25Okapi([n.split() for n in names]); pid_arr = np.array(pids)
    probe = [json.loads(l) for l in open(f"{DATA}/probe/probe.jsonl") if json.loads(l)["family"] == "A_entity"]

    qc = build_constructor(a.condition)
    rows, drops, emitted = [], [], []
    for r in probe:
        q = r["query"]; rel = r["relevant_pids"]
        try: sq = qc.invoke({"query": q})
        except Exception as e: print(f"  invoke failed {q}: {str(e)[:80]}"); continue
        cls = emitted_class(sq)
        emitted.append(1.0 if cls else 0.0)
        scores = bm.get_scores(list(tokset(q)) or ["x"]); order = pid_arr[np.argsort(-scores)].tolist()
        r_no = recall_at(order, rel)
        if cls:
            ctok = tokset(cls)
            keep = [pid for pid in order if ctok & tokset(cls_of.get(pid, ""))] or order
            drop = r_no - recall_at(keep, rel)
        else:
            drop = 0.0
        drops.append(drop)
        rows.append({"query": q, "emitted_product_class": cls, "recall_drop": drop})
    rep = {"condition": a.condition, "model": os.environ.get("LC_MODEL", "claude-haiku-4-5-20251001"),
           "n": len(rows), "category_emit_rate": boot(emitted), "recall_drop_all": boot(drops),
           "recall_drop_when_emitted": boot([d for d, e in zip(drops, emitted) if e]),
           "rows": rows}
    json.dump(rep, open(f"{RES}/langchain_selfquery_{a.condition}.json", "w"), indent=2)
    er, rd = rep["category_emit_rate"], rep["recall_drop_all"]
    print(f"\n=== LangChain self-query [{a.condition}] n={len(rows)} ===")
    print(f"  category-filter emit rate: {er[0]:.2f} [{er[1]:.2f},{er[2]:.2f}]")
    print(f"  recall@100 drop (all):     {rd[0]:.2f} [{rd[1]:.2f},{rd[2]:.2f}]")
    we = rep["recall_drop_when_emitted"]
    print(f"  recall@100 drop (when emitted): {we[0]:.2f} [{we[1]:.2f},{we[2]:.2f}]")

if __name__ == "__main__":
    main()
