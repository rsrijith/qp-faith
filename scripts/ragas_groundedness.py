#!/usr/bin/env python
"""R1.6: evaluate an ACTUAL RAGAS faithfulness/groundedness implementation on the mis-routing
cases, with balanced positive and negative controls, naming every model and threshold.

The submitted paper's grader (C) is a "RAGAS-groundedness STYLE" local-NLI approximation
(scripts/faithfulness_check.py). R1.6 asks for the real thing. We run the ragas library's
Faithfulness metric (statement decomposition + LLM verdict against the retrieved context) on
three balanced arms built exactly like the paper's dissociation:

  - MIS-ROUTED (test): query is type-less; the planner emitted a spurious product_type t;
    context = the listings a hard product_class~t filter surfaces (retrieve(t)); the answer
    asserts "the retrieved products are t products." Because the retrieval was filtered to t,
    a source-grounded grader should score this HIGH -> BLIND to the mis-routing.
  - LEGITIMATE (positive control): Family-B query with a genuine type t; same construction;
    a working grader should also score HIGH.
  - FABRICATED (negative control): the answer asserts a DIFFERENT (nonsense) type against the
    SAME real listings; a working grader should score LOW. This proves the grader is not
    trivially passing everything, so a high mis-routed score is genuine blindness.

If MIS-ROUTED ~ LEGITIMATE (both high) while FABRICATED is low, the standard groundedness
grader cannot separate mis-routing from legitimate typing -> the paper's claim, now on a real
RAGAS run rather than a custom NLI check.

Judge model: gpt-4o-mini, temperature 0 (named + logged). RAGAS Faithfulness default statement
threshold. Query-clustered bootstrap over distinct queries, seed 13. Restartable: per-instance
scores cached to results/ragas/ragas_scores.jsonl; re-runs skip cached instances and only spend
on new ones. Reads OPENAI_API_KEY from ../GroundLM/.env.

Run: ./.venv/bin/python scripts/ragas_groundedness.py [--judge gpt-4o-mini] [--limit-misrouted N]
"""
import os, re, json, argparse, collections
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
DATA = os.path.join(HERE, "..", "data")
RAGDIR = os.path.join(RES, "ragas")
os.makedirs(RAGDIR, exist_ok=True)
RNG = np.random.default_rng(13)
MODELS = ["qwen4", "llama4", "mistral4", "gemma4", "haiku", "sonnet", "together-llama70", "gpt4o", "gpt52"]
NONSENSE = ["florbex", "quzzint", "zylophant", "grunther", "wibbleton", "snorquil", "plemtar", "vundrisk"]


def load_env():
    envp = os.path.join(HERE, "..", "..", "GroundLM", ".env")
    if os.path.exists(envp):
        for line in open(envp):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def head_noun(t):
    return str(t).strip().lower().split()[-1] if str(t).strip() else ""


# --- catalog + retrieval (identical logic to faithfulness_check.py) ---
prod = pd.read_csv(os.path.join(DATA, "product.csv"), sep="\t")
prod["cl"] = prod["product_class"].astype(str).str.lower()
def listing_text(row):
    parts = [str(row.get("product_name", "")), str(row.get("product_class", "")),
             str(row.get("product_description", ""))[:300]]
    return ". ".join(p for p in parts if p and p != "nan")
by_class = collections.defaultdict(list)
for _, r in prod.iterrows():
    by_class[head_noun(r["cl"])].append(listing_text(r))
def retrieve(t, k=5):
    h = head_noun(t)
    hits = by_class.get(h, [])
    if not hits:
        mask = prod["cl"].str.contains(re.escape(h), regex=True, na=False)
        hits = [listing_text(rr) for _, rr in prod[mask].iterrows()]
    return hits[:k]


def build_instances(limit_misrouted=None):
    """Return list of dicts: {id, arm, query, type, contexts, answer}."""
    mis, leg = collections.defaultdict(list), collections.defaultdict(list)
    for m in MODELS:
        p = f"{RES}/probe-{m}-exp__probe_plans.jsonl"
        if not os.path.exists(p):
            continue
        for l in open(p):
            r = json.loads(l); plan = r.get("plan"); fam = r.get("family")
            if not isinstance(plan, dict) or "product_type" not in plan:
                continue
            pt = str(plan["product_type"]).strip()
            if not pt:
                continue
            if fam == "A_entity":
                mis[r["query"]].append(pt)
            elif fam == "B_attr":
                ks = r.get("known_slots")
                try:
                    d = json.loads(ks) if isinstance(ks, str) else (ks or {})
                except Exception:
                    d = {}
                leg[r["query"]].append(str(d.get("product_type", pt)).strip())

    inst = []
    def ans(query, t):
        # Phrasing matters for RAGAS statement decomposition. This template makes the single
        # verifiable claim "these products belong to category t", checked against the retrieved
        # listings. Validated to discriminate: legitimate typing -> ~1.0, fabricated type -> 0.0
        # (see the answer-template calibration in AUDIT_TRAIL_R1). Query text is intentionally
        # NOT in the claim, so the grader tests category-groundedness, not query recall.
        return f"All of the retrieved products belong to the {t} category."

    # MIS-ROUTED: one instance per (query, distinct emitted type)
    mkeys = sorted(mis)
    if limit_misrouted:
        mkeys = mkeys[:limit_misrouted]
    for q in mkeys:
        for t in sorted(set(mis[q])):
            ctx = retrieve(t)
            if not ctx:
                continue
            inst.append({"id": f"mis::{q}::{t}", "arm": "misrouted", "query": q,
                         "type": t, "contexts": ctx, "answer": ans(q, t)})
    # LEGITIMATE positive control (Family-B attributed queries)
    for q in sorted(leg):
        for t in sorted(set(leg[q])):
            ctx = retrieve(t)
            if not ctx:
                continue
            inst.append({"id": f"leg::{q}::{t}", "arm": "legitimate", "query": q,
                         "type": t, "contexts": ctx, "answer": ans(q, t)})
    # ENLARGED legitimate control (real typed WANDS queries; lifts n=18 -> ~88 so the positive
    # control is not the weak link). Sampled deterministically (seed 13) from auto_reference typed.
    _rng = np.random.default_rng(13)
    ref = [json.loads(l) for l in open(f"{DATA}/gold/auto_reference.jsonl")]
    typed = [(int(r["query_id"]), r["query"], (r.get("explicit") or {}).get("product_type"))
             for r in ref if (r.get("explicit") or {}).get("product_type")]
    for i in sorted(_rng.choice(len(typed), size=min(70, len(typed)), replace=False).tolist()):
        qid, q, tfull = typed[i]
        t = head_noun(tfull)  # clean category head-noun (retrieve() indexes by head-noun anyway)
        if not t:
            continue
        ctx = retrieve(t)
        if not ctx:
            continue
        inst.append({"id": f"legW::{qid}::{t}", "arm": "legitimate", "query": q,
                     "type": t, "contexts": ctx, "answer": ans(q, t)})
    # FABRICATED negative control: real spurious-type listings, but answer asserts a nonsense type
    for i, q in enumerate(mkeys):
        ts = sorted(set(mis[q]))
        if not ts:
            continue
        t = ts[0]
        ctx = retrieve(t)
        if not ctx:
            continue
        nz = NONSENSE[i % len(NONSENSE)]
        inst.append({"id": f"fab::{q}::{nz}", "arm": "fabricated", "query": q,
                     "type": nz, "contexts": ctx, "answer": ans(q, nz)})
    return inst


def cluster_ci(per_query, B=5000):
    a = np.array(per_query, float); n = len(a)
    if n == 0:
        return (None, None, None)
    b = [a[RNG.integers(0, n, n)].mean() for _ in range(B)]
    return (round(float(a.mean()), 3), round(float(np.percentile(b, 2.5)), 3),
            round(float(np.percentile(b, 97.5)), 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", default="gpt-4o-mini")
    ap.add_argument("--limit-misrouted", type=int, default=None)
    ap.add_argument("--batch", type=int, default=40)
    args = ap.parse_args()
    load_env()
    if not os.environ.get("OPENAI_API_KEY"):
        print("FATAL: OPENAI_API_KEY not found in env or ../GroundLM/.env"); raise SystemExit(2)

    inst = build_instances(args.limit_misrouted)
    jslug = re.sub(r"[^a-z0-9.-]", "_", args.judge.lower())
    scores_path = os.path.join(RAGDIR, f"ragas_scores_{jslug}.jsonl")
    done = {}
    if os.path.exists(scores_path):
        for l in open(scores_path):
            try:
                r = json.loads(l); done[r["id"]] = r["faithfulness"]
            except Exception:
                pass
    todo = [x for x in inst if x["id"] not in done]
    print(f"instances: {len(inst)} total ({sum(1 for x in inst if x['arm']=='misrouted')} mis, "
          f"{sum(1 for x in inst if x['arm']=='legitimate')} leg, "
          f"{sum(1 for x in inst if x['arm']=='fabricated')} fab); {len(done)} cached, {len(todo)} to score",
          flush=True)

    if todo:
        from ragas import evaluate, EvaluationDataset
        from ragas.dataset_schema import SingleTurnSample
        from ragas.metrics import Faithfulness
        from ragas.llms import LangchainLLMWrapper
        if args.judge.lower().startswith("claude"):
            from langchain_anthropic import ChatAnthropic
            llm = LangchainLLMWrapper(ChatAnthropic(model=args.judge, temperature=0, max_tokens=1024))
        else:
            from langchain_openai import ChatOpenAI
            llm = LangchainLLMWrapper(ChatOpenAI(model=args.judge, temperature=0))
        faith = Faithfulness(llm=llm)
        with open(scores_path, "a") as f:
            for i in range(0, len(todo), args.batch):
                chunk = todo[i:i + args.batch]
                samples = [SingleTurnSample(user_input=x["query"], response=x["answer"],
                                            retrieved_contexts=x["contexts"]) for x in chunk]
                ds = EvaluationDataset(samples=samples)
                res = evaluate(ds, metrics=[faith], llm=llm, show_progress=False)
                df = res.to_pandas()
                col = "faithfulness" if "faithfulness" in df.columns else df.columns[-1]
                for x, val in zip(chunk, df[col].tolist()):
                    v = float(val) if val == val else None  # NaN-safe
                    f.write(json.dumps({"id": x["id"], "arm": x["arm"], "query": x["query"],
                                        "type": x["type"], "faithfulness": v}) + "\n")
                    f.flush(); done[x["id"]] = v
                print(f"  scored {min(i+args.batch, len(todo))}/{len(todo)}", flush=True)

    # aggregate per arm, query-clustered
    per_arm = {"misrouted": collections.defaultdict(list), "legitimate": collections.defaultdict(list),
               "fabricated": collections.defaultdict(list)}
    for x in inst:
        v = done.get(x["id"])
        if v is not None:
            per_arm[x["arm"]][x["query"]].append(v)
    out = {"judge_model": args.judge, "metric": "ragas.Faithfulness (statement decomposition + "
           "LLM verdict vs retrieved context); ragas 0.2.15; temperature 0; default threshold; seed 13",
           "n_instances": len(inst),
           "pass_definition": "faithfulness score in [0,1]; per-query mean then query-clustered bootstrap"}
    for arm in per_arm:
        pq = [float(np.mean(v)) for v in per_arm[arm].values() if v]
        out[arm] = {"n_queries": len(pq), "n_instances": sum(len(v) for v in per_arm[arm].values()),
                    "mean_faithfulness": cluster_ci(pq)}
    out_path = os.path.join(RES, "ragas_groundedness.json" if args.judge == "gpt-4o-mini"
                            else f"ragas_groundedness_{jslug}.json")
    json.dump(out, open(out_path, "w"), indent=2)

    def f(t): return f"{t[0]} [{t[1]}, {t[2]}]" if t and t[0] is not None else "n/a"
    print("\n=== RAGAS Faithfulness (judge %s) ===" % args.judge)
    for arm in ["misrouted", "legitimate", "fabricated"]:
        d = out[arm]
        print(f"  {arm:<12} n_q={d['n_queries']:<4} mean_faithfulness {f(d['mean_faithfulness'])}")
    print("\nReading: mis-routed ~ legitimate (both high) with fabricated low => the standard "
          "groundedness grader is BLIND to mis-routing (passes it like legitimate typing).")
    print("wrote", out_path)


if __name__ == "__main__":
    main()
