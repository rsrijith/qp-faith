#!/usr/bin/env python
"""EXPLORATORY (significance): does the category mis-routing persist when the plan is produced via a
STRICT structured-output / function-calling API (JSON schema, nullable fields) rather than free-text
JSON? If yes, the failure is not a free-text-prompting quirk but a planning tendency that survives the
constrained-decoding path modern agentic RAG uses. If a strict schema with a nullable product_type
drives routing to ~0, that is instead a mitigation finding (and we would not use this as claim support).

Runs the 55 type-less WANDS motifs through OpenAI structured outputs (response_format json_schema,
strict) under a neutral and an aggressive system prompt, 2 models, and reports the product_type
population rate (spurious by construction). Seed-free (temp 0). Restartable. Reads OPENAI_API_KEY.

Run: set -a; . ../GroundLM/.env; set +a; ./.venv/bin/python scripts/structured_output_planner.py
"""
import os, json, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
DATA = os.path.join(HERE, "..", "data")
OUTDIR = os.path.join(RES, "structured")
os.makedirs(OUTDIR, exist_ok=True)
RNG = np.random.default_rng(13)
MODELS = ["gpt-4o", "gpt-4o-mini"]
SLOTS = ["product_type", "color", "material", "style", "brand", "price_range"]
PROMPTS = {
    "neutral": ("Extract structured search filters from the shopper's query. Fill a field ONLY if the "
                "query expresses that attribute; otherwise leave it null."),
    "aggressive": ("You are an e-commerce query planner. Produce a rich plan that narrows the catalog as "
                   "much as possible to what the shopper most likely wants; infer probable attributes even "
                   "if not stated, filling as many fields as you reasonably can."),
}
SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {s: {"type": ["string", "null"]} for s in SLOTS},
    "required": SLOTS,  # strict mode: all listed, but nullable so the model may set null
}


def load_env():
    envp = os.path.join(HERE, "..", "..", "GroundLM", ".env")
    if os.path.exists(envp):
        for line in open(envp):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def ci(vals):
    a = np.array([v for v in vals if v is not None], float)
    if not len(a):
        return (None, None, None)
    b = [a[RNG.integers(0, len(a), len(a))].mean() for _ in range(5000)]
    return (round(float(a.mean()), 3), round(float(np.percentile(b, 2.5)), 3), round(float(np.percentile(b, 97.5)), 3))


def main():
    load_env()
    from openai import OpenAI
    client = OpenAI()
    motifs = [json.loads(l)["query"] for l in open(f"{DATA}/probe/probe.jsonl")
              if json.loads(l).get("family") == "A_entity"]
    path = os.path.join(OUTDIR, "structured_plans.jsonl")
    done = {}
    if os.path.exists(path):
        for l in open(path):
            r = json.loads(l); done[(r["model"], r["prompt"], r["motif"])] = r
    jobs = [(m, p, mo) for m in MODELS for p in PROMPTS for mo in motifs if (m, p, mo) not in done]
    print(f"{len(MODELS)} models x {len(PROMPTS)} prompts x {len(motifs)} motifs; {len(done)} cached, {len(jobs)} to run", flush=True)
    lock = threading.Lock()

    def one(job):
        m, p, mo = job
        try:
            r = client.chat.completions.create(
                model=m, temperature=0,
                messages=[{"role": "system", "content": PROMPTS[p]},
                          {"role": "user", "content": f'Query: "{mo}"'}],
                response_format={"type": "json_schema", "json_schema":
                                 {"name": "search_plan", "strict": True, "schema": SCHEMA}})
            plan = json.loads(r.choices[0].message.content)
            pt = plan.get("product_type")
            emit = pt is not None and str(pt).strip().lower() not in ("", "none", "null", "n/a", "na")
            return {"model": m, "prompt": p, "motif": mo, "plan": plan, "emitted_pt": bool(emit)}
        except Exception as e:
            return {"model": m, "prompt": p, "motif": mo, "plan": None, "emitted_pt": None, "err": str(e)[:150]}

    if jobs:
        with open(path, "a") as f, ThreadPoolExecutor(max_workers=6) as ex:
            for fut in as_completed({ex.submit(one, j): j for j in jobs}):
                r = fut.result()
                with lock:
                    f.write(json.dumps(r) + "\n"); f.flush(); done[(r["model"], r["prompt"], r["motif"])] = r

    out = {"design": "category false-application (spurious product_type) on 55 type-less motifs via STRICT "
           "structured-output (json_schema, nullable product_type); neutral vs aggressive prompt; temp 0.",
           "by_model": {}}
    for m in MODELS:
        out["by_model"][m] = {}
        for p in PROMPTS:
            vals = [done[(m, p, mo)]["emitted_pt"] for mo in motifs if (m, p, mo) in done]
            vals = [1.0 if v else (0.0 if v is not None else None) for v in vals]
            out["by_model"][m][p] = {"false_application": ci(vals), "n_error": sum(1 for v in vals if v is None)}
    json.dump(out, open(f"{RES}/structured_output_planner.json", "w"), indent=2)
    def f(t): return f"{t[0]} [{t[1]}, {t[2]}]" if t and t[0] is not None else "n/a"
    print("=== structured-output category false-application (nullable product_type) ===")
    for m in MODELS:
        for p in PROMPTS:
            d = out["by_model"][m][p]
            print(f"  {m:<14} {p:<11} false_application {f(d['false_application'])}  (err={d['n_error']})")
    print("Compare free-text JSON (same motifs): neutral/stock ~0.02-0.04 for aligned models, expansion ~1.0.")
    print("wrote", f"{RES}/structured_output_planner.json")


if __name__ == "__main__":
    main()
