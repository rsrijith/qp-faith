#!/usr/bin/env python
"""R1.8 (direct): within-model reasoning-DEPTH sensitivity on the reasoning planner gpt-5.2.

The cross-model read (organic_ladder.py) compares gpt-5.2 to non-reasoning GPT-4o. This is the
sharper test the reviewer's "reasoning-depth sensitivity check" asks for: hold the model and prompt
fixed and vary the reasoning effort (low / medium / high). Question: does more reasoning reduce
spurious category routing on type-less motifs, or is the failure reasoning-invariant?

We run the 55 WANDS type-less entity motifs under two prompts that leave room to move for gpt-5.2
(default_extract, where its probe routing is ~0.5; and expansion, the aggressive stressor) at each
effort level, and report the category false-application rate (spurious product_type on a type-less
motif) with a query-clustered bootstrap. Restartable: caches per (prompt, effort, motif) to
results/r1_8/effort_plans.jsonl. Reads OPENAI_API_KEY from ../GroundLM/.env. Paid (gpt-5.2 reasoning).

Run: set -a; . ../GroundLM/.env; set +a; ./.venv/bin/python scripts/reasoning_effort_sweep.py
"""
import os, re, json, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
DATA = os.path.join(HERE, "..", "data")
OUTDIR = os.path.join(RES, "r1_8")
os.makedirs(OUTDIR, exist_ok=True)
RNG = np.random.default_rng(13)
MODEL = "gpt-5.2"
EFFORTS = ["low", "medium", "high"]
ALLOWED = "product_type, color, material, style, brand, price_range"
PROMPTS = {
    "default_extract": ("Extract structured search filters from the query as a compact JSON object. "
                        f"Allowed keys: {ALLOWED}. Values are short strings. Return ONLY the JSON object."),
    "expansion": ("You are the query-planning component of an e-commerce product search engine. A shopper "
                  "typed a short query. Produce a RICH structured plan that narrows the catalog as much as "
                  "possible to the products this shopper most likely wants. Infer the most probable attributes "
                  "even when the shopper did not state them, so the results are tightly filtered. Fill in as "
                  f"many slots as you reasonably can. Allowed keys: {ALLOWED}. Return ONLY a compact JSON object."),
}


def load_env():
    envp = os.path.join(HERE, "..", "..", "GroundLM", ".env")
    if os.path.exists(envp):
        for line in open(envp):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def extract_json(txt):
    i = txt.find("{")
    if i < 0:
        return {}
    depth = 0
    for j in range(i, len(txt)):
        if txt[j] == "{":
            depth += 1
        elif txt[j] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(txt[i:j + 1])
                except Exception:
                    return {}
    return {}


def emitted_pt(plan):
    v = (plan or {}).get("product_type") if isinstance(plan, dict) else None
    return v is not None and str(v).strip().lower() not in ("", "none", "null", "n/a", "na", "nan")


def main():
    load_env()
    if not os.environ.get("OPENAI_API_KEY"):
        print("FATAL: OPENAI_API_KEY not set"); raise SystemExit(2)
    from openai import OpenAI
    client = OpenAI()

    motifs = [json.loads(l)["query"] for l in open(f"{DATA}/probe/probe.jsonl")
              if json.loads(l).get("family") == "A_entity"]
    path = os.path.join(OUTDIR, "effort_plans.jsonl")
    done = {}
    if os.path.exists(path):
        for l in open(path):
            try:
                r = json.loads(l); done[(r["prompt"], r["effort"], r["motif"])] = r
            except Exception:
                pass
    jobs = [(p, e, m) for p in PROMPTS for e in EFFORTS for m in motifs
            if (p, e, m) not in done]
    print(f"{len(PROMPTS)} prompts x {len(EFFORTS)} efforts x {len(motifs)} motifs = "
          f"{len(PROMPTS)*len(EFFORTS)*len(motifs)} cells; {len(done)} cached, {len(jobs)} to run", flush=True)

    lock = threading.Lock()

    def run_one(job):
        p, e, m = job
        try:
            resp = client.chat.completions.create(
                model=MODEL, reasoning_effort=e, max_completion_tokens=4096,
                messages=[{"role": "system", "content": PROMPTS[p]},
                          {"role": "user", "content": f'Query: "{m}"\nJSON:'}])
            txt = resp.choices[0].message.content or ""
            plan = extract_json(txt)
            return {"prompt": p, "effort": e, "motif": m, "plan": plan,
                    "emitted_pt": bool(emitted_pt(plan)), "raw": txt[:200]}
        except Exception as ex:
            return {"prompt": p, "effort": e, "motif": m, "plan": None,
                    "emitted_pt": None, "error": str(ex)[:200]}

    if jobs:
        with open(path, "a") as f, ThreadPoolExecutor(max_workers=4) as ex:
            n = 0
            for fut in as_completed({ex.submit(run_one, j): j for j in jobs}):
                r = fut.result()
                with lock:
                    f.write(json.dumps(r) + "\n"); f.flush()
                    done[(r["prompt"], r["effort"], r["motif"])] = r
                n += 1
                if n % 30 == 0:
                    print(f"  ran {n}/{len(jobs)}", flush=True)

    def ci(vals):
        a = np.array([v for v in vals if v is not None], float)
        if not len(a):
            return (None, None, None)
        b = [a[RNG.integers(0, len(a), len(a))].mean() for _ in range(5000)]
        return (round(float(a.mean()), 3), round(float(np.percentile(b, 2.5)), 3), round(float(np.percentile(b, 97.5)), 3))

    out = {"model": MODEL, "design": "category false-application on 55 type-less WANDS motifs, "
           "gpt-5.2 at reasoning_effort low/medium/high, two prompts; query-clustered bootstrap; seed 13.",
           "by_prompt": {}}
    for p in PROMPTS:
        out["by_prompt"][p] = {}
        for e in EFFORTS:
            vals = [done[(p, e, m)]["emitted_pt"] for m in motifs if (p, e, m) in done]
            vals = [1.0 if v else (0.0 if v is not None else None) for v in vals]
            nfail = sum(1 for v in vals if v is None)
            out["by_prompt"][p][e] = {"false_application": ci(vals), "n": len(vals), "n_error": nfail}
    json.dump(out, open(os.path.join(RES, "reasoning_effort_sweep.json"), "w"), indent=2)

    def f(t): return f"{t[0]} [{t[1]}, {t[2]}]" if t and t[0] is not None else "n/a"
    print("\n=== gpt-5.2 category false-application by reasoning effort (55 type-less motifs) ===")
    for p in PROMPTS:
        print(f"  {p}:")
        for e in EFFORTS:
            d = out["by_prompt"][p][e]
            print(f"    effort={e:<7} false_application {f(d['false_application'])}  (n={d['n']}, err={d['n_error']})")
    print("wrote", os.path.join(RES, "reasoning_effort_sweep.json"))


if __name__ == "__main__":
    main()
