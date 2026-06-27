#!/usr/bin/env python
"""Run a FRONTIER planner (Claude API) over the probe / WANDS / ESCI query sources,
writing plans in the same format the local scorers consume. The decisive
external-validity test: do production-grade models also force a product_type onto
type-less entity queries? Determinism: temperature=0. Cost-guarded.

Usage: python scripts/run_frontier.py --model claude-haiku-4-5-20251001 \
         --tag haiku --source probe --mode expansion
Requires ANTHROPIC_API_KEY in env.
"""
import os, sys, json, time, argparse
sys.path.insert(0, os.path.dirname(__file__))
from run_pilot import PROMPTS, extract_json

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
RES = os.path.join(os.path.dirname(__file__), "..", "results")

def load_source(src, probe_file=None):
    if src == "probe":
        path = probe_file or f"{DATA}/probe/probe.jsonl"
        rows = [json.loads(l) for l in open(path)]
        return rows, "probe"
    if src == "wands":
        import pandas as pd
        q = pd.read_csv(f"{DATA}/query.csv", sep="\t")
        return [{"query_id": int(r.query_id), "query": str(r["query"])} for _, r in q.iterrows()], "wands"
    if src == "esci":
        return [json.loads(l) for l in open(f"{DATA}/esci/queries.jsonl")], "esci"
    raise ValueError(src)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--tag", required=True)
    ap.add_argument("--source", choices=["probe", "wands", "esci"], default="probe")
    ap.add_argument("--mode", choices=list(PROMPTS), default="expansion")
    ap.add_argument("--provider", choices=["anthropic", "gemini", "together", "groq", "cerebras", "mistral", "openai"], default="anthropic")
    ap.add_argument("--max", type=int, default=10000)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--probe-file", dest="probe_file", default=None)
    a = ap.parse_args()
    rows, kind = load_source(a.source, a.probe_file)
    rows = rows[:a.max]
    tmpl = PROMPTS[a.mode]

    if a.provider == "anthropic":
        import anthropic
        if not os.environ.get("ANTHROPIC_API_KEY"): sys.exit("ANTHROPIC_API_KEY not set")
        client = anthropic.Anthropic()
        def _do(prompt):
            m = client.messages.create(model=a.model, max_tokens=200, temperature=0,
                                       messages=[{"role": "user", "content": prompt}])
            return ("".join(b.text for b in m.content if b.type == "text"),
                    m.usage.input_tokens, m.usage.output_tokens)
    elif a.provider == "gemini":
        import google.generativeai as genai
        if not os.environ.get("GOOGLE_API_KEY"): sys.exit("GOOGLE_API_KEY not set")
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        gm = genai.GenerativeModel(a.model)
        def _do(prompt):
            r = gm.generate_content(prompt, generation_config={"temperature": 0, "max_output_tokens": 2048})
            try: txt = r.text
            except Exception: txt = ""
            u = getattr(r, "usage_metadata", None)
            return (txt, getattr(u, "prompt_token_count", 0), getattr(u, "candidates_token_count", 0))
    else:  # OpenAI-compatible endpoints: together / groq / cerebras / mistral
        from openai import OpenAI
        CFG = {"together": ("TOGETHER_API_KEY", "https://api.together.xyz/v1"),
               "groq": ("GROQ_API_KEY", "https://api.groq.com/openai/v1"),
               "cerebras": ("CEREBRAS_API_KEY", "https://api.cerebras.ai/v1"),
               "mistral": ("MISTRAL_API_KEY", "https://api.mistral.ai/v1"),
               "openai": ("OPENAI_API_KEY", "https://api.openai.com/v1")}
        keyenv, base = CFG[a.provider]
        if not os.environ.get(keyenv): sys.exit(f"{keyenv} not set")
        client = OpenAI(api_key=os.environ[keyenv], base_url=base)
        import re as _re
        reasoning = bool(_re.match(r"(gpt-5|o3|o4)", a.model))  # reasoning models: max_completion_tokens, no temperature
        def _do(prompt):
            kw = {"model": a.model, "messages": [{"role": "user", "content": prompt}]}
            if reasoning:
                kw["max_completion_tokens"] = 4096   # room for hidden reasoning before the JSON
            else:
                kw["max_tokens"] = 1024; kw["temperature"] = 0
            r = client.chat.completions.create(**kw)
            u = r.usage
            return (r.choices[0].message.content or "",
                    getattr(u, "prompt_tokens", 0), getattr(u, "completion_tokens", 0))

    def call(prompt):  # exponential backoff for free-tier rate limits / transient errors
        for i in range(7):
            try:
                return _do(prompt)
            except Exception as e:
                if i == 6: raise
                time.sleep(min(2 ** i, 60))
    prefix = {"probe": f"probe-{a.tag}", "wands": a.tag, "esci": f"esci-{a.tag}"}[a.source]
    suffix = "__probe_plans.jsonl" if a.source == "probe" else "__plans.jsonl"
    out = open(f"{RES}/{prefix}{suffix}", "w", buffering=1)
    t0 = time.time(); ntok_in = ntok_out = 0
    def work(i):
        r = rows[i]
        try:
            raw, ti, to = call(tmpl.format(q=r["query"]))
        except Exception as e:  # one bad call must not crash the whole run
            sys.stderr.write(f"  [call-fail q={r.get('query')!r}: {str(e)[:60]}]\n")
            raw, ti, to = "", 0, 0
        if a.source == "probe":
            rec = {**r, "mode": a.mode, "raw": raw, "plan": extract_json(raw)}
        else:
            rec = {"query_id": r.get("query_id", i), "query": r["query"], "mode": a.mode,
                   "raw": raw, "plan": extract_json(raw)}
        return rec, ti, to
    if a.concurrency > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=a.concurrency) as ex:
            for i, (rec, ti, to) in enumerate(ex.map(work, range(len(rows)))):  # ordered
                ntok_in += ti; ntok_out += to; out.write(json.dumps(rec) + "\n")
                if (i + 1) % 25 == 0:
                    print(f"  {prefix}: {i+1}/{len(rows)} ({time.time()-t0:.0f}s) [c={a.concurrency}]")
    else:
        for i in range(len(rows)):
            rec, ti, to = work(i); ntok_in += ti; ntok_out += to
            out.write(json.dumps(rec) + "\n")
            if (i + 1) % 25 == 0:
                print(f"  {prefix}: {i+1}/{len(rows)} ({time.time()-t0:.0f}s)")
    out.close()
    # rough cost estimate (Haiku/Sonnet tiers vary; report tokens, let caller price)
    man = {"model": a.model, "source": a.source, "mode": a.mode, "n": len(rows),
           "input_tokens": ntok_in, "output_tokens": ntok_out}
    json.dump(man, open(f"{RES}/{prefix}__frontier_manifest.json", "w"), indent=2)
    print(f"done {prefix}: n={len(rows)} in={ntok_in} out={ntok_out}")

if __name__ == "__main__":
    main()
