#!/usr/bin/env python
"""Restartable multi-provider planner runner for the R1 revision (R1.2 ecological
prompts, R1.5 mitigation arms, and any future condition sweep).

Design goals the user asked for:
  * CHECKPOINTED: every completed (model, query) result is appended to disk and flushed
    immediately, so nothing is lost if the process dies.
  * RESTARTABLE: on restart it reads the existing output and SKIPS finished queries, so
    re-running the exact same command resumes where it stopped.
  * GRACEFUL ON CREDIT EXHAUSTION: an insufficient-credit / quota-exceeded error is caught,
    the file is left consistent, a resume command is printed, and the process exits 0 (not
    a crash, not an infinite retry loop). Transient rate limits get bounded exponential
    backoff first.
  * CHEAP: Anthropic calls send the constant instruction as a cache_control block, so
    calls 2..N read the prefix at ~0.1x (verify via usage.cache_read_input_tokens, printed).

Output: results/cond-{condition}-{tag}__{source}_plans.jsonl (one JSON line per query:
{query, query_id?, family?, relevant_pids?, condition, provider, model, raw, plan}).

Providers: local (MLX), anthropic, gemini, openai. Keys come from the environment
(ANTHROPIC_API_KEY / GOOGLE_API_KEY / OPENAI_API_KEY); never printed.

Examples:
  # local (free) first, to validate a condition end-to-end at $0:
  ./.venv/bin/python scripts/run_conditions.py --condition stock_selfquery --source probe \
      --model local:mlx-community/Qwen2.5-7B-Instruct-4bit:qwen4
  # then a hosted arm, resumable:
  ./.venv/bin/python scripts/run_conditions.py --condition stock_selfquery --source probe \
      --model anthropic:claude-haiku-4-5:haiku
"""
import os, sys, json, time, argparse, re, threading
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
DATA = os.path.join(HERE, "..", "data")
RES = os.path.join(HERE, "..", "results")
from run_pilot import extract_json  # reuse the paper's JSON extractor

ALLOWED = "product_type, color, material, style, brand, price_range"

# Each condition = a constant SYSTEM instruction (cached) + a USER template with {q}.
# Provenance is documented in AUDIT_TRAIL_R1.md. These are neutral, task-focused prompts
# written to a realistic e-commerce query-planning spec and NOT phrased around the paper's
# hypothesis (R1 comment 2 explicitly accepts "prompts designed independently by
# practitioners who are unaware of the hypothesis"). `expansion` is the aggressive stressor
# already in the paper, kept here as the susceptibility reference point.
CONDITIONS = {
    # stock self-query-retriever style: extract only what the query states.
    "stock_selfquery": {
        "system": ("You translate a user's search query into structured metadata filters for a "
                   "document store. Extract only filters that are explicitly present in the query. "
                   f"Allowed keys: {ALLOWED}. If the query does not mention an attribute, do not "
                   "include it. Return ONLY a compact JSON object."),
        "user": 'Query: "{q}"\nJSON:'},
    # neutral practitioner prompt A: confident-only.
    "practitioner_shopping": {
        "system": ("You are the query-planning step of an online store's search. Turn the shopper's "
                   f"query into catalog filter fields. Allowed keys: {ALLOWED}. Include a field only "
                   "if the shopper's words point to it. Return ONLY a compact JSON object."),
        "user": 'Query: "{q}"\nJSON:'},
    # neutral practitioner prompt B: light inference to narrow, no instruction to fill many slots.
    "practitioner_narrow": {
        "system": ("Convert the search query into catalog filters that narrow the results to what the "
                   f"shopper wants. Allowed keys: {ALLOWED}. Include fields you are fairly confident "
                   "about; leave out ones you are unsure of. Return ONLY a compact JSON object."),
        "user": 'Query: "{q}"\nJSON:'},
    # plain default: extract filters, no narrowing pressure either way.
    "default_extract": {
        "system": ("Extract structured search filters from the query as a compact JSON object. "
                   f"Allowed keys: {ALLOWED}. Values are short strings. Return ONLY the JSON object."),
        "user": 'Query: "{q}"\nJSON:'},
    # the aggressive stressor from the paper, restated (susceptibility reference).
    "expansion": {
        "system": ("You are the query-planning component of an e-commerce product search engine. A "
                   "shopper typed a short query. Produce a RICH structured plan that narrows the "
                   "catalog as much as possible to the products this shopper most likely wants. Infer "
                   "the most probable attributes even when the shopper did not state them, so the "
                   f"results are tightly filtered. Fill in as many slots as you reasonably can. Allowed "
                   f"keys: {ALLOWED}. Return ONLY a compact JSON object."),
        "user": 'Query: "{q}"\nJSON:'},
    # the omit-instruction primary mitigation (R1.5), restated for hosted arms.
    "omit_instruction": {
        "system": ("You are the query-planning component of an e-commerce product search engine. "
                   "Produce a structured plan that narrows the catalog to what the shopper most likely "
                   f"wants. Allowed keys: {ALLOWED}. IMPORTANT: include product_type ONLY if the query "
                   "explicitly names a product category; if the query is a bare entity or theme, leave "
                   "product_type out. Return ONLY a compact JSON object."),
        "user": 'Query: "{q}"\nJSON:'},
    # brand/color-specific omit mitigation (R1.5 extension): keep aggressive category
    # behavior, but restrain brand and color to what the query actually names.
    "omit_bc": {
        "system": ("You are the query-planning component of an e-commerce product search engine. "
                   "Produce a structured plan that narrows the catalog to what the shopper most likely "
                   f"wants. Allowed keys: {ALLOWED}. IMPORTANT: include brand ONLY if the query names a "
                   "specific brand, and include color ONLY if the query names a specific color; if the "
                   "query does not state a brand or a color, leave that field out entirely (do not guess, "
                   "and do not write placeholders like 'any' or 'unknown'). Return ONLY a compact JSON object."),
        "user": 'Query: "{q}"\nJSON:'},
}


def load_source(src):
    if src == "probe":
        rows = [json.loads(l) for l in open(f"{DATA}/probe/probe.jsonl")]
        return [{"key": r["query"], **r} for r in rows]
    if src == "probe1162":
        rows = [json.loads(l) for l in open(f"{DATA}/probe/probe_1162.jsonl")]
        return [{"key": r["query"], **r} for r in rows]
    if src == "probe_hard":
        rows = [json.loads(l) for l in open(f"{DATA}/probe/probe_hard.jsonl")]
        return [{"key": r["query"], **r} for r in rows]
    if src == "wands":
        import pandas as pd
        q = pd.read_csv(f"{DATA}/query.csv", sep="\t")
        return [{"key": int(r.query_id), "query_id": int(r.query_id), "query": str(r["query"])}
                for _, r in q.iterrows()]
    if src == "esci":
        rows = [json.loads(l) for l in open(f"{DATA}/esci/queries.jsonl")]
        return [{"key": int(r["query_id"]), "query_id": int(r["query_id"]), "query": str(r["query"])}
                for r in rows]
    raise ValueError(src)


def is_credit_exhausted(e):
    m = str(e).lower()
    return any(s in m for s in ["credit balance", "insufficient", "quota", "billing",
                                "exceeded your current", "resource_exhausted", "resourceexhausted",
                                "insufficient_quota"])


def is_transient(e):
    m = str(e).lower()
    return any(s in m for s in ["rate limit", "ratelimit", "429", "overloaded", "500", "502", "503",
                                "timeout", "timed out", "connection", "unavailable"])


def is_fatal(e):
    """Permanent errors where retrying is pointless: bad model id, auth, malformed request."""
    m = str(e).lower()
    return any(s in m for s in ["not found", "no longer available", "404", "does not exist",
                                "invalid model", "unauthorized", "401", "403", "permission denied",
                                "authentication", "invalid api key"])


def make_caller(provider, model, cond):
    sys_text, user_tmpl = cond["system"], cond["user"]
    if provider == "local":
        from run_pilot import load_planner
        tmpl = sys_text + "\n\n" + user_tmpl  # single prompt for MLX
        planner = load_planner(model, tmpl)
        return lambda q: (planner(q), None)
    if provider == "anthropic":
        import anthropic
        if not os.environ.get("ANTHROPIC_API_KEY"):
            sys.exit("ANTHROPIC_API_KEY not set")
        client = anthropic.Anthropic()

        def _call(q):
            m = client.messages.create(
                model=model, max_tokens=200,
                system=[{"type": "text", "text": sys_text, "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": user_tmpl.format(q=q)}])
            txt = "".join(b.text for b in m.content if b.type == "text")
            return txt, getattr(m.usage, "cache_read_input_tokens", None)
        return _call
    if provider == "gemini":
        import google.generativeai as genai
        if not os.environ.get("GOOGLE_API_KEY"):
            sys.exit("GOOGLE_API_KEY not set")
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        gm = genai.GenerativeModel(model, system_instruction=sys_text)
        return lambda q: (gm.generate_content(
            user_tmpl.format(q=q),
            generation_config={"temperature": 0, "max_output_tokens": 256}).text, None)
    if provider in ("openai", "together", "groq", "cerebras", "mistral"):
        # OpenAI-compatible endpoints (mirrors run_frontier.py). gpt-5/o3/o4 are reasoning
        # models: max_completion_tokens, no temperature (room for hidden reasoning + JSON).
        from openai import OpenAI
        CFG = {"openai": ("OPENAI_API_KEY", "https://api.openai.com/v1"),
               "together": ("TOGETHER_API_KEY", "https://api.together.xyz/v1"),
               "groq": ("GROQ_API_KEY", "https://api.groq.com/openai/v1"),
               "cerebras": ("CEREBRAS_API_KEY", "https://api.cerebras.ai/v1"),
               "mistral": ("MISTRAL_API_KEY", "https://api.mistral.ai/v1")}
        keyenv, base = CFG[provider]
        if not os.environ.get(keyenv):
            sys.exit(f"{keyenv} not set")
        client = OpenAI(api_key=os.environ[keyenv], base_url=base)
        reasoning = bool(re.match(r"(gpt-5|o3|o4)", model))

        def _call(q):
            kw = {"model": model, "messages": [{"role": "system", "content": sys_text},
                                               {"role": "user", "content": user_tmpl.format(q=q)}]}
            if reasoning:
                kw["max_completion_tokens"] = 4096
            else:
                kw["max_tokens"] = 200
                kw["temperature"] = 0
            r = client.chat.completions.create(**kw)
            return (r.choices[0].message.content or ""), None
        return _call
    raise ValueError(provider)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True, choices=list(CONDITIONS))
    ap.add_argument("--source", required=True, choices=["probe", "probe1162", "probe_hard", "wands", "esci"])
    ap.add_argument("--model", required=True,
                    help="provider:model_id:tag  e.g. anthropic:claude-haiku-4-5:haiku")
    ap.add_argument("--max", type=int, default=100000)
    ap.add_argument("--max-retries", type=int, default=5)
    ap.add_argument("--concurrency", type=int, default=0,
                    help="0 = auto (local sequential; hosted parallel per provider). Local is always "
                         "forced to 1 to protect GPU/RAM.")
    a = ap.parse_args()
    provider, model, tag = a.model.split(":", 2)
    # Local MUST be sequential (one 7B MLX model in RAM at a time). Hosted runs concurrent;
    # Gemini free tier throttles hard, so it gets a lower default.
    AUTO_CONC = {"local": 1, "anthropic": 6, "openai": 6, "gemini": 2,
                 "together": 4, "groq": 4, "cerebras": 4, "mistral": 4}
    conc = 1 if provider == "local" else (a.concurrency or AUTO_CONC.get(provider, 4))
    rows = load_source(a.source)[:a.max]

    outpath = f"{RES}/cond-{a.condition}-{tag}__{a.source}_plans.jsonl"
    done = set()
    if os.path.exists(outpath):
        for l in open(outpath):
            try:
                done.add(json.loads(l)["key"])
            except Exception:
                pass
    todo = [r for r in rows if r["key"] not in done]
    print(f"[{a.condition}/{tag}/{a.source}] {len(done)} done, {len(todo)} to do -> {os.path.basename(outpath)}")
    if not todo:
        print("nothing to do (already complete).")
        return

    caller = make_caller(provider, model, CONDITIONS[a.condition])
    lock = threading.Lock()
    out = open(outpath, "a", buffering=1)
    t0 = time.time()
    state = {"cache_seen": False, "written": 0}
    resume_cmd = (f"./.venv/bin/python scripts/run_conditions.py --condition {a.condition} "
                  f"--source {a.source} --model {a.model}")

    def build_rec(r, raw):
        rec = {k: r[k] for k in ("query", "query_id", "family", "relevant_pids", "known_slots") if k in r}
        rec.update(key=r["key"], condition=a.condition, provider=provider, model=model,
                   raw=raw, plan=extract_json(raw))
        return rec

    def process_one(r):
        """One query with per-call retry/classify. Returns (status, payload);
        status in {ok, credit, fatal, error}. Transient errors (429/throttle) back off
        inside here, so parallel workers self-throttle without a global governor."""
        for attempt in range(a.max_retries + 1):
            try:
                raw, cache_read = caller(r["query"])
                if cache_read and cache_read > 0 and not state["cache_seen"]:
                    state["cache_seen"] = True
                    print(f"  prompt cache active (cache_read_input_tokens={cache_read})")
                return ("ok", build_rec(r, raw))
            except Exception as e:
                if provider != "local" and is_credit_exhausted(e) and not is_transient(e):
                    return ("credit", e)
                if is_fatal(e):
                    return ("fatal", e)
                if attempt < a.max_retries and is_transient(e):
                    time.sleep(min(60, 2 ** attempt) + attempt * 0.3)  # throttle backoff
                    continue
                if attempt < min(2, a.max_retries):
                    time.sleep(3 + attempt * 3)
                    continue
                return ("error", e)

    def finalize_and_exit(status, e):
        out.flush(); out.close()
        n = len(done) + state["written"]
        if status == "credit":
            print(f"\n*** CREDIT/QUOTA EXHAUSTED on {provider}:{model} after {n} rows. ***")
            print(f"    {type(e).__name__}: {str(e)[:200]}")
            print(f"    Progress saved. Refill and rerun the SAME command to resume:\n    {resume_cmd}")
            sys.exit(0)
        if status == "fatal":
            print(f"\n*** FATAL (no retry) on {provider}:{model}: {type(e).__name__}: {str(e)[:160]}")
            print(f"    Fix the model id / credentials and rerun to resume.")
            sys.exit(2)
        print(f"\n*** Stopping after repeated errors ({type(e).__name__}: {str(e)[:160]}). "
              f"Progress saved; rerun to resume:\n    {resume_cmd}")
        sys.exit(1)

    def record(payload):
        with lock:
            out.write(json.dumps(payload) + "\n"); out.flush()
            state["written"] += 1
            if state["written"] % 25 == 0:
                print(f"  {tag}: {len(done)+state['written']}/{len(rows)} ({time.time()-t0:.0f}s)")

    if conc <= 1:
        for r in todo:
            status, payload = process_one(r)
            if status != "ok":
                finalize_and_exit(status, payload)
            record(payload)
    else:
        print(f"  hosted parallel: {conc} workers (self-throttling on 429)")
        with ThreadPoolExecutor(max_workers=conc) as ex:
            futs = {ex.submit(process_one, r): r for r in todo}
            for fut in as_completed(futs):
                status, payload = fut.result()
                if status != "ok":
                    for f in futs:
                        f.cancel()
                    finalize_and_exit(status, payload)  # exits; in-flight workers drain
                record(payload)
    out.close()
    print(f"done {a.condition}/{tag}/{a.source}: {len(rows)} total ({time.time()-t0:.0f}s, conc={conc}).")


if __name__ == "__main__":
    main()
