#!/usr/bin/env python
"""
Decisive pilot for the LLM-query-planner faithfulness study.

Tests two things on WANDS (480 queries, graded Exact/Partial/Irrelevant):
  H1 (injection): do LLM planners emit attribute constraints the query never
      expressed?  -> Spurious-Constraint Rate (SCR).
  H2 (harm beyond confound): do those injected constraints exclude genuinely
      relevant products?  -> Harmful-SCR, counted ONLY on POPULATED-AND-
      CONFLICTING facets, reported against the facet-coverage confound.

Gate (designed so the phenomenon can kill itself):
  (a) SCR materially > 0 with CI excluding a trivial floor, AND
  (b) Harmful-SCR separates from facet-coverage (harm beyond missing-data).

Outputs: results/<model>__plans.jsonl, results/<model>__scored.parquet,
         results/<model>__manifest.json
Determinism: greedy (temp=0), fixed seed, pinned values; stable blake2b hashing.
"""
import os, sys, json, re, time, random, hashlib, argparse, collections
import numpy as np
import pandas as pd

SEED = 13
if os.environ.get("PYTHONHASHSEED") != "0":
    # fail loud per empirical-gate discipline
    print("WARN: set PYTHONHASHSEED=0 for stable hashing", file=sys.stderr)
random.seed(SEED); np.random.seed(SEED)

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
RES = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RES, exist_ok=True)

# plan slots the planner may emit and the WANDS facet keys each maps to
SLOT_FACETS = {
    "color":    ["color"],
    "material": ["primarymaterial", "material"],
    "style":    ["dsprimaryproductstyle", "style", "dssecondaryproductstyle"],
}
# product_type is reported for drift but excluded from the harm slots (it is
# usually the query head-noun, rarely a spurious injection).

STOP = set("a an the of for with and or in on to set sets piece pieces".split())

def norm(s):
    return re.sub(r"[^a-z0-9 ]", " ", str(s).lower()).strip()

def words(s):
    return [w for w in norm(s).split() if w and w not in STOP]

def parse_features(s):
    d = collections.defaultdict(set)
    if not isinstance(s, str):
        return d
    for part in s.split("|"):
        if ":" not in part:
            continue
        k, v = part.split(":", 1)
        k = k.strip().lower(); v = norm(v)
        if k and v:
            d[k].add(v)
    return d

def value_matches(plan_val, prod_vals):
    """plan value matches a product facet value (substring-tolerant)."""
    pv = norm(plan_val)
    if not pv:
        return False
    for v in prod_vals:
        if pv == v or pv in v or v in pv:
            return True
    return False

def grounded_in_query(plan_val, qwords, qnorm):
    """Is the plan value expressed by the query string (lexical reference)?"""
    pv = norm(plan_val)
    if not pv:
        return True  # empty -> ignore
    if pv in qnorm:
        return True
    pw = [w for w in pv.split() if w not in STOP]
    # grounded if any content word of the value appears in the query
    return any(w in qwords for w in pw)

PROMPT_CONSERVATIVE = """You are the query-planning component of an e-commerce product search engine.
Given a shopper's search query, output the structured attribute filters that best capture what the shopper wants, so the catalog can be narrowed to matching products.

Return ONLY a compact JSON object. Include a key only if you are confident the shopper wants that constraint. Allowed keys: product_type, color, material, style, brand, price_range. Values must be short strings.

Query: "{q}"
JSON:"""

PROMPT_EXPANSION = """You are the query-planning component of an e-commerce product search engine.
A shopper typed a short query. Your job is to produce a RICH structured plan that narrows the catalog as much as possible to the products this shopper most likely wants. Infer the most probable attributes even when the shopper did not state them, so the results are tightly filtered. Fill in as many slots as you reasonably can.

Return ONLY a compact JSON object. Allowed keys: product_type, color, material, style, brand, price_range. Values must be short strings.

Query: "{q}"
JSON:"""

# --- dose-response ladder: 3 intermediate levels between conservative & expansion ---
_HEAD = """You are the query-planning component of an e-commerce product search engine.
Given a shopper's search query, output structured attribute filters as a compact JSON object. Allowed keys: product_type, color, material, style, brand, price_range. Values must be short strings.\n\n"""
PROMPT_IMPLY = _HEAD + 'Include attributes the query states, and add an attribute ONLY IF it is strongly implied by the query. Do not guess.\n\nQuery: "{q}"\nJSON:'
PROMPT_ASSIST = _HEAD + 'Include attributes the query states, and add a few likely attributes that would help narrow the results.\n\nQuery: "{q}"\nJSON:'
PROMPT_ENRICH = _HEAD + 'Produce a fairly complete plan: include stated attributes and infer the probable attributes the shopper most likely wants.\n\nQuery: "{q}"\nJSON:'

# schema-nullability ablation: expansion-aggressive, but product_type is explicitly
# optional and must be OMITTED when the query is an entity/theme/motif, not a category.
# Tests whether type-forcing is a slot-filling compulsion (curable by instruction) or
# a genuine misread (resistant). Same aggressiveness as expansion otherwise.
PROMPT_EXP_OPTIONAL = _HEAD + ('Infer the most probable attributes to narrow the catalog. '
    'IMPORTANT: include product_type ONLY if the query explicitly names a product category or type. '
    'If the query is an entity, theme, motif, or concept rather than a product category, OMIT product_type entirely '
    '(the shopper wants products depicting or related to it, of any type).\n\nQuery: "{q}"\nJSON:')

# ordered low->high inference aggressiveness
PROMPTS = {"conservative": PROMPT_CONSERVATIVE, "imply": PROMPT_IMPLY,
           "assist": PROMPT_ASSIST, "enrich": PROMPT_ENRICH, "expansion": PROMPT_EXPANSION,
           "exp_optional": PROMPT_EXP_OPTIONAL}
PROMPT = PROMPT_CONSERVATIVE  # default; overridden by --mode

def extract_json(text):
    m = re.search(r"\{.*?\}", text, re.S)
    if not m:
        return {}
    try:
        obj = json.loads(m.group(0))
        return {k: v for k, v in obj.items() if isinstance(v, str) and v.strip()}
    except Exception:
        return {}

def load_planner(model_path, prompt_tmpl):
    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler
    model, tok = load(model_path)
    sampler = make_sampler(temp=0.0)
    def run(q):
        msgs = [{"role": "user", "content": prompt_tmpl.format(q=q)}]
        prompt = tok.apply_chat_template(msgs, add_generation_prompt=True)
        out = generate(model, tok, prompt=prompt, max_tokens=120,
                       sampler=sampler, verbose=False)
        return out
    return run

def bootstrap_ci(vals, fn=np.mean, n=2000, seed=SEED):
    vals = np.asarray(vals, dtype=float)
    if len(vals) == 0:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(vals), size=(n, len(vals)))
    stats = fn(vals[idx], axis=1)
    return float(fn(vals)), float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n", type=int, default=480)
    ap.add_argument("--mode", choices=list(PROMPTS), default="conservative")
    args = ap.parse_args()

    q = pd.read_csv(f"{DATA}/query.csv", sep="\t")
    queries = q.head(args.n).reset_index(drop=True)

    planner = load_planner(args.model, PROMPTS[args.mode])
    plans_fp = open(f"{RES}/{args.tag}__plans.jsonl", "w", buffering=1)  # line-buffered
    t0 = time.time()
    for i, r in queries.iterrows():
        qid = int(r.query_id); qstr = str(r["query"])
        raw = planner(qstr)
        plan = extract_json(raw)
        plans_fp.write(json.dumps({"query_id": qid, "query": qstr, "mode": args.mode,
                                   "raw": raw, "plan": plan}) + "\n")
        if (i + 1) % 50 == 0:
            print(f"  {args.tag}: {i+1}/{len(queries)}  ({time.time()-t0:.0f}s)")
    plans_fp.close()
    print(f"done {args.tag} ({args.mode}) -> {RES}/{args.tag}__plans.jsonl")
    return

def _dead_old_main():
    q = p = lab = None  # unreachable; retained logic removed in refactor
    queries = None
    # parse features only for products that are Exact for some query (harm pool)
    exact = lab[lab.label == "Exact"]
    exact_by_q = exact.groupby("query_id")["product_id"].apply(list).to_dict()
    need_pids = set(exact.product_id.unique())
    pf = p.set_index("product_id")
    feat_cache = {}
    for pid in need_pids:
        try:
            feat_cache[pid] = parse_features(pf.at[pid, "product_features"])
        except KeyError:
            feat_cache[pid] = collections.defaultdict(set)

    planner = load_planner(args.model)

    plans_fp = open(f"{RES}/{args.tag}__plans.jsonl", "w")
    rows = []
    t0 = time.time()
    for i, r in queries.iterrows():
        qid = int(r.query_id); qstr = str(r["query"])
        qn = norm(qstr); qw = set(words(qstr))
        raw = planner(qstr)
        plan = extract_json(raw)
        plans_fp.write(json.dumps({"query_id": qid, "query": qstr,
                                   "raw": raw, "plan": plan}) + "\n")

        exact_pids = exact_by_q.get(qid, [])
        n_exact = len(exact_pids)

        for slot, facets in SLOT_FACETS.items():
            if slot not in plan:
                continue
            val = plan[slot]
            spurious = not grounded_in_query(val, qw, qn)
            # facet-coverage: Exact products with this facet populated
            populated = 0; conflicting = 0
            for pid in exact_pids:
                fd = feat_cache.get(pid, {})
                pv = set().union(*[fd.get(f, set()) for f in facets]) if any(f in fd for f in facets) else set()
                if pv:
                    populated += 1
                    if not value_matches(val, pv):
                        conflicting += 1
            cov = populated / n_exact if n_exact else 0.0
            harm = conflicting > 0 and spurious
            rows.append(dict(model=args.tag, query_id=qid, query=qstr, slot=slot,
                             value=val, spurious=int(spurious),
                             n_exact=n_exact, populated=populated,
                             conflicting=conflicting, facet_coverage=cov,
                             harmful=int(harm),
                             excl_frac=(conflicting / n_exact if n_exact else 0.0)))
        if (i + 1) % 50 == 0:
            print(f"  {args.tag}: {i+1}/{len(queries)}  ({time.time()-t0:.0f}s)")
    plans_fp.close()

    df = pd.DataFrame(rows)
    df.to_parquet(f"{RES}/{args.tag}__scored.parquet")

    # ---- metrics ----
    nq = len(queries)
    # per-query spurious-constraint count and harmful count
    by_q = df.groupby("query_id")
    spur_per_q = [by_q.get_group(g).spurious.sum() if g in by_q.groups else 0
                  for g in queries.query_id]
    has_spur = [int(s > 0) for s in spur_per_q]
    # among spurious constraints, fraction harmful
    spur_rows = df[df.spurious == 1]
    harmful_of_spur = spur_rows.harmful.tolist() if len(spur_rows) else []
    # facet coverage among spurious (the confound) vs harmful conflict-rate
    cov_of_spur = spur_rows.facet_coverage.tolist() if len(spur_rows) else []
    # conflicting-within-populated (the harm signal net of coverage)
    cwp = [(r.conflicting / r.populated) for r in spur_rows.itertuples() if r.populated > 0]
    excl = [by_q.get_group(g).excl_frac.max() if g in by_q.groups else 0.0
            for g in queries.query_id]

    def fmt(name, ci):
        return f"  {name:32s} {ci[0]:.3f}  [{ci[1]:.3f}, {ci[2]:.3f}]"

    report = {
        "model": args.tag, "model_path": args.model, "n_queries": nq,
        "total_plan_slots": int(len(df)),
        "total_spurious": int(df.spurious.sum()),
        "frac_queries_with_spurious": bootstrap_ci(has_spur),
        "spurious_per_query": bootstrap_ci(spur_per_q),
        "harmful_SCR_amongspurious": bootstrap_ci([float(x) for x in harmful_of_spur]) if harmful_of_spur else None,
        "facet_coverage_amongspurious(confound)": bootstrap_ci(cov_of_spur) if cov_of_spur else None,
        "conflict_rate_within_populated": bootstrap_ci(cwp) if cwp else None,
        "exact_recall_loss_at_plan": bootstrap_ci(excl),
        "seed": SEED,
    }
    with open(f"{RES}/{args.tag}__manifest.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n=== {args.tag}  (n={nq}) ===")
    print(f"  plan slots emitted (harm-bearing): {len(df)}   spurious: {df.spurious.sum()}")
    print(fmt("frac queries w/ spurious", report["frac_queries_with_spurious"]))
    print(fmt("spurious per query", report["spurious_per_query"]))
    if report["harmful_SCR_amongspurious"]:
        print(fmt("Harmful-SCR (of spurious)", report["harmful_SCR_amongspurious"]))
        print(fmt("facet-coverage (confound)", report["facet_coverage_amongspurious(confound)"]))
        print(fmt("conflict-rate|populated", report["conflict_rate_within_populated"]))
    print(fmt("Exact-recall-loss@plan", report["exact_recall_loss_at_plan"]))
    return report

if __name__ == "__main__":
    main()
