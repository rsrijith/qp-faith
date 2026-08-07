#!/usr/bin/env python
"""Faithfulness-check experiment (blind-panel-2 STEP 2 hardened).

Claim: content-faithfulness graders that check the emitted VALUE against the
SOURCE (the retrieved catalog listings) are structurally blind to mis-routing,
because a mis-routed category is a real category with real matching products, so
it is faithfully grounded. A QUERY-ENTAILMENT check (premise = the search query)
separates mis-routing from legitimate typing.

Graders, all off-the-shelf, local, no API, deterministic (seed=13),
query-clustered bootstrap CIs over DISTINCT queries (not model-query pairs):

  (C) source-grounded content grader (RAGAS-groundedness style): premise = a
      retrieved product listing of the emitted class; hypothesis "This product is
      a {t}." Run on BOTH mis-routed and legitimate types. If it passes BOTH it
      does not separate -> blind. (The representative grader the panel asked for,
      promoted over the WordNet lexical-existence check as headline evidence.)
  (V) value-validity (WordNet head-noun existence): the kind of check that
      catches a FABRICATED value. Passes mis-routed REAL categories; rejects a
      nonsense control. Retained; secondary.
  (Q) query-entailment (two NLI checkpoints): premise = the search query,
      hypothesis "The shopper wants a {t}." Flags mis-routing; spares legit types.

Robustness of (Q) against a "separation is just lexical overlap" objection:
  (L) lexical baseline: is the type head-noun a token of the query? mis-routed
      ~0, explicit legit-B ~1 -> lexical alone separates on EXPLICIT controls.
  Implicit-type controls: a small curated set of legitimate queries whose correct
  type is NOT lexically present (hyponym->hypernym, e.g. "sectional"->sofa). If
  (Q) spares these while (L) does not, (Q) is doing more than lexical matching.

Run: ./.venv/bin/python scripts/faithfulness_check.py
"""
import os, json, collections, re
import numpy as np
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, "..", "results")
DATA = os.path.join(HERE, "..", "data")
MODELS = ["qwen4","llama4","mistral4","gemma4","haiku","sonnet","together-llama70","gpt4o","gpt52"]
RNG = np.random.default_rng(13)

def head_noun(t): return str(t).strip().lower().split()[-1] if str(t).strip() else ""

# --- catalog (source documents for the content grader) ---------------------
prod = pd.read_csv(os.path.join(DATA, "product.csv"), sep="\t")
prod["cl"] = prod["product_class"].astype(str).str.lower()
def listing_text(row):
    parts = [str(row.get("product_name", "")), str(row.get("product_class", "")),
             str(row.get("product_description", ""))[:300]]
    return ". ".join(p for p in parts if p and p != "nan")
by_class = collections.defaultdict(list); by_pid = {}
for _, r in prod.iterrows():
    by_class[head_noun(r["cl"])].append(listing_text(r))
    by_pid[r["product_id"]] = listing_text(r)
def retrieve(t, k=5):
    """the listings a hard product_class~t filter surfaces (RAGAS 'context')."""
    h = head_noun(t)
    hits = by_class.get(h, [])
    if not hits:
        mask = prod["cl"].str.contains(re.escape(h), regex=True, na=False)
        hits = [listing_text(r) for _, r in prod[mask].iterrows()]
    return hits[:k]
def relevant_listings(pids, k=5):
    """listings of the motif's TRUE relevant set (what SHOULD be retrieved)."""
    out = [by_pid[p] for p in pids if p in by_pid]
    return out[:k]

# --- collect injections grouped by DISTINCT query (the cluster unit) --------
mis = collections.defaultdict(list); leg = collections.defaultdict(list); mis_relpids = {}
for m in MODELS:
    p = f"{RES}/probe-{m}-exp__probe_plans.jsonl"
    if not os.path.exists(p): continue
    for l in open(p):
        r = json.loads(l); plan = r.get("plan"); fam = r.get("family")
        if not isinstance(plan, dict) or "product_type" not in plan: continue
        pt = str(plan["product_type"]).strip()
        if not pt: continue
        if fam == "A_entity":
            mis[r["query"]].append(pt)
            if r["query"] not in mis_relpids:
                rp = r.get("relevant_pids")
                try: mis_relpids[r["query"]] = json.loads(rp) if isinstance(rp, str) else (rp or [])
                except Exception: mis_relpids[r["query"]] = []
        elif fam == "B_attr":
            ks = r.get("known_slots")
            try: d = json.loads(ks) if isinstance(ks, str) else (ks or {})
            except Exception: d = {}
            intended = str(d.get("product_type", pt)).strip()  # the TRUE type for a legit query
            leg[r["query"]].append(intended)
n_mis_pairs = sum(len(v) for v in mis.values()); n_leg_pairs = sum(len(v) for v in leg.values())
print(f"mis-routed: {n_mis_pairs} injections / {len(mis)} motifs | Family-B: {n_leg_pairs} / {len(leg)} queries", flush=True)

# implicit-type legitimate controls: correct type entailed, NOT lexically present
IMPLICIT = [("sectional","sofa"),("recliner","chair"),("duvet","bedding"),
            ("runner","rug"),("sconce","lamp"),("credenza","cabinet"),
            ("hutch","cabinet"),("bunk","bed"),("armoire","wardrobe"),
            ("vanity","table"),("chaise","chair"),("ottoman","stool")]
IMPLICIT = [(q,t) for q,t in IMPLICIT if head_noun(t) not in q.lower().split()]
imp = {q:[t] for q,t in IMPLICIT}

NONSENSE = ["florbex","quzzint","zylophant","grunther","wibbleton","snorquil","plemtar","vundrisk"]

# --- WordNet value-validity -------------------------------------------------
import nltk
try: from nltk.corpus import wordnet as wn; wn.synsets("chair")
except Exception:
    nltk.download("wordnet"); nltk.download("omw-1.4"); from nltk.corpus import wordnet as wn
real = lambda t: len(wn.synsets(head_noun(t), pos=wn.NOUN)) > 0

def cluster_ci(per_query_fracs, B=5000):
    a = np.array(per_query_fracs, dtype=float); n = len(a)
    if n == 0: return (0.0, 0.0, 0.0)
    boots = [a[RNG.integers(0, n, n)].mean() for _ in range(B)]
    return round(a.mean(), 3), round(float(np.percentile(boots, 2.5)), 3), round(float(np.percentile(boots, 97.5)), 3)

def per_query_rate(groups, predicate):
    return [np.mean([predicate(q, t) for t in types]) for q, types in groups.items()]

def per_query_rate_nan(groups, predicate):
    """nan-tolerant: average over a query's non-missing types; drop all-missing queries."""
    out = []
    for q, types in groups.items():
        vals = [predicate(q, t) for t in types]
        vals = [v for v in vals if v == v]  # keep non-nan
        if vals: out.append(float(np.mean(vals)))
    return out

out = {"n_misrouted_pairs": n_mis_pairs, "n_misrouted_motifs": len(mis),
       "n_familyB_pairs": n_leg_pairs, "n_familyB_queries": len(leg),
       "n_implicit_controls": len(imp), "implicit_controls": IMPLICIT,
       "effective_n_note": "rates are per-distinct-query means; CIs bootstrap the query clusters"}

# (V) value-validity
out["V_value_validity_misrouted"] = cluster_ci(per_query_rate(mis, lambda q, t: real(t)))
fab = {q: [NONSENSE[i % len(NONSENSE)]] for i, q in enumerate(mis)}
out["V_value_validity_fabricated"] = cluster_ci(per_query_rate(fab, lambda q, t: real(t)))

# (L) lexical-overlap baseline
lex = lambda q, t: head_noun(t) in re.findall(r"[a-z]+", q.lower())
out["L_lexical_misrouted"] = cluster_ci(per_query_rate(mis, lex))
out["L_lexical_legitB"]    = cluster_ci(per_query_rate(leg, lex))
out["L_lexical_implicit"]  = cluster_ci(per_query_rate(imp, lex))

# NLI graders: (C) source-grounded content, (Q) query-entailment
from sentence_transformers import CrossEncoder
GRADERS = ["cross-encoder/nli-deberta-v3-base", "cross-encoder/nli-roberta-base"]
for name in GRADERS:
    nli = CrossEncoder(name)
    ent_idx = [i for i, l in nli.model.config.id2label.items() if str(l).lower().startswith("entail")][0]
    def entail(prem, hyp):
        s = nli.predict([(prem, hyp)], apply_softmax=True, convert_to_numpy=True)[0]
        return int(s.argmax()) == ent_idx
    _cache = {}
    def grounded_pass(q, t):
        docs = _cache.get(t)
        if docs is None: docs = _cache[t] = retrieve(t)
        if not docs: return np.nan
        hyp = f"This product is a {t}."
        return 1.0 if np.mean([entail(d, hyp) for d in docs]) >= 0.5 else 0.0  # majority-grounded
    _rcache = {}
    def grounded_relevantset_pass(q, t):
        """same value check, but grounded against the motif's TRUE relevant set."""
        docs = _rcache.get(q)
        if docs is None: docs = _rcache[q] = relevant_listings(mis_relpids.get(q, []))
        if not docs: return np.nan
        hyp = f"This product is a {t}."
        return 1.0 if np.mean([entail(d, hyp) for d in docs]) >= 0.5 else 0.0
    def q_entailed(q, t):
        return entail(f'A shopper searched an online store for "{q}".', f"The shopper wants a {t}.")
    key = name.split("/")[-1]
    cg = per_query_rate_nan(mis, grounded_pass)
    out[f"C_grounded_misrouted__{key}"] = cluster_ci(cg)
    out[f"n_content_grader_motifs"] = len(cg)  # motifs with >=1 retrievable emitted class (<= 55)
    out[f"C_grounded_legitB__{key}"]    = cluster_ci(per_query_rate_nan(leg, grounded_pass))
    out[f"C_grounded_relevantset_misrouted__{key}"] = cluster_ci(per_query_rate_nan(mis, grounded_relevantset_pass))
    out[f"Q_pass_misrouted__{key}"] = cluster_ci(per_query_rate(mis, q_entailed))
    out[f"Q_pass_legitB__{key}"]    = cluster_ci(per_query_rate(leg, q_entailed))
    out[f"Q_pass_implicit__{key}"]  = cluster_ci(per_query_rate(imp, q_entailed))

json.dump(out, open(f"{RES}/faithfulness_check.json", "w"), indent=2)
def f(k): m,lo,hi = out[k]; return f"{m} [{lo}, {hi}]"
print("\n=== faithfulness-check (query-clustered CIs; two NLI checkpoints) ===")
print(f"(V) WordNet value-validity   mis-routed {f('V_value_validity_misrouted')} ; fabricated {f('V_value_validity_fabricated')}")
print(f"(L) lexical overlap          mis-routed {f('L_lexical_misrouted')} ; legit-B {f('L_lexical_legitB')} ; implicit {f('L_lexical_implicit')}")
for name in GRADERS:
    k = name.split("/")[-1]
    print(f"[{k}]")
    print(f"  (C) grounded vs FILTERED results  mis-routed {f('C_grounded_misrouted__'+k)} ; legit-B {f('C_grounded_legitB__'+k)}   (both high = BLIND; n_motifs={out['n_content_grader_motifs']})")
    print(f"  (C') grounded vs TRUE relevant set mis-routed {f('C_grounded_relevantset_misrouted__'+k)}   (low = CATCHES it -> blindness is about which docs you ground against)")
    print(f"  (Q) query-entailment         mis-routed pass {f('Q_pass_misrouted__'+k)} ; legit-B {f('Q_pass_legitB__'+k)} ; implicit {f('Q_pass_implicit__'+k)}")
print("wrote", f"{RES}/faithfulness_check.json")
