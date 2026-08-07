#!/usr/bin/env python
"""R1.6 follow-up: enlarge the implicit-entailment control set and re-grade.

The reviewer notes the query-conditioned check's "performance on 12 implicit controls is weak, so
the abstract statement that conditioning on the query 'catches' the error is too categorical."
We enlarge the implicit-type control set from 12 to a principled, reproducible set derived from
WordNet hyponym->hypernym pairs (the same form as the paper's seeds: a query term whose correct
product TYPE is a hypernym NOT lexically present, e.g. sectional->sofa, sconce->lamp), plus the
paper's original 12. Each pair is a LEGITIMATE implicit-type query: the shopper genuinely wants
the hypernym type, but the type word is not in the query.

We then grade every pair with the paper's query-entailment check (Q) using the two NLI checkpoints
(cross-encoder/nli-deberta-v3-base, cross-encoder/nli-roberta-base; entailment via softmax on the
entail label) and the lexical baseline (L). The claim under test: (Q) should SPARE these legitimate
implicit types (high entailment) where (L) cannot, and it should do so on a set far larger than 12.
We report the entailment pass rate of (Q) and (L) with a bootstrap CI, so "query entailment catches
it" is calibrated rather than categorical.

$0, local (WordNet + local cross-encoders; the checkpoints download once). Seed 13.
Run: ./.venv/bin/python scripts/implicit_entailment_set.py
"""
import os, re, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
RNG = np.random.default_rng(13)
GRADERS = ["cross-encoder/nli-deberta-v3-base", "cross-encoder/nli-roberta-base"]

# product-type hypernyms whose WordNet hyponyms are plausible catalog subtypes
HYPERNYMS = ["sofa", "chair", "table", "lamp", "bed", "cabinet", "rug", "desk", "stool",
             "dresser", "shelf", "clock", "mirror", "couch", "wardrobe", "bench", "bookcase"]
# the paper's original 12 implicit controls (kept for continuity)
SEED = [("sectional", "sofa"), ("recliner", "chair"), ("duvet", "bedding"), ("runner", "rug"),
        ("sconce", "lamp"), ("credenza", "cabinet"), ("hutch", "cabinet"), ("bunk", "bed"),
        ("armoire", "wardrobe"), ("vanity", "table"), ("chaise", "chair"), ("ottoman", "stool")]


def norm(s):
    return re.sub(r"[^a-z0-9 ]", " ", str(s).lower()).strip()
def head(s):
    t = norm(s).split(); return t[-1] if t else ""
def toks(s):
    return set(norm(s).split())


def build_pairs():
    import nltk
    try:
        from nltk.corpus import wordnet as wn; wn.synsets("chair")
    except Exception:
        nltk.download("wordnet"); nltk.download("omw-1.4"); from nltk.corpus import wordnet as wn
    pairs = {}
    for hyper in HYPERNYMS:
        for syn in wn.synsets(hyper, pos=wn.NOUN):
            for hypo in syn.hyponyms():
                for lemma in hypo.lemma_names():
                    term = lemma.replace("_", " ").lower()
                    # single- or two-word subtype whose hypernym head is NOT a token (truly implicit)
                    if 1 <= len(term.split()) <= 2 and head(hyper) not in toks(term) and term.isascii():
                        pairs[term] = hyper
    # add seeds (override), drop lexical-present
    for q, t in SEED:
        pairs[q] = t
    pairs = {q: t for q, t in pairs.items() if head(t) not in toks(q)}
    return sorted(pairs.items())


def load_env():
    envp = os.path.join(HERE, "..", "..", "GroundLM", ".env")
    if os.path.exists(envp):
        for line in open(envp):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def validate_genuine(pairs):
    """WordNet hyponymy is noisy (berm->shelf, altar->table, board->table are not product
    implicit-types). Keep only pairs an LLM confirms as a genuine product subtype whose shopper
    would want the hypernym. Cached to results/implicit_valid.jsonl. gpt-4o-mini, cheap."""
    load_env()
    cache = os.path.join(RES, "implicit_valid.jsonl")
    done = {}
    if os.path.exists(cache):
        for l in open(cache):
            r = json.loads(l); done[(r["q"], r["t"])] = r["genuine"]
    from openai import OpenAI
    client = OpenAI()
    sys = ("You judge whether a search term is a genuine, purchasable PRODUCT subtype of a category, "
           "such that a real shopper typing the term would want products of that category. Answer STRICT "
           'JSON {"genuine": true|false}. genuine=true only if the term names a concrete product/furniture '
           "type a store would sell whose natural category is the given one (e.g. sectional->sofa true, "
           "recliner->chair true). false for archaic, metaphorical, geological, or non-retail senses "
           "(e.g. berm->shelf false, altar->table false, board->table false).")
    out = []
    with open(cache, "a") as f:
        for q, t in pairs:
            if (q, t) in done:
                if done[(q, t)]:
                    out.append((q, t))
                continue
            try:
                r = client.chat.completions.create(model="gpt-4o-mini", temperature=0, max_tokens=20,
                    messages=[{"role": "system", "content": sys},
                              {"role": "user", "content": f'term "{q}", category "{t}". JSON:'}])
                txt = r.choices[0].message.content
                g = bool(json.loads(txt[txt.find("{"):txt.rfind("}") + 1]).get("genuine"))
            except Exception:
                g = False
            f.write(json.dumps({"q": q, "t": t, "genuine": g}) + "\n"); f.flush()
            if g:
                out.append((q, t))
    return out


def main():
    from sentence_transformers import CrossEncoder
    candidates = build_pairs()
    pairs = validate_genuine(candidates)
    print(f"implicit-type pairs: {len(candidates)} WordNet candidates -> {len(pairs)} LLM-confirmed "
          f"genuine (was 12); grading with {len(GRADERS)} NLI checkpoints", flush=True)

    # (L) lexical baseline: is the type head-noun a token of the query? (should be ~0 by construction)
    lex = [1.0 if head(t) in toks(q) else 0.0 for q, t in pairs]

    # (Q) query-entailment per checkpoint
    per_ckpt = {}
    for name in GRADERS:
        nli = CrossEncoder(name)
        ent_idx = [i for i, l in nli.model.config.id2label.items() if str(l).lower().startswith("entail")][0]
        passes = []
        for q, t in pairs:
            prem = f'A shopper searched an online store for "{q}".'
            hyp = f"The shopper wants a {t}."
            s = nli.predict([(prem, hyp)], apply_softmax=True, convert_to_numpy=True)[0]
            passes.append(1.0 if int(np.argmax(s)) == ent_idx else 0.0)
        per_ckpt[name] = passes

    def ci(v):
        a = np.array(v, float)
        b = [a[RNG.integers(0, len(a), len(a))].mean() for _ in range(5000)]
        return (round(float(a.mean()), 3), round(float(np.percentile(b, 2.5)), 3), round(float(np.percentile(b, 97.5)), 3))

    # (Q) passes if EITHER checkpoint entails (the paper aggregates the two); also report each
    both = [max(per_ckpt[GRADERS[0]][i], per_ckpt[GRADERS[1]][i]) for i in range(len(pairs))]
    out = {"n_pairs": len(pairs), "n_wordnet_candidates": len(candidates), "prior_n": 12,
           "design": "enlarged implicit-type controls: WordNet hyponym->hypernym candidates + paper's 12, "
                     "LLM-confirmed to genuine purchasable product subtypes (removes WordNet noise like "
                     "berm->shelf). Query-entailment (Q) should spare these legitimate implicit types; "
                     "lexical (L) cannot.",
           "lexical_baseline_pass": ci(lex),
           "query_entailment_pass_either": ci(both)}
    for name in GRADERS:
        out[f"query_entailment_pass[{name}]"] = ci(per_ckpt[name])
    # example pairs where Q still fails (for honest discussion)
    fails = [f"{q}->{t}" for i, (q, t) in enumerate(pairs) if both[i] == 0.0][:20]
    out["examples_Q_fails"] = fails
    json.dump(out, open(f"{RES}/implicit_entailment.json", "w"), indent=2)

    def f(x): return f"{x[0]} [{x[1]}, {x[2]}]"
    print(f"\n=== enlarged implicit-entailment controls (n={len(pairs)}) ===")
    print(f"  lexical baseline (L) pass:            {f(out['lexical_baseline_pass'])}  (expect ~0 by construction)")
    print(f"  query-entailment (Q) pass, either:    {f(out['query_entailment_pass_either'])}")
    for name in GRADERS:
        print(f"    {name:<38} {f(out[f'query_entailment_pass[{name}]'])}")
    print(f"  Q still fails on (sample): {fails[:10]}")
    print("wrote", f"{RES}/implicit_entailment.json")


if __name__ == "__main__":
    main()
