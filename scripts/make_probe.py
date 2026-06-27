#!/usr/bin/env python
"""
make_probe — procedurally generate a query-planner faithfulness probe set with
labels KNOWN BY CONSTRUCTION (SHARED_CONTEXT route #1: automatic labels, no human
annotation, scoop-resistant). Answers the multi-class-Exact-set critique head-on:
for a GENERATED entity query we DEFINE the ground-truth reference, so a product_type
injection is provably spurious — we are not guessing from holistic relevance grades.

Two families, both with deterministic relevant sets derived from the WANDS catalog:

  A_entity  — a bare decorative-motif noun (pineapple, owl, anchor) that appears in
              product NAMES across many product classes but is NOT itself a product
              class. Ground-truth reference: NO attribute slots (type-less). Relevant
              set = products whose name contains the motif (multi-class BY DESIGN).
              Any product_type the planner injects is spurious-and-harmful by
              construction (we know the user asked for a motif, not a category).

  B_attr    — "{value} {class_noun}" from catalog vocab (e.g. "turquoise sofa").
              Ground-truth reference = exactly {that attribute, product_type}. Any
              extra slot the planner emits is spurious by construction. Relevant set
              = products of that class carrying that facet value.

Output: data/probe/probe.jsonl (one row per query) + data/probe/manifest.json
"""
import os, re, json, collections, random
import pandas as pd
from nltk.corpus import wordnet as wn

random.seed(13)
# decorative-motif filter: dominant noun sense is an animal/plant/food (clean,
# unambiguous entities; product-type words are noun.artifact and excluded).
MOTIF_LEX = {"noun.animal", "noun.plant", "noun.food"}
def is_motif(w):
    # dominant sense (top-2 WordNet noun senses) is an animal/plant/food entity.
    ss = wn.synsets(w, pos=wn.NOUN)
    return bool(ss) and ss[0].lexname() in MOTIF_LEX
DATA = os.path.join(os.path.dirname(__file__), "..", "data")
OUT = os.path.join(DATA, "probe")
os.makedirs(OUT, exist_ok=True)
STOP = set("a an the of for with and or in on to set sets piece pieces that have by your our new large small".split())

def norm(s): return re.sub(r"[^a-z0-9 ]", " ", str(s).lower()).strip()
def words(s): return [w for w in norm(s).split() if w and w not in STOP and len(w) >= 4 and w.isalpha()]

def parse_features(s):
    d = collections.defaultdict(set)
    if not isinstance(s, str): return d
    for part in s.split("|"):
        if ":" not in part: continue
        k, v = part.split(":", 1)
        k = k.strip().lower(); v = norm(v)
        if k and v: d[k].add(v)
    return d

def main():
    p = pd.read_csv(f"{DATA}/product.csv", sep="\t")
    p["nname"] = p["product_name"].map(norm)
    p["nclass"] = p["product_class"].map(norm)

    # vocab: class head tokens, and facet values (so motifs aren't actually attrs/types)
    class_tokens = set()
    for c in p["nclass"].dropna():
        class_tokens.update(c.split())
    color_vocab, mat_vocab, style_vocab = set(), set(), set()
    for s in p["product_features"]:
        d = parse_features(s)
        for v in d.get("color", set()): color_vocab.update(v.split())
        for v in d.get("primarymaterial", set()) | d.get("material", set()): mat_vocab.update(v.split())
        for v in d.get("dsprimaryproductstyle", set()) | d.get("style", set()): style_vocab.update(v.split())
    attr_vocab = color_vocab | mat_vocab | style_vocab

    # ---- Family A: cross-category motif tokens mined from product NAMES ----
    tok_classes = collections.defaultdict(set)   # token -> distinct product classes
    tok_count = collections.Counter()
    for nname, nclass in zip(p["nname"], p["nclass"]):
        seen = set(w for w in words(nname))
        for w in seen:
            tok_classes[w].add(nclass); tok_count[w] += 1
    motifs = []
    for w, classes in tok_classes.items():
        if w in class_tokens or w in attr_vocab: continue          # not a type/attr word
        if len(classes) >= 3 and tok_count[w] >= 5 and is_motif(w):  # cross-class animal/plant/food
            motifs.append((w, len(classes), tok_count[w]))
    motifs.sort(key=lambda x: (-x[1], -x[2]))
    motifs = motifs[:200]                                          # cap

    pid = p["product_id"].tolist()
    name_arr = p["nname"].tolist()
    class_arr = p["nclass"].tolist()

    rows = []
    for w, ncls, ntot in motifs:
        rel = [pid[i] for i in range(len(pid)) if re.search(rf"\b{re.escape(w)}\b", name_arr[i])]
        rel_classes = set(class_arr[i] for i in range(len(pid)) if pid[i] in set(rel))
        if len(rel) < 8: continue
        rows.append(dict(query=w, family="A_entity", known_slots={},
                         relevant_pids=rel, n_relevant=len(rel),
                         n_distinct_class=len(rel_classes)))

    # ---- Family B: attributed "{value} {class_noun}" with known reference ----
    # pick common single-word colors/materials and common single-word class nouns
    def common_vals(vocab, facet, k=12):
        cnt = collections.Counter()
        for s in p["product_features"]:
            d = parse_features(s)
            for v in d.get(facet, ()):
                if " " not in v and v in vocab: cnt[v] += 1
        return [v for v, _ in cnt.most_common(k)]
    colors = common_vals(color_vocab, "color")
    mats = common_vals(mat_vocab, "primarymaterial")
    # single-word class nouns
    cls_cnt = collections.Counter(c for c in p["nclass"] if c and c != "nan" and " " not in c)
    class_nouns = [c for c, n in cls_cnt.most_common(20) if n >= 20]

    def rel_for(classnoun, facet, value):
        out = []
        for i in range(len(pid)):
            if class_arr[i] != classnoun: continue
            d = parse_features(p["product_features"].iloc[i])
            vals = d.get(facet, set())
            if any(value in v.split() for v in vals): out.append(pid[i])
        return out

    B = []
    for cn in class_nouns[:10]:
        for col in colors[:4]:
            rel = rel_for(cn, "color", col)
            if len(rel) >= 8:
                B.append(dict(query=f"{col} {cn}", family="B_attr",
                              known_slots={"color": col, "product_type": cn},
                              relevant_pids=rel, n_relevant=len(rel),
                              n_distinct_class=1))
        for mat in mats[:3]:
            rel = rel_for(cn, "primarymaterial", mat)
            if len(rel) >= 8:
                B.append(dict(query=f"{mat} {cn}", family="B_attr",
                              known_slots={"material": mat, "product_type": cn},
                              relevant_pids=rel, n_relevant=len(rel), n_distinct_class=1))
    random.shuffle(B); B = B[:120]
    rows.extend(B)

    with open(f"{OUT}/probe.jsonl", "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    man = dict(n_total=len(rows),
               n_A_entity=sum(1 for r in rows if r["family"] == "A_entity"),
               n_B_attr=sum(1 for r in rows if r["family"] == "B_attr"),
               sample_A=[r["query"] for r in rows if r["family"] == "A_entity"][:25],
               sample_B=[r["query"] for r in rows if r["family"] == "B_attr"][:15],
               seed=13)
    json.dump(man, open(f"{OUT}/manifest.json", "w"), indent=2)
    print(json.dumps(man, indent=2))

if __name__ == "__main__":
    main()
