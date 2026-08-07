#!/usr/bin/env python
"""Pin the facet-coverage mechanism numbers to a concrete computation (final-review
integrity item): product_class vs soft-slot coverage among the entity probe's relevant
products, and the distinct-product_class span of the two motifs that also appear as
WANDS human-graded queries (dinosaur, flamingo). Writes results/facet_coverage.json."""
import json, re, collections, os
import pandas as pd
DATA = os.path.join(os.path.dirname(__file__), "..", "data")
RES = os.path.join(os.path.dirname(__file__), "..", "results")
def norm(s): return re.sub(r"[^a-z0-9 ]", " ", str(s).lower()).strip()
def parse(s):
    d = collections.defaultdict(set)
    if isinstance(s, str):
        for part in s.split("|"):
            if ":" in part:
                k, v = part.split(":", 1); d[k.strip().lower()].add(norm(v))
    return d
def has(s, keys): d = parse(s); return any(d.get(k) for k in keys)

p = pd.read_csv(f"{DATA}/product.csv", sep="\t")
probe = [json.loads(l) for l in open(f"{DATA}/probe/probe.jsonl") if json.loads(l)["family"] == "A_entity"]
rel = set().union(*[set(r["relevant_pids"]) for r in probe])
pr = p[p.product_id.isin(rel)]
cov = dict(n_relevant=int(len(pr)),
           product_class=round(float(pr.product_class.notna().mean()), 3),
           color=round(float(pr.product_features.map(lambda s: has(s, ["color"])).mean()), 3),
           material=round(float(pr.product_features.map(lambda s: has(s, ["primarymaterial", "material"])).mean()), 3),
           style=round(float(pr.product_features.map(lambda s: has(s, ["dsprimaryproductstyle", "style", "dssecondaryproductstyle"])).mean()), 3),
           catalog_product_class=round(float(p.product_class.notna().mean()), 3))

# dinosaur / flamingo distinct product_class among human-Exact, non-null vs dropna=False
lab = pd.read_csv(f"{DATA}/label.csv", sep="\t"); q = pd.read_csv(f"{DATA}/query.csv", sep="\t")
exact = lab[lab.label.astype(str).str.lower().str.startswith("exact")].merge(q, on="query_id")
pc = dict(zip(p.product_id, p.product_class))
span = {}
for motif in ["dinosaur", "flamingo"]:
    pids = exact[exact["query"].str.lower() == motif].product_id.tolist()
    cls = [pc.get(pid) for pid in pids]
    span[motif] = dict(distinct_nonnull=len({c for c in cls if pd.notna(c)}),
                       distinct_incl_blank=len(set(cls)), n_exact=len(pids))
out = dict(coverage_among_relevant=cov, motif_class_span=span)
json.dump(out, open(f"{RES}/facet_coverage.json", "w"), indent=2)
print(json.dumps(out, indent=2))
