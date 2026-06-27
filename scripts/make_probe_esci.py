#!/usr/bin/env python
"""Second-domain entity probe on the Amazon/ESCI catalog. Same construction as the
WANDS probe (WordNet animal/plant/food motifs that span many products in titles;
relevant set = products whose TITLE contains the motif). Labels known by
construction: the query expresses no product_type. Confirms the headline transfers
to a different domain/catalog. Output: data/probe_esci/{probe.jsonl, catalog.parquet}."""
import os, re, json, collections
import pandas as pd
from huggingface_hub import hf_hub_download
from nltk.corpus import wordnet as wn

OUT = os.path.join(os.path.dirname(__file__), "..", "data", "probe_esci")
os.makedirs(OUT, exist_ok=True)
STOP = set("a an the the of for with and or in on to set sets by your our new large small pack of count".split())
MOTIF_LEX = {"noun.animal", "noun.plant"}  # drop noun.food (broad ESCI -> sugar/drink noise)
BLOCK = {"male", "female", "down", "blade", "blades", "bulb", "bulbs", "body", "head", "hand", "back",
         "stickers", "sticker", "mount", "charger", "simple", "queen", "human", "needle", "crop",
         "giant", "acer", "mouse", "heather"}  # ESCI domain-ambiguous (device/brand/size) terms
def is_motif(w):
    if w in BLOCK: return False
    ss = wn.synsets(w, pos=wn.NOUN); return bool(ss) and ss[0].lexname() in MOTIF_LEX
def norm(s): return re.sub(r"[^a-z0-9 ]", " ", str(s).lower()).strip()
def words(s): return [w for w in norm(s).split() if w and w not in STOP and len(w) >= 4 and w.isalpha()]

def main():
    cat = f"{OUT}/catalog.parquet"
    if os.path.exists(cat):
        df = pd.read_parquet(cat)                          # reuse cached catalog
    else:
        f = hf_hub_download("tasksource/esci",
                            "data/test-00000-of-00004-d48474212b95f33b.parquet", repo_type="dataset")
        df = pd.read_parquet(f)
        df = df[df.product_locale == "us"].drop_duplicates("product_id")[["product_id", "product_title"]].dropna()
        df = df.reset_index(drop=True)
        df["ntitle"] = df["product_title"].map(norm)
        df.to_parquet(cat)
    n = len(df); print(f"ESCI us catalog: {n} unique products")

    tok_count = collections.Counter()
    for t in df["ntitle"]:
        for w in set(words(t)): tok_count[w] += 1
    # motif: animal/plant/food noun, in 15..2000 titles (not too rare/generic)
    motifs = [w for w, c in tok_count.items() if 8 <= c <= 3000 and is_motif(w)]
    motifs = sorted(motifs, key=lambda w: -tok_count[w])[:500]

    titles = df["ntitle"].tolist(); pids = df["product_id"].tolist()
    rows = []
    for w in motifs:
        rel = [pids[i] for i in range(len(pids)) if re.search(rf"\b{re.escape(w)}\b", titles[i])]
        if len(rel) < 8: continue
        rows.append(dict(query=w, family="A_entity", known_slots={},
                         relevant_pids=rel, n_relevant=len(rel), n_distinct_class=-1))
    with open(f"{OUT}/probe.jsonl", "w") as fh:
        for r in rows: fh.write(json.dumps(r) + "\n")
    man = dict(catalog_size=n, n_entity=len(rows),
               sample=[r["query"] for r in rows][:30],
               median_relevant=int(pd.Series([r["n_relevant"] for r in rows]).median()) if rows else 0)
    json.dump(man, open(f"{OUT}/manifest.json", "w"), indent=2)
    print(json.dumps(man, indent=2))

if __name__ == "__main__":
    main()
