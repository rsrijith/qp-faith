#!/usr/bin/env python
"""Build a confirmatory second/third-domain catalog + motif probe from a McAuley
Amazon-Reviews-2023 product-metadata category, streamed line-by-line (no full
download). Mirrors build_esci_full.py's motif logic EXACTLY so the only thing that
changes across domains is the catalog, not the probe-construction method.

Each domain -> data/probe_domain/<dom>/{catalog.parquet, probe.jsonl, manifest.json}.

Usage: python build_domain_probe.py --category meta_Electronics --dom electronics --cap 400000
"""
import os, re, json, argparse, collections
import pandas as pd
from huggingface_hub import HfFileSystem
from nltk.corpus import wordnet as wn

OUT_ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "probe_domain")
# identical motif lexicon + block list to build_esci_full.py (parity)
MOTIF_LEX = {"noun.animal", "noun.plant"}
BLOCK = {"male","female","down","blade","blades","bulb","bulbs","body","head","hand","back",
         "stickers","sticker","mount","charger","simple","queen","human","needle","crop","giant",
         "acer","mouse","heather"}
def is_motif(w):
    if w in BLOCK: return False
    ss = wn.synsets(w, pos=wn.NOUN); return bool(ss) and ss[0].lexname() in MOTIF_LEX
STOP = set("a an the of for with and or in on to set sets by your our new large small pack count".split())
def norm(s): return re.sub(r"[^a-z0-9 ]", " ", str(s).lower()).strip()
def words(s): return [w for w in norm(s).split() if w and w not in STOP and len(w) >= 4 and w.isalpha()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", required=True, help="e.g. meta_Electronics")
    ap.add_argument("--dom", required=True, help="short domain id, e.g. electronics")
    ap.add_argument("--cap", type=int, default=400000, help="max titles to stream")
    ap.add_argument("--min-tok", type=int, default=10, help="min title-occurrences for a motif")
    ap.add_argument("--min-rel", type=int, default=8, help="min relevant products per motif")
    ap.add_argument("--motifs-from", default=None,
                    help="path to an existing probe.jsonl; restrict to its (validated) "
                         "A_entity motif strings instead of auto-mining this domain. "
                         "Low-degrees-of-freedom design: identical decorative-motif queries "
                         "measured across catalogs, varying only the domain.")
    a = ap.parse_args()
    outdir = os.path.join(OUT_ROOT, a.dom); os.makedirs(outdir, exist_ok=True)
    catf = os.path.join(outdir, "catalog.parquet")

    if os.path.exists(catf):
        df = pd.read_parquet(catf); print(f"reuse {a.dom} catalog: {len(df)}")
    else:
        fs = HfFileSystem()
        path = f"datasets/McAuley-Lab/Amazon-Reviews-2023/raw/meta_categories/{a.category}.jsonl"
        ids, titles, seen = [], [], set()
        n = 0
        with fs.open(path, "r") as fh:
            for line in fh:
                n += 1
                try: r = json.loads(line)
                except Exception: continue
                pid = r.get("parent_asin"); t = r.get("title")
                if not pid or not t or pid in seen: continue
                seen.add(pid); ids.append(pid); titles.append(t)
                if len(ids) >= a.cap: break
        df = pd.DataFrame({"product_id": ids, "product_title": titles})
        df["ntitle"] = df["product_title"].map(norm)
        df = df[df.ntitle.str.len() > 0].reset_index(drop=True)
        df.to_parquet(catf)
        print(f"{a.dom}: streamed {n} rows -> {len(df)} unique-title products")

    if a.motifs_from:
        fixed = sorted({json.loads(l)["query"] for l in open(a.motifs_from)
                        if json.loads(l).get("family") == "A_entity"})
        # principled clean-up: keep only the strict animal/plant motifs (drop WANDS'
        # food/abstract entries like queen, giant, bulb, noodle, sauce that are not
        # decorative motifs and become product types in other domains). is_motif +
        # BLOCK are the same domain-general predicate used for auto-mining.
        motifs = [w for w in fixed if is_motif(w)]
        print(f"fixed motif set from {a.motifs_from}: {len(fixed)} WANDS -> "
              f"{len(motifs)} after strict animal/plant filter")
    else:
        tok = collections.Counter()
        for t in df["ntitle"]:
            for w in set(words(t)): tok[w] += 1
        motifs = [w for w in tok if tok[w] >= a.min_tok and is_motif(w)]
        motifs = sorted(motifs, key=lambda w: -tok[w])
    titles = df["ntitle"].tolist(); pids = df["product_id"].tolist()
    rows = []
    for w in motifs:
        rel = [pids[i] for i in range(len(pids)) if re.search(rf"\b{re.escape(w)}\b", titles[i])]
        if len(rel) < a.min_rel: continue
        rows.append(dict(query=w, family="A_entity", known_slots={},
                         relevant_pids=rel, n_relevant=len(rel), n_distinct_class=-1))
    with open(os.path.join(outdir, "probe.jsonl"), "w") as fh:
        for r in rows: fh.write(json.dumps(r) + "\n")
    man = dict(domain=a.dom, category=a.category, catalog=len(df), n_motifs=len(rows),
               judgments=sum(r["n_relevant"] for r in rows),
               sample_motifs=[r["query"] for r in rows[:30]])
    json.dump(man, open(os.path.join(outdir, "manifest.json"), "w"), indent=2)
    print(json.dumps(man, indent=2))

if __name__ == "__main__":
    main()
