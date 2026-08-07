#!/usr/bin/env python
"""R1.1 correction: re-classify ALL probe motifs with an EDIBLE-AWARE criterion, after the human
adjudication showed the first pass (and the WordNet noun.plant/noun.animal mining) systematically
mislabeled edible/grocery items (onion, cashew, plum, cilantro, noodle, flour...) as decorative
"type-less" motifs. Those are genuine products a shopper buys — emitting a food category for them is
correct, not spurious — so they must leave the type-less probe.

Corrected criterion (the adjudicator's implicit test — "can a shopper buy THIS THING ITSELF?"):
  type-less            = a DECORATIVE-ONLY motif you cannot buy as a product: an animal, mythical
                         creature, abstract theme, or ornament a shopper wants items FEATURING
                         (dinosaur, flamingo, butterfly, koala, cheetah, octopus).
  genuine-product-type = something bought AS the product: a food / grocery / produce / nut / herb /
                         spice / grain (onion, cashew, plum, cilantro, noodle, flour, popcorn), a
                         live plant (houseplant, seedling, shrub), or an actual product category
                         (drone, clock, sticker, feeder, loofah).
  ambiguous            = readable either way (queen, bulb, trunk, blade, needles, mice, poppies).

Validates the corrected labels against the 150 human adjudications first (must agree well), then
labels all 1,217 motifs -> results/r1_1/typeless_labels_v2.jsonl. gpt-4o-mini. Restartable.
Run: set -a; . ../GroundLM/.env; set +a; ./.venv/bin/python scripts/reclassify_edible_aware.py
"""
import os, json, csv, threading
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
DATA = os.path.join(HERE, "..", "data")
OUTDIR = os.path.join(RES, "r1_1")
WANDS_PROBE = os.path.join(DATA, "probe", "probe.jsonl")
ESCI_PROBE = os.path.join(DATA, "probe", "probe_1162.jsonl")

SYSTEM = (
    "You classify one e-commerce search term for a study of query planners. Apply this test: CAN A "
    "SHOPPER BUY THIS THING ITSELF as a catalog product? Output STRICT JSON {\"label\":\"...\"}:\n"
    "  genuine-product-type = YES, the term names something bought AS the product: any FOOD / GROCERY "
    "/ produce / fruit / vegetable / nut / herb / spice / grain (onion, cashew, plum, cilantro, "
    "noodle, flour, popcorn, chickpeas), a LIVE PLANT a shopper buys (houseplant, seedling, shrub, "
    "succulent sold as a plant), or an actual product category (drone, clock, sticker, feeder, loofah, "
    "mule shoe).\n"
    "  type-less = NO, you can only buy items FEATURING/THEMED ON it: a decorative motif, animal, "
    "mythical creature, abstract theme, or ornament (dinosaur, flamingo, butterfly, koala, cheetah, "
    "octopus, spider, dragon).\n"
    "  ambiguous = readable either way / context-dependent (queen, bulb, trunk, blade, needles, mice, "
    "poppies, husky).\n"
    "Note: a decorative animal you would NOT buy live (cheetah, koala) is type-less; an edible or a "
    "plant you DO buy (cashew, houseplant) is genuine-product-type. Judge as a real shopper would.")


def load_env():
    envp = os.path.join(HERE, "..", "..", "GroundLM", ".env")
    if os.path.exists(envp):
        for line in open(envp):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def load_motifs():
    m = [(json.loads(l)["query"], "WANDS") for l in open(WANDS_PROBE) if json.loads(l).get("family") == "A_entity"]
    m += [(json.loads(l)["query"], "ESCI") for l in open(ESCI_PROBE) if json.loads(l).get("family") == "A_entity"]
    seen, out = set(), []
    for q, s in m:
        if (q, s) not in seen:
            seen.add((q, s)); out.append((q, s))
    return out


def extract(txt):
    i = txt.find("{")
    if i < 0:
        return {}
    try:
        return json.loads(txt[i:txt.rfind("}") + 1])
    except Exception:
        return {}


def main():
    load_env()
    from openai import OpenAI
    client = OpenAI()
    valid = {"type-less", "genuine-product-type", "ambiguous"}

    def label_one(item):
        q, s = item
        try:
            r = client.chat.completions.create(model="gpt-4o-mini", temperature=0, max_tokens=20,
                messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": f'Term: "{q}"\nJSON:'}])
            t = r.choices[0].message.content
            lab = str(extract(t).get("label", "")).strip().lower()
            if lab not in valid:
                lab = "ambiguous"
            return {"motif": q, "source": s, "label_v2": lab}
        except Exception as e:
            return {"motif": q, "source": s, "label_v2": "ERROR", "err": str(e)[:120]}

    motifs = load_motifs()
    path = os.path.join(OUTDIR, "typeless_labels_v2.jsonl")
    done = {}
    if os.path.exists(path):
        for l in open(path):
            r = json.loads(l); done[(r["motif"], r["source"])] = r["label_v2"]
    todo = [m for m in motifs if (m[0], m[1]) not in done]
    print(f"motifs: {len(motifs)}; {len(done)} cached, {len(todo)} to label", flush=True)
    lock = threading.Lock()
    if todo:
        with open(path, "a") as f, ThreadPoolExecutor(max_workers=8) as ex:
            n = 0
            for fut in as_completed({ex.submit(label_one, m): m for m in todo}):
                r = fut.result()
                with lock:
                    f.write(json.dumps(r) + "\n"); f.flush(); done[(r["motif"], r["source"])] = r["label_v2"]
                n += 1
                if n % 200 == 0:
                    print(f"  {n}/{len(todo)}", flush=True)

    # VALIDATE against the 150 human adjudications
    adj = list(csv.DictReader(open(os.path.join(OUTDIR, "ADJUDICATION_SAMPLE.csv"))))
    agree = tl_agree = tl_h = 0
    conflicts = []
    for r in adj:
        h = r["human_label"].strip().lower()
        v = done.get((r["motif"], r["source"]))
        if v is None:
            continue
        agree += (h == v)
        if h == "type-less":
            tl_h += 1; tl_agree += (v == "type-less")
        if h != v:
            conflicts.append((r["source"], r["motif"], f"v2={v}", f"human={h}"))
    print(f"\nCorrected(v2) vs human on {len(adj)} adjudicated: raw agreement {agree/len(adj):.3f}; "
          f"of human-type-less, v2 keeps {tl_agree}/{tl_h} type-less")
    print("v2 vs human conflicts (sample):", conflicts[:15])

    import collections
    by = collections.Counter((s, done[(q, s)]) for q, s in motifs if (q, s) in done)
    print("\n=== corrected (edible-aware) label distribution over the full probe ===")
    for src in ["WANDS", "ESCI"]:
        tot = sum(v for (s, _), v in by.items() if s == src)
        parts = ", ".join(f"{lab} {by.get((src, lab), 0)} ({by.get((src, lab), 0)/tot:.1%})"
                          for lab in ["type-less", "genuine-product-type", "ambiguous"])
        print(f"  {src} (n={tot}): {parts}")
    print("wrote", path)


if __name__ == "__main__":
    main()
