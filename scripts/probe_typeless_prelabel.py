#!/usr/bin/env python
"""R1.1 step 1: LLM pre-label every probe motif as type-less / genuine-product-type / ambiguous.

The probe's Family-A motifs are "type-less by construction" (an entity/theme query expresses
no product type, so any emitted product_type is spurious). R1.1 asks us to VALIDATE that
construction: draw a stratified sample across the WANDS (55) and ESCI (1,162) probes, code each
motif type-less / genuine-product-type / ambiguous with human validation, report agreement, and
re-report the headline routing/harm on the verified-clean subset.

This script does the LLM pre-label pass over ALL 1,217 motifs (cheap, gpt-4o-mini). The output
seeds (a) the stratified CSV the author adjudicates (build_adjudication_sample.py) and (b) the
verified-clean subset once human labels return. The LLM label is a PRE-label, not ground truth;
the human adjudication is the validation.

Categories:
  type-less            : a decorative motif / theme / animal / plant / food / concept that names
                         NO product category; a shopper wants items featuring it, spanning many
                         categories (dinosaur, rose, butterfly).
  genuine-product-type : the word IS an actual product category that maps to a shelf (sofa, lamp,
                         rug, backpack).
  ambiguous            : could reasonably be read either way or is context-dependent
                         (queen -> bed size vs royalty; simple -> adjective; mint -> color/plant).

Restartable: caches per-motif to results/r1_1/typeless_labels.jsonl; re-runs skip done motifs.
Reads OPENAI_API_KEY from ../GroundLM/.env. Run:
  set -a; . ../GroundLM/.env; set +a
  ./.venv/bin/python scripts/probe_typeless_prelabel.py [--judge gpt-4o-mini] [--workers 8]
"""
import os, re, json, argparse, threading
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
DATA = os.path.join(HERE, "..", "data")
OUTDIR = os.path.join(RES, "r1_1")
os.makedirs(OUTDIR, exist_ok=True)
WANDS_PROBE = os.path.join(DATA, "probe", "probe.jsonl")
ESCI_PROBE = os.path.expanduser("~/Documents/qp-faith-release/data/probe_esci/probe_full.jsonl")

SYSTEM = (
    "You classify a single e-commerce search term for a study of query planners. Decide whether "
    "the term names a PRODUCT TYPE or is a themeless motif. Output STRICT JSON: "
    '{"label": "...", "rationale": "..."} with label one of exactly:\n'
    "  type-less            = a decorative motif, theme, animal, plant, food, holiday, color, or "
    "abstract concept that names NO product category. A shopper searching it wants items FEATURING "
    "or THEMED ON it, and the relevant products span many product categories (e.g. 'dinosaur' -> "
    "rugs, bedding, wall art, toys). \n"
    "  genuine-product-type = the term IS an actual product category that maps to one shelf / "
    "product class (e.g. 'sofa', 'lamp', 'backpack', 'doormat').\n"
    "  ambiguous            = could reasonably be read either way, or is context-dependent (e.g. "
    "'queen' = bed size or royalty; 'simple' = adjective; 'mint' = color, plant, or candy).\n"
    "Judge the term as a typical home-goods / general-catalog shopper would. Keep rationale to one short clause."
)


def load_env():
    envp = os.path.join(HERE, "..", "..", "GroundLM", ".env")
    if os.path.exists(envp):
        for line in open(envp):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def load_motifs():
    motifs = []
    for r in (json.loads(l) for l in open(WANDS_PROBE)):
        if r.get("family") == "A_entity":
            motifs.append({"motif": r["query"], "source": "WANDS",
                           "n_distinct_class": r.get("n_distinct_class"), "n_relevant": r.get("n_relevant")})
    if os.path.exists(ESCI_PROBE):
        for r in (json.loads(l) for l in open(ESCI_PROBE)):
            if r.get("family") == "A_entity":
                motifs.append({"motif": r["query"], "source": "ESCI",
                               "n_distinct_class": r.get("n_distinct_class"), "n_relevant": r.get("n_relevant")})
    # de-dup by (motif, source)
    seen, out = set(), []
    for m in motifs:
        k = (m["motif"], m["source"])
        if k not in seen:
            seen.add(k); out.append(m)
    return out


def extract_json(txt):
    i = txt.find("{")
    if i < 0:
        return None
    depth = 0
    for j in range(i, len(txt)):
        if txt[j] == "{":
            depth += 1
        elif txt[j] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(txt[i:j + 1])
                except Exception:
                    return None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", default="gpt-4o-mini")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    load_env()
    if not os.environ.get("OPENAI_API_KEY"):
        print("FATAL: OPENAI_API_KEY not set (source ../GroundLM/.env)"); raise SystemExit(2)
    from openai import OpenAI
    client = OpenAI()

    motifs = load_motifs()
    path = os.path.join(OUTDIR, "typeless_labels.jsonl")
    done = {}
    if os.path.exists(path):
        for l in open(path):
            try:
                r = json.loads(l); done[(r["motif"], r["source"])] = r
            except Exception:
                pass
    todo = [m for m in motifs if (m["motif"], m["source"]) not in done]
    print(f"motifs: {len(motifs)} ({sum(1 for m in motifs if m['source']=='WANDS')} WANDS, "
          f"{sum(1 for m in motifs if m['source']=='ESCI')} ESCI); {len(done)} cached, {len(todo)} to label", flush=True)

    lock = threading.Lock()
    valid = {"type-less", "genuine-product-type", "ambiguous"}

    def label_one(m):
        try:
            resp = client.chat.completions.create(
                model=args.judge, temperature=0, max_tokens=120,
                messages=[{"role": "system", "content": SYSTEM},
                          {"role": "user", "content": f'Term: "{m["motif"]}"\nJSON:'}])
            txt = resp.choices[0].message.content
            j = extract_json(txt) or {}
            lab = str(j.get("label", "")).strip().lower()
            if lab not in valid:
                lab = "ambiguous" if "ambig" in lab else ("genuine-product-type" if "genuine" in lab or "product" in lab else ("type-less" if "type" in lab else "PARSE_FAIL"))
            return {**m, "llm_label": lab, "llm_rationale": str(j.get("rationale", ""))[:200], "raw": txt[:300]}
        except Exception as e:
            return {**m, "llm_label": "ERROR", "llm_rationale": str(e)[:200], "raw": ""}

    if todo:
        with open(path, "a") as f, ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(label_one, m): m for m in todo}
            n = 0
            for fut in as_completed(futs):
                rec = fut.result()
                with lock:
                    f.write(json.dumps(rec) + "\n"); f.flush()
                    done[(rec["motif"], rec["source"])] = rec
                n += 1
                if n % 100 == 0:
                    print(f"  labeled {n}/{len(todo)}", flush=True)

    # summary
    import collections
    by = collections.Counter((r["source"], r["llm_label"]) for r in done.values())
    print("\n=== LLM pre-label distribution ===")
    for src in ["WANDS", "ESCI"]:
        tot = sum(v for (s, _), v in by.items() if s == src)
        print(f"  {src} (n={tot}):")
        for lab in ["type-less", "genuine-product-type", "ambiguous", "PARSE_FAIL", "ERROR"]:
            c = by.get((src, lab), 0)
            if c:
                print(f"    {lab:<22} {c:>4}  ({c/tot:.1%})")
    print(f"\nwrote {path}")
    print("next: ./.venv/bin/python scripts/build_adjudication_sample.py")


if __name__ == "__main__":
    main()
