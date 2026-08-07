#!/usr/bin/env python
"""R1.1 step 2: build the stratified motif sample the author adjudicates.

Reads the LLM pre-labels (results/r1_1/typeless_labels.jsonl) and writes ONE CSV the author
fills in (a human_label column), plus an INSTRUCTIONS file. Sampling (seed 13):
  - ALL 55 WANDS motifs (the headline instrument; small enough to adjudicate fully).
  - ESCI: a stratified sample = every motif the LLM flagged genuine-product-type or ambiguous
    (the candidate impurities we most need a human to confirm), capped per stratum, PLUS a random
    sample of type-less-predicted ESCI motifs so we can estimate agreement + the type-less purity
    of the full probe. Each row carries a `stratum` and a `random_flag` (1 = drawn at random for
    the agreement/purity estimate; 0 = targeted impurity candidate) so the stats use the right rows.

After the author fills `human_label` (type-less / genuine-product-type / ambiguous), run
score_adjudication.py to compute LLM-human agreement, the confirmed type-less purity with a CI,
and the verified-clean subset for re-reporting the headline routing/harm.

Run: ./.venv/bin/python scripts/build_adjudication_sample.py
"""
import os, json, csv
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
OUTDIR = os.path.join(RES, "r1_1")
RNG = np.random.default_rng(13)

CAP_ESCI_TYPELESS = 45   # random draw for agreement/purity
CAP_GENUINE = 35         # targeted (all, capped)
CAP_AMBIG = 35           # targeted (all, capped)


def main():
    rows = [json.loads(l) for l in open(os.path.join(OUTDIR, "typeless_labels.jsonl"))]
    wands = [r for r in rows if r["source"] == "WANDS"]
    esci = [r for r in rows if r["source"] == "ESCI"]

    def pick(pool, label, cap, random_flag):
        cand = [r for r in pool if r["llm_label"] == label]
        idx = np.arange(len(cand))
        if len(cand) > cap:
            idx = RNG.choice(idx, size=cap, replace=False)
        return [{**cand[i], "random_flag": random_flag} for i in sorted(idx.tolist())]

    sample = []
    # ALL WANDS (mark random_flag=1 so they count toward purity/agreement)
    sample += [{**r, "random_flag": 1} for r in wands]
    # ESCI strata
    sample += pick(esci, "type-less", CAP_ESCI_TYPELESS, 1)          # random -> agreement/purity
    sample += pick(esci, "genuine-product-type", CAP_GENUINE, 0)     # targeted impurity candidates
    sample += pick(esci, "ambiguous", CAP_AMBIG, 0)                  # targeted impurity candidates
    # any parse/errors -> include for cleanup
    sample += [{**r, "random_flag": 0} for r in esci if r["llm_label"] in ("PARSE_FAIL", "ERROR")]

    # de-dup, stable order: WANDS first then ESCI, grouped by llm_label
    seen, ordered = set(), []
    for r in sample:
        k = (r["motif"], r["source"])
        if k not in seen:
            seen.add(k); ordered.append(r)

    csv_path = os.path.join(OUTDIR, "ADJUDICATION_SAMPLE.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row", "motif", "source", "llm_label", "llm_rationale",
                    "n_distinct_class", "human_label", "notes", "stratum", "random_flag"])
        for i, r in enumerate(ordered, 1):
            w.writerow([i, r["motif"], r["source"], r["llm_label"], r.get("llm_rationale", ""),
                        r.get("n_distinct_class", ""), "", "", r["llm_label"], r["random_flag"]])

    instr = f"""# R1.1 motif adjudication — fill in the `human_label` column (~30-45 min)

Open `ADJUDICATION_SAMPLE.csv` in any spreadsheet. Each row is one search term from the probe.
For each row, put ONE of these in the **human_label** column (leave `notes` for anything unclear):

- **type-less** — the term names a decorative motif / theme / animal / plant / food / concept, NOT
  a product category. A shopper wants items *featuring* it, spanning many product categories.
  (dinosaur, rose, butterfly, flamingo)
- **genuine-product-type** — the term IS an actual product category that maps to one shelf.
  (sofa, lamp, doormat, backpack)
- **ambiguous** — could reasonably be read either way, or is context-dependent.
  (queen = bed size or royalty; simple = adjective; mint = color/plant/candy)

Judge as a typical general-catalog shopper would. The `llm_label` and `llm_rationale` columns show
the model's pre-label — you are ADJUDICATING it, so feel free to disagree; disagreements are the
point. Don't overthink; go with the natural shopper reading.

Rows: {len(ordered)} total ({sum(1 for r in ordered if r['source']=='WANDS')} WANDS, {sum(1 for r in ordered if r['source']=='ESCI')} ESCI).
The `random_flag` column marks rows drawn at random (for the agreement/purity estimate) vs targeted
candidate impurities — you fill in all of them the same way; the script handles the rest.

When done, save the file and tell me. I run `scripts/score_adjudication.py`, which computes:
- LLM-vs-human agreement (Cohen's kappa) on the random rows,
- the confirmed type-less purity of the probe with a bootstrap CI,
- the verified-clean subset, and re-reports the headline routing/harm on it (R1.1).
"""
    open(os.path.join(OUTDIR, "ADJUDICATION_INSTRUCTIONS.md"), "w").write(instr)

    import collections
    by = collections.Counter((r["source"], r["llm_label"]) for r in ordered)
    print(f"wrote {csv_path}  ({len(ordered)} rows)")
    for (s, lab), c in sorted(by.items()):
        print(f"  {s:<6} {lab:<22} {c}")
    print(f"wrote {os.path.join(OUTDIR, 'ADJUDICATION_INSTRUCTIONS.md')}")


if __name__ == "__main__":
    main()
